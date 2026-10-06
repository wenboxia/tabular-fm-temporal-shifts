"""
真实世界漂移数据集加载器 (Phase 5)

提供与 `SyntheticDataset` 接口对齐的 `RealWorldDataset`，覆盖：
  - Electricity (OpenML id=151)：~45k 样本，8 特征（含 1 类别），二分类，gradual / seasonal
  - Insects abrupt_balanced (USP DS via Google Drive)：52,848 样本，33 数值特征，
    原 6 类多分类按 sex-pair 二值化（Phase 5 决策）

Prequential 协议硬要求：normalization 仅 fit on 前 200 样本，再 transform 全 segment，
严禁全段 fit（会泄漏未来统计进入 baseline 评估）。
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class RealWorldDataset:
    X: np.ndarray  # (n_samples, n_features) float32
    y: np.ndarray  # (n_samples,) int (0/1)
    drift_points: List[int] = field(default_factory=list)
    name: str = ""


# ---------------------------------------------------------------------------
# Insects abrupt_balanced
# ---------------------------------------------------------------------------

# Google Drive 直链来自 river master (`river/datasets/insects.py`).
# river 0.23 内置 URL 已 404 (USP labic.icmc.usp.br 已下线 creme/ 路径)。
_INSECTS_GDRIVE = {
    "abrupt_balanced": (
        "https://drive.google.com/uc?export=download&"
        "id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t"
    ),
}

# 首次下载后通过 sha256(open(path,'rb').read()).hexdigest() 计算并写死，
# 防止 Google Drive 静默替换文件（as of 2026-04-29）。
_INSECTS_SHA256 = {
    # as of 2026-04-29，首次下载实测；如 Google Drive 静默替换会抛错
    "abrupt_balanced": "f368a6f4b7f28ce2e9aa0a9e542e1f6924999cae3e8607f0637549798cb7b94a",
}

_INSECTS_CACHE_DIR = os.path.expanduser("~/.cache/insects_drift")
_INSECTS_CSV_FEATURE_COLS = [f"f{i}" for i in range(1, 34)]

# Phase 5 决策：按相邻 ID 配对的奇偶二值化。class IDs [2,3,4,5,11,12] 被**推断**为
# 3 物种 × 2 性别（{2,3} / {4,5} / {11,12} 是相邻整数对），每对偶数 ID → 0、奇数 ID → 1。
# ⚠️ Phase 5.5 更正：Souza 2020 与 USP 仓库均**未提供** class ID → 物种/性别的对应表
# （原文只说 6 类来自 Aedes aegypti / Aedes albopictus / Culex quinquefasciatus 的雌雄）。
# 因此这只能称 "ID 分组 (pair parity)"，不能称 sex 分类。
_INSECTS_BINARIZE_MAP = {2: 0, 4: 0, 11: 0, 3: 1, 5: 1, 12: 1}

# Phase 5.5：可选的标签方案。
#   pair_parity  —— 既有方案，全部 6 类按奇偶折叠成 2 类，保留全部 52,848 行。
#                   问题：TabPFN 在这上面能到 96–98%，只剩 2–4pp headroom，
#                   而 adapter cold-start 的固定成本就有 0.17pp ⇒ 设计上出不了正面结果。
#   pair_A_vs_B  —— 只保留 {2,3} 与 {4,5} 两组，丢弃 {11,12}；标签 = 属于哪一组。
#                   任务更难（官方点 33,240 处 TabPFN 探针 99% → 62%），代价是丢约 1/3 样本，
#                   且被丢的那对类的"消失-重现"现象也随之消失。
# ⚠️ 两者都只是 **ID 分组**，没有官方语义（见上方 _INSECTS_BINARIZE_MAP 注释）。
_INSECTS_LABEL_SCHEMES = {
    "pair_parity": {"keep": None, "positive": {3, 5, 12}},
    "pair_A_vs_B": {"keep": {2, 3, 4, 5}, "positive": {4, 5}},
}


def _binarize_insects(raw: np.ndarray, label_scheme: str) -> "tuple[np.ndarray, np.ndarray]":
    """把原始 6 类 ID 折成 0/1，并返回保留行的布尔掩码。

    Returns:
        y:    (n_kept,) int64，0/1 标签
        keep: (n,) bool，哪些原始行被保留（pair_parity 全 True）
    """
    if label_scheme not in _INSECTS_LABEL_SCHEMES:
        raise ValueError(
            f"label_scheme 必须 ∈ {sorted(_INSECTS_LABEL_SCHEMES)}，收到 {label_scheme!r}"
        )
    spec = _INSECTS_LABEL_SCHEMES[label_scheme]
    keep = (
        np.ones(len(raw), dtype=bool)
        if spec["keep"] is None
        else np.isin(raw, sorted(spec["keep"]))
    )
    y = np.isin(raw[keep], sorted(spec["positive"])).astype(np.int64)
    return y, keep

# ── 漂移坐标：官方 vs 经验推断（Phase 5.5 更正，2026-09-06）────────────────────
#
# 官方坐标（Souza et al. 2020, Table 2, "Abrupt (bal.)", 52,848 instances）。
# 漂移由捕虫器内温度变化引起（30°C → 20°C → ~35°C → ...），即 P(X|y) 变化。
_INSECTS_OFFICIAL_DRIFT_POINTS = [14_352, 19_500, 33_240, 38_682, 39_510]

# 经验推断坐标（2026-04-29 的 50-chunk P(y) 诊断产物）。
# ⚠️ 这组点不是官方漂移点：它们是**标签构成**
# 突变点，不是温度漂移点。实测（2026-09-06）：
#   - 这些点处的类条件特征偏移 |Δμ|/σ 仅 0.05–0.09（≈全流中位数 0.07），
#     而官方点处是 0.25–0.49；
#   - 12,672 与 14,256 都落在同一段连续 class 5（[12,598, 14,352) 共 1,754 条）内部，
#     不是那段单类区间的真实边界。
# 保留此常量仅为复现既有 B1+ 结果；新实验一律用官方坐标。
_INSECTS_EMPIRICAL_PY_SHIFT_POINTS = [12_672, 14_256, 17_952, 46_728, 52_008]

# 向后兼容别名（既有代码路径引用）；语义 = 经验 P(y) 变化点，非官方漂移点。
_INSECTS_DRIFT_POINTS_HINT = _INSECTS_EMPIRICAL_PY_SHIFT_POINTS

# Phase 5 Stage B re-aligned (B1+, 2026-05-01)：按**经验 P(y) 变化点**对齐的 4 段。
# ⚠️ Phase 5.5 更正：按官方坐标看，这 4 段只覆盖 2/5 个真实漂移
#   （early 含 14,352 → local 4,352；mid 含 19,500 → local 3,500；
#    late_pre / late_post 不含任何官方漂移）。
# 保留此切法仅为复现既有 60 runs；新实验用以官方坐标为中心的段（aligned_v2）。
_INSECTS_ALIGNED_BOUNDS = {
    "early":     (10_000, 15_000),  # 经验点 12,672 + 14,256；官方点 14,352（local 4,352）
    "mid":       (16_000, 21_000),  # 经验点 17,952；官方点 19,500（local 3,500）
    "late_pre":  (42_500, 47_500),  # 经验点 46,728；无官方漂移
    "late_post": (47_848, 52_848),  # 经验点 52,008；无官方漂移
}
_INSECTS_ALIGNED_SEGMENTS = list(_INSECTS_ALIGNED_BOUNDS.keys())

# Phase 5.5 新切法（`aligned_v2`）：以**官方**变点为中心的 5 个不重叠段。
# 设计要点：
#   - 每段 3000 样本，官方变点大致居中（旧 B1+ 有几段把 drift 挤在末尾，
#     漂移后只剩几百步，恢复曲线根本画不完）；
#   - 变点距段两端 ≥ 1000 步，够 ADWIN 累积窗口 + 够观察恢复；
#   - 38,682 与 39,510 只差 828 步，放同一段（`d4_double`）而不是拆开。
# 段名用 dN 前缀标出它对齐的是哪个官方变点。
_INSECTS_ALIGNED_V2_BOUNDS = {
    "d1_14352":   (12_900, 15_900),   # 官方 14,352 → local 1,452
    "d2_19500":   (18_000, 21_000),   # 官方 19,500 → local 1,500
    "d3_33240":   (31_700, 34_700),   # 官方 33,240 → local 1,540
    "d4_double":  (37_600, 40_600),   # 官方 38,682 / 39,510 → local 1,082 / 1,910
    "d0_control": (25_000, 28_000),   # 无官方变点的对照段（测误报率）
}
_INSECTS_ALIGNED_V2_SEGMENTS = list(_INSECTS_ALIGNED_V2_BOUNDS.keys())

# ⚠️ 段有效性（2026-09-06 实测）：官方变点 14,352 正好落在一段 **1,754 条连续 class 5**
# （[12,598, 14,352)）的末尾，所以任何包含它的窗口，漂移**之前**都是单一类别。
# 那里的温度漂移 P(X|y) 与标签构成变化 P(y) 完全混淆，无法区分方法是在应对哪一个，
# 且单类 context 会让 TabPFN 走常量 fallback（漂移前准确率恒为 100%）。
# 同理 d2_19500 在 pair_A_vs_B 下丢行后漂移前也退化成单类。
# 判定统一由 `_check_segment_validity()` 在加载时执行，不靠人记。
_INSECTS_V2_KNOWN_DEGENERATE = {
    ("d1_14352", "pair_parity"),
    ("d1_14352", "pair_A_vs_B"),
    ("d2_19500", "pair_A_vs_B"),
}
# 可用于漂移实验的段（两种 label_scheme 下都干净）
_INSECTS_V2_USABLE_DRIFT_SEGMENTS = ["d3_33240", "d4_double"]


def _check_segment_validity(
    y: np.ndarray, drift_points: "List[int]", segment_id: str,
    label_scheme: str, context_size: int = 200, allow_degenerate: bool = False,
) -> None:
    """漂移前后都必须有两个类别，否则这段测不出概念漂移。

    单类的漂移前区间意味着：(a) TabPFN 走常量 fallback，漂移前准确率恒为 100%；
    (b) 观察到的"漂移"其实是 P(y) 的构成变化，与温度引起的 P(X|y) 漂移无法区分。
    这类段会安静地产出漂亮但无意义的数字，所以默认直接报错。
    """
    if not drift_points:
        return
    for dp in drift_points:
        pre, post = y[context_size:dp], y[dp:]
        bad = []
        if len(pre) and len(np.unique(pre)) < 2:
            bad.append(f"漂移前 {len(pre)} 行全是类别 {int(pre[0])}")
        if len(post) and len(np.unique(post)) < 2:
            bad.append(f"漂移后 {len(post)} 行全是类别 {int(post[0])}")
        if bad:
            msg = (
                f"segment {segment_id!r} + label_scheme {label_scheme!r} 在漂移点 {dp} 处退化："
                + "；".join(bad)
                + "。这段的 P(y) 构成变化与温度漂移 P(X|y) 完全混淆，"
                  "且单类 context 会让 TabPFN 走常量 fallback，测不出概念漂移。"
                  f" 可用的漂移段：{_INSECTS_V2_USABLE_DRIFT_SEGMENTS}。"
                  " 确知后果可传 allow_degenerate=True 绕过。"
            )
            if allow_degenerate:
                print(f"[warn] {msg}")
            else:
                raise ValueError(msg)


def _ensure_insects_csv(variant: str = "abrupt_balanced") -> str:
    if variant not in _INSECTS_GDRIVE:
        raise ValueError(
            f"Unsupported Insects variant: {variant!r}; "
            f"available: {list(_INSECTS_GDRIVE)}"
        )
    os.makedirs(_INSECTS_CACHE_DIR, exist_ok=True)
    target = os.path.join(_INSECTS_CACHE_DIR, f"{variant}.csv")

    if os.path.exists(target):
        # 缓存命中：校验 sha256（首轮 PENDING 时跳过校验，仅打印实际 hash）
        with open(target, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        expected = _INSECTS_SHA256.get(variant, "PENDING")
        if expected != "PENDING" and actual != expected:
            raise RuntimeError(
                f"Cached Insects CSV checksum mismatch for {variant!r}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"  path:     {target}\n"
                f"Delete the file to re-download."
            )
        if expected == "PENDING":
            print(
                f"[insects] sha256({variant})={actual}  "
                f"(write into _INSECTS_SHA256 to enable verification)"
            )
        return target

    # Cache miss → download
    url = _INSECTS_GDRIVE[variant]
    print(f"[insects] downloading {variant} from Google Drive (~14 MB)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as r:
            data = r.read()
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        # 有些 Python 安装没有系统根证书；文件内容由下面的 sha256 校验保证，所以可以不验证证书再试一次
        print("[insects] TLS certificate check failed on this Python; retrying without it (content is hash-checked)")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
            data = r.read()
    actual = hashlib.sha256(data).hexdigest()
    expected = _INSECTS_SHA256.get(variant, "PENDING")
    if expected != "PENDING" and actual != expected:
        raise RuntimeError(f"Downloaded Insects CSV has the wrong checksum for {variant!r}: {actual} (expected {expected})")
    tmp = f"{target}.part{os.getpid()}"
    with open(tmp, "wb") as f:      # 先写临时文件再原子替换：并行的进程不会读到写了一半的文件
        f.write(data)
    os.replace(tmp, target)
    print(f"[insects] saved to {target} ({len(data)} bytes, sha256 {actual})")
    return target


def load_insects(
    segment_id: str = "start",
    size: int = 5000,
    variant: str = "abrupt_balanced",
    insects_aligned: bool = False,
    aligned_v2: bool = False,
    label_scheme: str = "pair_parity",
    allow_degenerate: bool = False,
) -> RealWorldDataset:
    """加载 Insects 二值化 segment。

    三种切法：
      - 默认 (A+)：segment_id ∈ {start, middle, end}，size 任意 ≤ N。
        ⚠️ 15 段中有 14 段不含任何漂移，只保留用于复现早期实验，勿用于新实验。
      - insects_aligned=True (B1+)：segment_id ∈ {early, mid, late_pre, late_post}。
        按**经验 P(y) 变化点**对齐，按官方坐标只覆盖 2/5 个真实漂移。保留以复现既有 60 runs。
      - aligned_v2=True (Phase 5.5)：segment_id ∈ `_INSECTS_ALIGNED_V2_SEGMENTS`。
        以**官方**变点为中心，每段 3000 样本、变点距两端 ≥ 1000 步，另含一个无漂移对照段。

    label_scheme ∈ {"pair_parity"（默认，全 6 类折叠）, "pair_A_vs_B"（只留 {2,3} vs {4,5}）}。
    非 pair_parity 时会**丢行**，因此 drift 的段内坐标必须在过滤后重新映射 —— 见下方实现。
    """
    if insects_aligned and aligned_v2:
        raise ValueError("insects_aligned 与 aligned_v2 互斥，只能选一种切法")

    csv_path = _ensure_insects_csv(variant=variant)
    cols = _INSECTS_CSV_FEATURE_COLS + ["class"]
    df = pd.read_csv(csv_path, header=None, names=cols)

    raw_full = df["class"].to_numpy()
    unknown = set(np.unique(raw_full)) - set(_INSECTS_BINARIZE_MAP.keys())
    if unknown:
        raise RuntimeError(
            f"Unexpected Insects class IDs {unknown}; "
            f"known IDs are {sorted(_INSECTS_BINARIZE_MAP)}"
        )
    X_full = df[_INSECTS_CSV_FEATURE_COLS].to_numpy(dtype=np.float32)

    # ── 选段边界 + 该段对应的漂移坐标（绝对） ─────────────────────────
    if aligned_v2:
        if segment_id not in _INSECTS_ALIGNED_V2_BOUNDS:
            raise ValueError(
                f"aligned_v2=True requires segment_id ∈ "
                f"{_INSECTS_ALIGNED_V2_SEGMENTS}, got {segment_id!r}"
            )
        seg_start, seg_end = _INSECTS_ALIGNED_V2_BOUNDS[segment_id]
        drift_abs = _INSECTS_OFFICIAL_DRIFT_POINTS      # 官方温度变点
        suffix = "v2_"
    elif insects_aligned:
        if segment_id not in _INSECTS_ALIGNED_BOUNDS:
            raise ValueError(
                f"insects_aligned=True requires segment_id ∈ "
                f"{_INSECTS_ALIGNED_SEGMENTS}, got {segment_id!r}"
            )
        seg_start, seg_end = _INSECTS_ALIGNED_BOUNDS[segment_id]
        drift_abs = _INSECTS_EMPIRICAL_PY_SHIFT_POINTS  # 复现既有 runs
        suffix = "aligned_"
    else:
        if segment_id not in {"start", "middle", "end"}:
            raise ValueError(
                f"insects_aligned=False requires segment_id ∈ "
                f"{{start, middle, end}}, got {segment_id!r}"
            )
        seg_start, seg_end = _segment_bounds(len(X_full), segment_id, size)
        drift_abs = _INSECTS_EMPIRICAL_PY_SHIFT_POINTS
        suffix = ""

    X_seg_raw = X_full[seg_start:seg_end]
    raw_seg = raw_full[seg_start:seg_end]

    # ── 二值化 + 行过滤 + 漂移坐标重映射 ──────────────────────────────
    # ⚠️ 关键正确性点：label_scheme 丢行后，段内漂移坐标必须按**保留行的累计数**
    # 重新映射，否则 oracle 触发时刻会指向错误的样本（label_scheme 与切段的交互问题）。
    y_seg, keep = _binarize_insects(raw_seg, label_scheme)
    X_seg = X_seg_raw[keep].copy()
    kept_cumsum = np.cumsum(keep)          # kept_cumsum[i] = 前 i+1 行中保留了几行
    local_drift = []
    for d in drift_abs:
        if not (seg_start < d < seg_end):
            continue
        d_local_raw = int(d - seg_start)
        d_local = int(kept_cumsum[d_local_raw - 1]) if d_local_raw > 0 else 0
        local_drift.append(d_local)

    # 过滤后段可能明显变短（例如 d2_19500 在 pair_A_vs_B 下 3000 → 1474，
    # 因为该时段以 {11,12} 为主）。**原始时间窗保持不变**，两种 label_scheme 因此
    # 覆盖同一段时间、只是任务不同，可比；但太短的段没有观察恢复的余地，直接报错。
    _MIN_SEG_ROWS = 1000
    if len(X_seg) < _MIN_SEG_ROWS:
        raise ValueError(
            f"segment {segment_id!r} 在 label_scheme={label_scheme!r} 下只剩 "
            f"{len(X_seg)} 行（原始窗 {seg_end - seg_start}），少于 {_MIN_SEG_ROWS}，"
            "不足以观察漂移后的恢复。请换段或换 label_scheme。"
        )
    if local_drift and min(local_drift) < 200:
        raise ValueError(
            f"segment {segment_id!r} 在 label_scheme={label_scheme!r} 下漂移点 "
            f"{local_drift} 距段首不足 200（context_size），漂移会落在 warm-up 里。"
        )

    _check_segment_validity(
        y_seg, local_drift, segment_id, label_scheme,
        allow_degenerate=allow_degenerate,
    )

    X_seg = _prequential_normalize(X_seg, fit_size=200)

    scheme_tag = "" if label_scheme == "pair_parity" else f"{label_scheme}_"
    return RealWorldDataset(
        X=X_seg,
        y=y_seg,
        drift_points=local_drift,
        name=f"insects_{variant}_{scheme_tag}{suffix}{segment_id}",
    )


@dataclass
class InsectsStream:
    """完整的 Insects 数据流（Round 2）：原生 6 类，不切段、不分组。"""

    X: np.ndarray                 # (n, 33) float32，未归一化
    y: np.ndarray                 # (n,) int64，0..5
    class_ids: "list[int]"        # 原始类别 ID，按升序；y == i 表示 class_ids[i]
    change_points: "list[int]"    # 官方变点（绝对行号）
    regime: np.ndarray            # (n,) int64，每行所属的官方阶段 0..len(change_points)
    variant: str


def load_insects_stream(variant: str = "abrupt_balanced") -> InsectsStream:
    """加载完整 Insects 流，原生多分类标签。

    官方变点来自 Souza et al. 2020 Table 2；阶段 = 相邻变点之间的区间。
    论文 §5：abrupt 流的 R0 在 30 °C、R1 在 20 °C、R2 约 35 °C，其余三段温度未给出。
    """
    if variant != "abrupt_balanced":
        raise ValueError("Round 2 只预注册了 abrupt_balanced")
    csv_path = _ensure_insects_csv(variant=variant)
    cols = _INSECTS_CSV_FEATURE_COLS + ["class"]
    df = pd.read_csv(csv_path, header=None, names=cols)
    raw = df["class"].to_numpy()
    class_ids = sorted(int(c) for c in np.unique(raw))
    lut = {c: i for i, c in enumerate(class_ids)}
    y = np.array([lut[int(c)] for c in raw], dtype=np.int64)
    X = df[_INSECTS_CSV_FEATURE_COLS].to_numpy(dtype=np.float32)
    cps = list(_INSECTS_OFFICIAL_DRIFT_POINTS)
    regime = np.searchsorted(np.asarray(cps), np.arange(len(y)), side="right").astype(np.int64)
    return InsectsStream(X=X, y=y, class_ids=class_ids, change_points=cps,
                         regime=regime, variant=variant)


# ---------------------------------------------------------------------------
# Electricity (OpenML 151)
# ---------------------------------------------------------------------------


def load_electricity(segment_id: str = "start", size: int = 5000) -> RealWorldDataset:
    """加载 Electricity 二分类 segment。

    OpenML id=151，N=45,312，8 features (1 类别 day-of-week + 7 数值)。
    时序按 (date, period) 已排序，不重排。target = class ∈ {UP, DOWN} → {0, 1}.

    缺失值策略：drop rows（OpenML 151 的原版无缺失，dropna 是 defensive no-op）。
    """
    import openml  # 懒加载，避免合成实验路径强依赖

    print("[electricity] fetching OpenML 151 (cached at ~/.openml/ if previously downloaded)...")
    ds = openml.datasets.get_dataset(
        151, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    X_df, y_series, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )

    # drop missing rows (defensive)
    n_before = len(X_df)
    valid = X_df.notna().all(axis=1) & y_series.notna()
    X_df = X_df.loc[valid].reset_index(drop=True)
    y_series = y_series.loc[valid].reset_index(drop=True)
    if len(X_df) < n_before:
        print(f"[electricity] dropped {n_before - len(X_df)} rows with NaN")

    # 类别特征 one-hot；数值列直通
    X_encoded = pd.get_dummies(X_df, drop_first=False)
    X_full = X_encoded.to_numpy(dtype=np.float32)
    # target: class 是字符串 "UP"/"DOWN" → 0/1 (UP=1)
    y_full = (y_series.astype(str).str.upper() == "UP").to_numpy(dtype=np.int64)

    X_seg, y_seg = take_segment(X_full, y_full, segment_id=segment_id, size=size)
    X_seg = _prequential_normalize(X_seg, fit_size=200)

    # Electricity 漂移是 gradual/seasonal，无 crisp drift points
    return RealWorldDataset(
        X=X_seg, y=y_seg, drift_points=[], name=f"electricity_{segment_id}"
    )


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _segment_bounds(n: int, segment_id: str, size: int) -> tuple[int, int]:
    if size > n:
        raise ValueError(f"size {size} exceeds dataset length {n}")
    if segment_id == "start":
        return 0, size
    if segment_id == "end":
        return n - size, n
    if segment_id == "middle":
        mid = n // 2
        return mid - size // 2, mid - size // 2 + size
    raise ValueError(
        f"segment_id must be one of {{start, middle, end}}, got {segment_id!r}"
    )


def take_segment(
    X: np.ndarray, y: np.ndarray, segment_id: str = "start", size: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous 时序切片，保持原顺序，不 shuffle。"""
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    lo, hi = _segment_bounds(len(X), segment_id, size)
    return X[lo:hi].copy(), y[lo:hi].copy()


def _prequential_normalize(X: np.ndarray, fit_size: int = 200) -> np.ndarray:
    """StandardScaler fit on 前 fit_size 样本，transform 全段。Prequential 协议硬要求。"""
    if len(X) < fit_size:
        raise ValueError(
            f"segment too short ({len(X)}) for fit_size={fit_size}; "
            f"need at least {fit_size} samples for the initial-context fit"
        )
    scaler = StandardScaler()
    scaler.fit(X[:fit_size])
    out = scaler.transform(X).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def load_real_world(
    name: str, segment_id: str = "start", size: int = 5000,
    insects_aligned: bool = False, aligned_v2: bool = False,
    label_scheme: str = "pair_parity", allow_degenerate: bool = False, **kwargs,
) -> RealWorldDataset:
    """
    name ∈ {"electricity", "insects"};
    segment_id ∈ {"start", "middle", "end"}，或 insects_aligned=True 时
                  {"early", "mid", "late_pre", "late_post"}，
                  或 aligned_v2=True 时 `_INSECTS_ALIGNED_V2_SEGMENTS`。

    Insects 当前固定 variant="abrupt_balanced"（Phase 5 §决策），可由 kwargs 传入覆盖。
    insects_aligned / aligned_v2 / label_scheme 仅对 Insects 生效，Electricity 忽略。
    """
    if name == "electricity":
        return load_electricity(segment_id=segment_id, size=size)
    if name == "insects":
        variant = kwargs.pop("variant", "abrupt_balanced")
        return load_insects(
            segment_id=segment_id, size=size, variant=variant,
            insects_aligned=insects_aligned, aligned_v2=aligned_v2,
            label_scheme=label_scheme, allow_degenerate=allow_degenerate,
        )
    raise ValueError(f"unknown real-world dataset {name!r}")
