"""第三轮：统一的数据流（Insects 突变 / Gas Sensor / Electricity / Insects 渐变截取段）。

每个数据流给出：
  X, y            按时间顺序的特征与标签（y = 0..K-1）
  regime          每行所属的评估阶段：留出集按它抽，保留率按它算
  boundaries      评估阶段的边界（新阶段第一行的位置）
  mem_boundaries  记忆在哪里"关闭阶段"（按阶段均衡的方法用）；通常与 boundaries 相同
  holdout_ok      哪些行允许抽进留出集（渐变区间不抽）
  holdout_k       每个（阶段, 类别）抽多少行
定义与理由见 results/phase56_prereg.md 修订 3；本文件的常量与那里一一对应。
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATASETS = ("insects", "gas", "electricity", "insects_gradual", "insects_gradual_start")

# 修订 3 的留出规则：每个（阶段, 类别）抽 min(50, ⌊可抽行数 / 4⌋) 行；不足 10 行的类别不抽，
# 该阶段的保留率只按其余类别算。第二轮的 Insects 突变流保留原规则（每类 50 行，R4 每类 20 行）。
HOLDOUT_CAP, HOLDOUT_MIN = 50, 10

OPENML_CACHE_DIR = os.path.expanduser("~/.cache/openml_streams")

# Gas Sensor Array Drift（UCI 224；OpenML 1476 与 USP 副本逐行相同，都是第 10 批排在最前）
GAS_OPENML_ID = 1476
GAS_FINGERPRINT = "2025486c25c43795bfdf14f33f392828491fc34653b5dcfbcd151d6c320a66a5"
GAS_BATCH_SIZES = [445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600]   # UCI 文档，第 1–10 批
GAS_REGIMES = [[1], [2], [3], [4, 5], [6], [7], [8, 9], [10]]              # 少于 300 行的批次与下一批合并
GAS_SHUFFLE_SEED = 0              # 批内顺序在源文件里按标签排列，用固定种子打乱；所有 run 共用同一条流
GAS_CLASS_NAMES = ["ethanol", "ethylene", "ammonia", "acetaldehyde", "acetone", "toluene"]

# Electricity（OpenML 151，NSW 电力市场，每天 48 个半小时）
ELEC_OPENML_ID = 151
ELEC_FINGERPRINT = "dc4c9a3bf1697d2d6576482ce5eac04106bdd49831fc09dc82a7fc4ab6804cd0"
ELEC_DUP_DAY = 363                # 第 363 天（行 17,424–17,471）是第 362 天（1997-05-04）的复制
ELEC_START = _dt.date(1996, 5, 7)
ELEC_FEATURES = ["day", "period", "nswprice", "nswdemand"]   # 去掉 date（按时间查表会泄露答案）与前 39% 是占位值的维州三列
ELEC_MIN_SEASON_DAYS = 30         # 不足 30 天的首尾季节并入相邻季节

# Insects incremental-gradual（不平衡版）截取段
GRADUAL_CP = 58_158               # Table 2 的 58159 按 1 起算（不平衡版的约定），渐变结束、纯 23°C 的第一行
GRADUAL_WINDOW = 3_000            # 交替区间 [55,158, 58,158)：23°C 占比 30/50/70%
GRADUAL_MARGIN = 10_000           # 10 × 最大记忆（1000 行）
GRADUAL_SLICE = (GRADUAL_CP - GRADUAL_WINDOW - GRADUAL_MARGIN, GRADUAL_CP + GRADUAL_MARGIN)   # [45,158, 68,158)


@dataclass
class Stream:
    name: str
    X: np.ndarray
    y: np.ndarray
    class_ids: list
    regime: np.ndarray
    boundaries: list
    mem_boundaries: list
    holdout_ok: np.ndarray
    holdout_k: dict
    feature_names: list = field(default_factory=list)


def quarter_rule(y: np.ndarray, regime: np.ndarray, ok: np.ndarray) -> dict:
    k = {}
    for r in np.unique(regime):
        for c in np.unique(y[regime == r]):
            n = int(((regime == r) & (y == c) & ok).sum())
            kk = min(HOLDOUT_CAP, n // 4)
            if kk >= HOLDOUT_MIN:
                k[(int(r), int(c))] = kk
    return k


def carve_stream_holdouts(s: Stream, seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """按 holdout_k 在每个（阶段, 类别）的可抽行里均匀随机抽留出集。

    循环顺序与随机数用法和第二轮的 carve_regime_holdouts 完全一致，所以 Insects 突变流的留出集不变。
    """
    rng = np.random.default_rng(seed)
    picked = []
    for r in np.unique(s.regime):
        rows_r = np.flatnonzero((s.regime == r) & s.holdout_ok)
        for c in np.unique(s.y[rows_r]):
            k = s.holdout_k.get((int(r), int(c)))
            if not k:
                continue
            rows_rc = rows_r[s.y[rows_r] == c]
            if len(rows_rc) < k:
                raise ValueError(f"{s.name}: 阶段 {r} 类别 {c} 只有 {len(rows_rc)} 行，不够留出 {k}")
            picked.append(np.sort(rng.choice(rows_rc, size=k, replace=False)))
    hold = np.sort(np.concatenate(picked)) if picked else np.zeros(0, np.int64)
    hold = hold[np.lexsort((hold, s.regime[hold]))]
    mask = np.ones(len(s.y), dtype=bool); mask[hold] = False
    return np.flatnonzero(mask), hold.astype(np.int64)


def _fingerprint(X: np.ndarray, y_str: "pd.Series") -> str:
    return hashlib.sha256(np.ascontiguousarray(X).tobytes() + y_str.astype(str).str.cat().encode()).hexdigest()


def _openml(did: int, fingerprint: str):
    """OpenML 数据集（原样）。本地缓存 ~/.cache/openml_streams/openml_<id>.npz 优先（GPU 机器离线运行时用），
    没有缓存才联网下载并写缓存；两条路都必须通过同一个内容指纹。"""
    path = os.path.join(OPENML_CACHE_DIR, f"openml_{did}.npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            A, ys, names = d["X"], d["y"], [str(n) for n in d["names"]]
    else:
        import openml  # 懒加载
        ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                         download_features_meta_data=False)
        X, y, _, names = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
        A, ys, names = X.to_numpy(float), y.astype(str).to_numpy(dtype=str), list(names)
    got = _fingerprint(A, pd.Series(ys))
    if got != fingerprint:
        raise RuntimeError(f"OpenML {did} 的内容与预注册版本不同：{got}（应为 {fingerprint}）")
    if not os.path.exists(path):
        os.makedirs(OPENML_CACHE_DIR, exist_ok=True)
        np.savez(path, X=A, y=ys, names=np.array(names, dtype=str))
    return A, ys, names


def _insects() -> Stream:
    from src.data.real_world import load_insects_stream
    s = load_insects_stream()
    per_class = {0: 50, 1: 50, 2: 50, 3: 50, 4: 20, 5: 50}          # 第二轮规则，不改
    k = {(int(r), int(c)): per_class[int(r)] for r in np.unique(s.regime) for c in np.unique(s.y[s.regime == r])}
    return Stream("insects", s.X, s.y, s.class_ids, s.regime, list(s.change_points), list(s.change_points),
                  np.ones(len(s.y), dtype=bool), k)


def _gas() -> Stream:
    A, ys, _ = _openml(GAS_OPENML_ID, GAS_FINGERPRINT)
    starts_src = np.cumsum([0, GAS_BATCH_SIZES[9]] + GAS_BATCH_SIZES[:9])[:-1]   # 源文件顺序：B10, B1, …, B9
    src_start = {10: int(starts_src[0]), **{b: int(starts_src[b]) for b in range(1, 10)}}
    rng = np.random.default_rng(GAS_SHUFFLE_SEED)
    order, batch = [], []
    for b in range(1, 11):                                                      # 按时间：B1 … B10
        rows = np.arange(src_start[b], src_start[b] + GAS_BATCH_SIZES[b - 1])
        order.append(rng.permutation(rows)); batch.append(np.full(len(rows), b))
    order = np.concatenate(order); batch = np.concatenate(batch)
    b2r = {b: r for r, bs in enumerate(GAS_REGIMES) for b in bs}
    regime = np.array([b2r[b] for b in batch], dtype=np.int64)
    class_ids = sorted(int(c) for c in np.unique(ys))
    y = np.array([class_ids.index(int(c)) for c in ys[order]], dtype=np.int64)
    X = A[order].astype(np.float32)
    bnds = [int(i) for i in np.flatnonzero(np.diff(regime)) + 1]
    ok = np.ones(len(y), dtype=bool)
    return Stream("gas", X, y, class_ids, regime, bnds, bnds, ok, quarter_rule(y, regime, ok),
                  [f"f{i}" for i in range(1, X.shape[1] + 1)])


def _season(d: _dt.date) -> str:
    return {12: "summer", 1: "summer", 2: "summer", 3: "autumn", 4: "autumn", 5: "autumn",
            6: "winter", 7: "winter", 8: "winter"}.get(d.month, "spring")


def elec_regimes(n_days: int) -> np.ndarray:
    """每天所属的季节段编号：南半球气象季节，按时间连成段；不足 30 天的首尾段并入相邻段。"""
    seasons = [_season(ELEC_START + _dt.timedelta(days=i)) for i in range(n_days)]
    seg = np.zeros(n_days, dtype=np.int64)
    for i in range(1, n_days):
        seg[i] = seg[i - 1] + (seasons[i] != seasons[i - 1])
    lengths = np.bincount(seg)
    if lengths[0] < ELEC_MIN_SEASON_DAYS:
        seg[seg == 0] = 1
    if lengths[-1] < ELEC_MIN_SEASON_DAYS:
        seg[seg == seg.max()] = seg.max() - 1
    _, seg = np.unique(seg, return_inverse=True)
    return seg.astype(np.int64)


def _electricity() -> Stream:
    A, ys, names = _openml(ELEC_OPENML_ID, ELEC_FINGERPRINT)
    keep = np.ones(len(A), dtype=bool); keep[ELEC_DUP_DAY * 48:(ELEC_DUP_DAY + 1) * 48] = False
    A, ys = A[keep], ys[keep]
    col = {n: i for i, n in enumerate(names)}
    day = A[:, col["day"]].copy()
    after = np.arange(len(A)) >= ELEC_DUP_DAY * 48
    day[after] = (day[after] - 2) % 7 + 1          # 复制日之后星期列错了一位（周日被记成 1）
    A[:, col["day"]] = day
    X = A[:, [col[f] for f in ELEC_FEATURES]].astype(np.float32)
    n_days = len(A) // 48
    regime = np.repeat(elec_regimes(n_days), 48)
    class_ids = sorted(np.unique(ys).tolist())                 # ['DOWN', 'UP']
    y = np.array([class_ids.index(c) for c in ys], dtype=np.int64)
    bnds = [int(i) for i in np.flatnonzero(np.diff(regime)) + 1]
    ok = np.ones(len(y), dtype=bool)
    return Stream("electricity", X, y, class_ids, regime, bnds, bnds, ok, quarter_rule(y, regime, ok),
                  list(ELEC_FEATURES))


def _insects_gradual(boundary: str) -> Stream:
    from src.data.real_world import _INSECTS_CSV_FEATURE_COLS, _ensure_insects_csv
    df = pd.read_csv(_ensure_insects_csv("gradual_imbalanced"), header=None,
                     names=_INSECTS_CSV_FEATURE_COLS + ["class"])
    a, b = GRADUAL_SLICE
    df = df.iloc[a:b]
    class_ids = sorted(int(c) for c in np.unique(df["class"]))
    y = np.array([class_ids.index(int(c)) for c in df["class"]], dtype=np.int64)
    X = df[_INSECTS_CSV_FEATURE_COLS].to_numpy(np.float32)
    cp = GRADUAL_CP - a                                          # 13,000
    zone = (cp - GRADUAL_WINDOW, cp)                             # [10,000, 13,000)
    regime = (np.arange(len(y)) >= cp).astype(np.int64)
    ok = np.ones(len(y), dtype=bool); ok[zone[0]:zone[1]] = False
    mem = [cp] if boundary == "end" else [zone[0]]
    name = "insects_gradual" if boundary == "end" else "insects_gradual_start"
    return Stream(name, X, y, class_ids, regime, [cp], mem, ok, quarter_rule(y, regime, ok),
                  list(_INSECTS_CSV_FEATURE_COLS))


def load_stream(name: str) -> Stream:
    if name == "insects":
        return _insects()
    if name == "gas":
        return _gas()
    if name == "electricity":
        return _electricity()
    if name == "insects_gradual":
        return _insects_gradual("end")
    if name == "insects_gradual_start":
        return _insects_gradual("start")
    raise ValueError(f"未知数据流 {name!r}；可选 {DATASETS}")
