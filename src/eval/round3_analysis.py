"""第三轮分析用的纯函数（载入、一致性检查、统计、标签重放、过渡区指标、--quick 快检）。

此模块属于第三轮（results/phase56_prereg.md 的 Amendment 3 及其后的 Note）的分析部分，
只读已有的 npz，不调用 TabPFN，不产生任何新的预测结果。

使用流程（scripts/analyze_round3.py 与 scripts/plot_round3.py 调用）：
    runs, probs = load_runs([round2_dir, round3_dir], n_estimators=4)      # 键 (dataset, method, seed)
    probs += check_consistency(runs)
    m = run_metrics(runs[("gas", "dual200_rb", 0)])                         # ADAPT / RET / RET_Rj / FGT / 过渡区指标
    pr = paired(m_ours, m_kdd); ni = noninferiority(a, b); adj = holm(ps)
    rep = replay_run(runs[("gas", "dual200_rb", 0)], "dual200_rb")          # 标签重放，断言 digest 与 npz 一致
    quick_check(runs[("gas", "dual200_rb", 0)])                             # --quick 的逐文件检查
    bl = baselines_full(runs[("gas", "sw200", 0)])                          # 预注册口径与按批揭示口径的平凡基线
    left = regime_leftover(rep, d); cov = class_coverage(rep, d)            # 阶段结束后记忆里还剩什么（只用重放）

约定："ours" = dual{200,1000}_rb，"KDD" = dual{200,1000}，"sw" = sw{200,1000}；ADWIN 版以 _adwin 结尾。
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np

from src.eval.round2_metrics import (adapt, balanced_accuracy, fgt_per_regime, regime_end, ret, ret_per_regime,
                                     single_class_run_mask)

# ------------------------------------------------------------------ 预注册的数据集与方法

_BASE = ["sw200", "sw1000", "dual200", "dual1000", "dual200_rb", "dual1000_rb"]
DATASET_METHODS = {
    "insects": list(_BASE),
    "gas": list(_BASE),
    "electricity": list(_BASE),
    "insects_gradual": _BASE + ["dual200_rb_adwin", "dual1000_rb_adwin"],
    "insects_gradual_start": ["dual200_rb", "dual1000_rb"],
}
ROUND2_INSECTS = ("sw200", "sw1000", "dual1000", "dual1000_rb")      # 这四个来自第二轮
HOLM_DATASETS = ("insects", "gas", "electricity", "insects_gradual")  # Holm 家族：4 数据集 × 2 规模
SIZES = (200, 1000)
GRADUAL_DATASETS = ("insects_gradual", "insects_gradual_start")
GRADUAL_ZONE = [[9750, 12750]]
NONINF_MARGIN = 0.005                      # ADAPT 非劣性边界 −0.5 pp（同第二轮）
_METHOD_RE = re.compile(r"^(sw|dual)(\d+)(_rb)?(_adwin)?$")


def parse_method(m: str):
    """'dual200_rb_adwin' -> ('ours', 200, True)；角色 ∈ {'sw', 'kdd', 'ours'}。"""
    g = _METHOD_RE.match(m)
    if not g:
        raise ValueError(f"无法解析方法名 {m!r}")
    kind, size, rb, adw = g.groups()
    role = "sw" if kind == "sw" else ("ours" if rb else "kdd")
    return role, int(size), bool(adw)


def stem_prefix(dataset: str) -> str:
    return "" if dataset == "insects" else f"{dataset}_"


# ------------------------------------------------------------------ 载入与一致性

def load_runs(dirs, n_estimators: int = 4, stride: int = 50, tag: str = "", keep=None, tags=None):
    """载入若干目录里的 r2_*.npz。返回 (runs, problems)，runs 的键是 (dataset, method, seed)。

    dataset 取 npz 的 dataset 字段，没有该字段（第二轮文件）视为 insects；带 'calib' 的文件跳过。
    keep：要保留的 (dataset, method) 集合（None = 全部）。后面的目录覆盖前面的同键文件，并在 problems 里记一条。
    tags：与 dirs 等长的文件名后缀列表（第二轮目录永远是 ''），默认全部用 tag。
    """
    tags = list(tags) if tags is not None else [tag] * len(dirs)
    runs, problems, src = {}, [], {}
    for dir_, tg in zip(dirs, tags):
        if not dir_ or not os.path.isdir(dir_):
            continue
        pat = re.compile(rf"^r2_(.+)_n{n_estimators}_S{stride}_seed(\d+){re.escape(tg)}\.npz$")
        for f in sorted(glob.glob(os.path.join(dir_, f"r2_*_n{n_estimators}_S{stride}_seed*{tg}.npz"))):
            base = os.path.basename(f)
            g = pat.match(base)
            if not g or "calib" in base:
                continue
            with np.load(f, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            dataset = str(d["dataset"][0]) if "dataset" in d else "insects"
            method = str(d["method"][0]) if "method" in d else g.group(1)
            seed = int(d["seed"][0]) if "seed" in d else int(g.group(2))
            if g.group(1) != stem_prefix(dataset) + method or int(g.group(2)) != seed:
                problems.append(f"{base}: 文件名与内容不符（dataset={dataset}, method={method}, seed={seed}）")
                continue
            if keep is not None and (dataset, method) not in keep:
                continue
            key = (dataset, method, seed)
            if key in runs:
                problems.append(f"{key} 在 {src[key]} 与 {dir_} 里都有，用后者")
            runs[key] = d
            src[key] = dir_
    return runs, problems


def all_pairs():
    return {(ds, m) for ds, ms in DATASET_METHODS.items() for m in ms}


def _mem_boundaries(d):
    return d["mem_boundaries"] if "mem_boundaries" in d else d["boundaries"]


def _zone(d):
    return np.asarray(d["zone"]).reshape(-1, 2) if "zone" in d else np.zeros((0, 2), dtype=np.int64)


def effective_rows(d) -> int:
    msr = int(d["max_stream_rows"][0]) if "max_stream_rows" in d else -1
    n = len(d["y"])
    return n if msr < 0 else min(n, msr)


def check_consistency(runs):
    """同一 (dataset, seed) 的所有方法必须有相同的 y、hold_idx、stream_idx、boundaries（与 mem_boundaries、zone）；
    同一数据集内 n_estimators / 截断长度一致；insects 合并两轮时允许 commit / device 不同，其余数据集不允许；
    insects_gradual_start 与 insects_gradual 按 seed 跨数据集比较 y / hold_idx / stream_idx / boundaries / zone / commit。
    另外：leak_count = 0、上下文不超预算、完整运行的 pred[start:] 没有 -1。返回问题列表（空 = 通过）。"""
    problems = []
    groups = {}
    for (ds, m, s), d in runs.items():
        groups.setdefault((ds, s), []).append((m, d))
    for (ds, s), lst in sorted(groups.items()):
        lst = sorted(lst, key=lambda x: x[0])
        m0, d0 = lst[0]
        for m, d in lst[1:]:
            for k in ("y", "hold_idx", "stream_idx", "boundaries"):
                if not np.array_equal(d[k], d0[k]):
                    problems.append(f"{ds} seed {s}: {m} 与 {m0} 的 {k} 不同")
            if not np.array_equal(_zone(d), _zone(d0)):
                problems.append(f"{ds} seed {s}: {m} 与 {m0} 的 zone 不同")
            # 记忆边界允许在 insects_gradual_start 里不同，但那是另一个数据集名，所以同组内应相同
            if not np.array_equal(_mem_boundaries(d), _mem_boundaries(d0)):
                problems.append(f"{ds} seed {s}: {m} 与 {m0} 的 mem_boundaries 不同")
    by_ds = {}
    for (ds, m, s), d in runs.items():
        by_ds.setdefault(ds, []).append((m, s, d))
    for ds, lst in sorted(by_ds.items()):
        cfg = {(int(d["n_estimators"][0]), int(d["max_stream_rows"][0])) for _, _, d in lst}
        if len(cfg) > 1:
            problems.append(f"{ds}: n_estimators / 截断长度不一致 {sorted(cfg)}")
        if ds != "insects":
            ids = {(str(d["commit"][0]), str(d["device"][0])) for _, _, d in lst}
            if len(ids) > 1:
                problems.append(f"{ds}: 代码版本 / 设备不一致 {sorted(ids)}")
    # 渐变段 (b) 臂（insects_gradual_start）与 (a) 臂（insects_gradual）按 seed 配对：流、留出集、zone 必须相同
    # （mem_boundaries 本来就该不同，不比）
    for (ds, m, s), d in sorted(runs.items()):
        if ds != "insects_gradual_start":
            continue
        ref = next((runs[k] for k in sorted(runs) if k[0] == "insects_gradual" and k[2] == s), None)
        if ref is None:
            continue
        for k in ("y", "hold_idx", "stream_idx", "boundaries", "regime_hold"):
            if not np.array_equal(d[k], ref[k]):
                problems.append(f"insects_gradual_start/{m} seed {s} 与 insects_gradual 的 {k} 不同")
        if not np.array_equal(_zone(d), _zone(ref)):
            problems.append(f"insects_gradual_start/{m} seed {s} 与 insects_gradual 的 zone 不同")
        if str(d["commit"][0]) != str(ref["commit"][0]):
            problems.append(f"insects_gradual_start/{m} seed {s} 与 insects_gradual 的 commit 不同")
    for (ds, m, s), d in sorted(runs.items()):
        if int(d["leak_count"][0]) != 0:
            problems.append(f"{ds}/{m} seed {s}: leak_count = {int(d['leak_count'][0])}")
        if int(d["max_ctx"][0]) > int(d["budget"][0]):
            problems.append(f"{ds}/{m} seed {s}: 上下文 {int(d['max_ctx'][0])} 超预算 {int(d['budget'][0])}")
        start, n_end = int(d["start"][0]), effective_rows(d)
        if (d["pred"][start:n_end] < 0).any():
            problems.append(f"{ds}/{m} seed {s}: pred[{start}:{n_end}] 里有 {(d['pred'][start:n_end] < 0).sum()} 个 -1（运行不完整）")
    return problems


def find_missing(runs, seeds, dataset_methods=None):
    """返回 {(dataset, method): [缺失的 seed]}，只列有缺的。"""
    dm = dataset_methods or DATASET_METHODS
    out = {}
    for ds, ms in dm.items():
        for m in ms:
            miss = [s for s in seeds if (ds, m, s) not in runs]
            if miss:
                out[(ds, m)] = miss
    return out


# ------------------------------------------------------------------ 指标

def zone_metrics(d, rows: int = 500) -> dict:
    """过渡区（npz 的 zone，取第一个区间 [a, b)）上的平衡准确率：整体、三等分各段、区间开始后 rows 行。"""
    z = _zone(d)
    if len(z) == 0:
        return {}
    a, b = int(z[0, 0]), int(z[0, 1])
    y, pred = d["y"], d["pred"]

    def ba(lo, hi):
        sl = slice(lo, hi)
        m = pred[sl] >= 0
        return balanced_accuracy(y[sl][m], pred[sl][m]) if m.any() else float("nan")

    cuts = [a + (b - a) * k // 3 for k in range(4)]
    out = {"ZONE": ba(a, b), "ZONE_500": ba(a, a + rows)}
    for k in range(3):
        out[f"ZONE_T{k + 1}"] = ba(cuts[k], cuts[k + 1])
    return out


def run_metrics(d) -> dict:
    """一个文件的全部指标：ADAPT、ADAPT_masked、RET、RET_Rj、FGT（各阶段平均）、FGT_Rj、过渡区指标。"""
    s = dict(ADAPT=adapt(d), ADAPT_masked=adapt(d, masked=True), RET=ret(d))
    for j, v in ret_per_regime(d).items():
        s[f"RET_R{j}"] = v
    fg = fgt_per_regime(d)
    s["FGT"] = float(np.mean(list(fg.values()))) if fg else float("nan")
    for j, v in fg.items():
        s[f"FGT_R{j}"] = v
    s.update(zone_metrics(d))
    return s


def baseline_acc(d) -> "tuple[float, float]":
    """不用 TabPFN 的两个基线（预注册口径）：No-Change（上一行的标签）与被预测行之前 200 行的多数类，
    在该文件已预测的行上的平衡准确率。**注意**：逐行取"上一行"，批内尚未揭示的标签也被用到了，偏乐观。
    与 TabPFN 信息相同的口径见 baselines_full 的 nc_batch / maj_batch。"""
    y = d["y"]
    k = int(y.max()) + 1
    done = np.flatnonzero(d["pred"] >= 0)
    nochange = y[done - 1]
    maj = np.array([np.bincount(y[max(0, t - 200):t], minlength=k).argmax() for t in done])
    return balanced_accuracy(y[done], nochange), balanced_accuracy(y[done], maj)


def run_batches(d) -> "list[tuple[int, int]]":
    """按 run_stream 的规则重建批次 [(a, b), ...]：从 start 起每 stride 行一批，批不跨越 boundaries ∪ mem_boundaries
    的任何一个边界，截断到有效行数。"""
    n = effective_rows(d)
    start, stride = int(d["start"][0]), int(d["stride"][0])
    cuts = sorted({int(b) for b in d["boundaries"] if 0 < b < n} | {int(b) for b in _mem_boundaries(d) if 0 < b < n})
    t, out = start, []
    while t < n:
        nxt = [b for b in cuts if b > t]
        end = min(t + stride, n, nxt[0] if nxt else n)
        out.append((t, end))
        t = end
    return out


def baselines_full(d, window: int = 200) -> dict:
    """两种口径的平凡基线（流上已预测行；平衡准确率，另给普通准确率）。

    预注册口径（逐行）：nc_row = 上一行的标签，maj_row = 被预测行之前 window 行的多数类。批内尚未揭示的标签也被用到，偏乐观。
    按批揭示口径（看过第三轮结果之后补的描述性指标，与 TabPFN 信息相同）：nc_batch = 当前批开始前最后一个已揭示的标签，
    maj_batch = 当前批开始前 window 个已揭示行的多数类（平票取类别编号最小者）；批次由 run_batches 重建。
    nc_row_plain = 已预测行里"标签与上一行相同"的比例，也就是逐行 No-Change 的普通准确率（标签自相关的度量）。"""
    y = d["y"]
    k = int(y.max()) + 1
    n = len(y)
    done = np.flatnonzero(d["pred"] >= 0)
    nc_b, mj_b = np.full(n, -1), np.full(n, -1)
    for a, b in run_batches(d):
        nc_b[a:b] = y[a - 1]
        mj_b[a:b] = np.bincount(y[max(0, a - window):a], minlength=k).argmax()
    assert (nc_b[done] >= 0).all(), "有已预测的行不在重建的批次里"
    nc_r = y[done - 1]
    mj_r = np.array([np.bincount(y[max(0, t - window):t], minlength=k).argmax() for t in done])
    yd = y[done]
    return dict(nc_row=balanced_accuracy(yd, nc_r), maj_row=balanced_accuracy(yd, mj_r),
                nc_batch=balanced_accuracy(yd, nc_b[done]), maj_batch=balanced_accuracy(yd, mj_b[done]),
                nc_row_plain=float((yd == nc_r).mean()), nc_batch_plain=float((yd == nc_b[done]).mean()),
                n_pred=int(len(done)), n_batches=len(run_batches(d)))


def mask_removed_share(d, denominator: str = "pred") -> float:
    """ADAPT (masked) 去掉的行的比例（单类连续段 ≥ 30 行的掩码，定义见 round2_metrics）。
    denominator='pred'（默认）：占已预测行的比例，因为掩码作用在已预测行上；'stream'：占整条流（含前 start 行没有预测的行）。"""
    mask = single_class_run_mask(d["y"])
    if denominator == "stream":
        return float(mask.mean())
    assert denominator == "pred", denominator
    done = d["pred"] >= 0
    return float((mask & done).sum() / done.sum())


def stream_acc_per_regime(d) -> dict:
    """每个评估阶段 j（regime == j 的流行）内已预测行的平衡准确率；没有已预测行的阶段为 nan。"""
    reg, y, pred = d["regime"], d["y"], d["pred"]
    out = {}
    for j in range(int(reg.max()) + 1):
        m = (reg == j) & (pred >= 0)
        out[j] = balanced_accuracy(y[m], pred[m]) if m.any() else float("nan")
    return out


def holdout_class_recall(d, j: int) -> np.ndarray:
    """阶段 j 的留出集在 j 结束之后各检查点上的逐类召回（对检查点取平均）；留出集里没有的类为 nan。长度 = 类别数。"""
    k = int(max(d["y"].max(), d["y_hold"].max())) + 1
    sel = d["regime_hold"] == j
    yh = d["y_hold"][sel]
    end = regime_end(d, j)
    rows = []
    for c, t in enumerate(d["ckpt_t"]):
        pr = d["ckpt_pred"][c, sel]
        if t > end and (pr >= 0).all():
            rows.append([float((pr[yh == q] == q).mean()) if (yh == q).any() else float("nan") for q in range(k)])
    if not rows:
        return np.full(k, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # 留出集里没有的类整列是 nan
        return np.nanmean(np.array(rows), axis=0)


def first_diff_row(da, db) -> "int | None":
    """两个运行的预测第一次不同的流内行号（只比两边都已预测的行）；完全相同返回 None。"""
    both = (da["pred"] >= 0) & (db["pred"] >= 0)
    diff = np.flatnonzero(both & (da["pred"] != db["pred"]))
    return int(diff[0]) if len(diff) else None


def class_share(d) -> np.ndarray:
    """整条流（npz 的 y）各类所占比例。"""
    return np.bincount(d["y"]) / len(d["y"])


# ------------------------------------------------------------------ 统计

def finite_pairs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def mean_sd_ci(v) -> dict:
    """seed 间的均值、SD（ddof=1）、95% t 区间。"""
    from scipy import stats
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"), lo=float("nan"), hi=float("nan"))
    mean = float(v.mean())
    if n < 2:
        return dict(n=n, mean=mean, sd=float("nan"), lo=float("nan"), hi=float("nan"))
    sd = float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return dict(n=n, mean=mean, sd=sd, lo=mean - h, hi=mean + h)


def paired(a, b) -> dict:
    """a − b 的配对差：均值、SD、95% 双侧 CI、双侧配对 t 检验 p、差值为正的 seed 数。差值全相等（SD = 0）时 p = nan。"""
    from scipy import stats
    a, b = finite_pairs(a, b)
    d = a - b
    n = len(d)
    nan = float("nan")
    out = dict(n=n, mean=float(d.mean()) if n else nan, sd=nan, lo=nan, hi=nan, p=nan, npos=int((d > 0).sum()))
    if n < 2:
        return out
    out["sd"] = float(d.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * out["sd"] / np.sqrt(n))
    out["lo"], out["hi"] = out["mean"] - h, out["mean"] + h
    if out["sd"] > 0:
        out["p"] = float(stats.ttest_rel(a, b).pvalue)
    return out


def noninferiority(a, b, margin: float = NONINF_MARGIN) -> dict:
    """ADAPT 非劣性：d = a − b，单侧检验 d > −margin（ttest_1samp(d + margin, 0, alternative='greater')）
    与单侧 95% 下界 mean − t_{0.95} · SD/√n。检验本身与第二轮 H2 相同；第二轮 H2 还取原始版与 masked 版中较大的 p
    并加"至少 9/10 个 seed"的符号规则，这里不做（调用方对两种版本分别报告）。"""
    from scipy import stats
    a, b = finite_pairs(a, b)
    d = a - b
    n = len(d)
    nan = float("nan")
    out = dict(n=n, mean=float(d.mean()) if n else nan, p=nan, lower=nan, npos=int((d > -margin).sum()))
    if n >= 2 and d.std(ddof=1) > 0:
        out["p"] = float(stats.ttest_1samp(d + margin, 0, alternative="greater").pvalue)
        out["lower"] = float(d.mean() - stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return out


def holm(ps) -> np.ndarray:
    """Holm 校正（逐步下降，单调化）。nan 的 p 在排序时按 1 处理（不能拒绝），返回里仍是 nan；家族大小 = len(ps)。"""
    ps = np.asarray(ps, float)
    m = len(ps)
    q = np.where(np.isfinite(ps), ps, 1.0)
    order = np.argsort(q, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * q[i]))
        adj[i] = run
    adj[~np.isfinite(ps)] = np.nan
    return adj


def spearman(x, y) -> "tuple[float, float, int]":
    """(rho, p, n)；n < 3 或任一列为常数时为 nan。"""
    from scipy import stats
    x, y = finite_pairs(x, y)
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), n
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue), n


# ------------------------------------------------------------------ 阶段差异（探索性）

def stage_auc(X: np.ndarray, regime: np.ndarray, n_splits: int = 5, seed: int = 0, max_iter: int = 500) -> dict:
    """对每个阶段 j（最后一个阶段除外）：逻辑回归区分"阶段 j 的行"与"之后所有阶段的行"，
    5 折分层交叉验证 ROC-AUC 的均值。任一类少于 n_splits 行时为 nan。"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    out = {}
    for j in range(int(regime.max())):
        later = regime > j
        cur = regime == j
        sel = cur | later
        yy = cur[sel].astype(int)
        if yy.sum() < n_splits or (1 - yy).sum() < n_splits:
            out[j] = float("nan")
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sc = cross_val_score(LogisticRegression(max_iter=max_iter), X[sel], yy, cv=cv, scoring="roc_auc")
        out[j] = float(sc.mean())
    return out


# ------------------------------------------------------------------ 标签重放（不跑模型）

class _ReplayDetector:
    """重放 ADWIN 方法：runner 每个已预测行恰好调用一次 update；在 npz 记录的关闭位置（报警行 + 1）返回 True。"""

    def __init__(self, closures, start: int):
        self.closures = {int(c) for c in closures}
        self.row = int(start)

    def update(self, _err) -> bool:
        self.row += 1
        return self.row in self.closures


def replay_run(d, method: str | None = None, verify: bool = True) -> dict:
    """用真实的 run_stream 与记忆类重放一个 dual* / dual*_rb 运行（哑预测器，只用标签）。

    批次按 boundaries ∪ mem_boundaries 切；official 方法在 mem_boundaries 处 close_regime；adwin 方法在 npz 的
    closures 处 close_regime。verify=True 时，把每个检查点的 memory.digest() 与 npz 的 digests 比较（还有 closures
    与检查点时刻），不一致就抛 AssertionError——这是重放正确性的证明。

    返回 dict：ts（每批之后已揭示的行数）、comp（n_reg × T，长期池里各评估阶段的行数）、long_len（长期池大小）、
    mem_comp（n_reg × T，整个记忆即长期加短期里各评估阶段的行数）、age_med（长期池行年龄的中位数，年龄 = t − 流内下标，
    t = 已揭示行数；池为空时 nan）、final_long（流末长期池的流内下标）、
    overlap（记忆行集合 ∩ 最近 N 个已揭示行 的占比，N = 记忆预算）、full（记忆已满）、closures、n_reg。
    """
    from src.eval.stream_runner import needs_detector, run_stream
    from src.memory.context_memory import make_memory

    method = method or str(d["method"][0])
    y, regime = d["y"], d["regime"]
    n_reg = int(regime.max()) + 1
    K = len(d["class_ids"]) if "class_ids" in d else int(y.max()) + 1
    n = len(y)
    Xd = np.zeros((n, 1), np.float32)
    Xh = np.zeros((len(d["y_hold"]), 1), np.float32)
    start, stride, ckpt_every = int(d["start"][0]), int(d["stride"][0]), int(d["ckpt_every"][0])
    msr = int(d["max_stream_rows"][0]) if "max_stream_rows" in d else -1
    adwin = needs_detector(method)
    mem = make_memory(method, Xd, y, K, int(d["seed"][0]) if "seed" in d else 0)
    N = mem.max_context()

    ts, comp, long_len, overlap, full, mem_comp, age_med = [], [], [], [], [], [], []
    final = {}
    orig = mem.update

    def update(idx):
        orig(idx)
        t = int(np.max(idx)) + 1
        long_ = np.asarray(mem._long, dtype=np.int64)
        rows = mem._all()
        ts.append(t)
        comp.append(np.bincount(regime[long_], minlength=n_reg) if len(long_) else np.zeros(n_reg, np.int64))
        long_len.append(len(long_))
        mem_comp.append(np.bincount(regime[rows], minlength=n_reg) if len(rows) else np.zeros(n_reg, np.int64))
        age_med.append(float(np.median(t - long_)) if len(long_) else float("nan"))
        final["long"] = long_.copy()
        overlap.append(float((rows >= t - N).mean()) if len(rows) else float("nan"))
        full.append(t >= N)
    mem.update = update

    dummy = lambda Xc, yc, Xq: np.full((len(Xq), K), 1.0 / K)           # noqa: E731
    det = _ReplayDetector(d["closures"], start) if adwin else None
    r = run_stream(mem, dummy, Xd, y, regime, Xh, d["regime_hold"], list(d["boundaries"]), stride=stride, start=start,
                   ckpt_every=ckpt_every, segmentation="adwin" if adwin else "official", detector=det,
                   mem_boundaries=list(_mem_boundaries(d)), max_rows=None if msr < 0 else msr)
    ok = dict(
        digests=[str(x) for x in r.digests] == [str(x) for x in d["digests"]],
        closures=np.array_equal(r.closures, np.asarray(d["closures"])),
        ckpt_t=np.array_equal(r.ckpt_t, np.asarray(d["ckpt_t"])),
    )
    if verify and not all(ok.values()):
        bad = [k for k, v in ok.items() if not v]
        raise AssertionError(f"重放与 npz 不一致：{bad}（{method}）")
    return dict(ts=np.asarray(ts), comp=np.stack(comp, axis=1), long_len=np.asarray(long_len),
                mem_comp=np.stack(mem_comp, axis=1), age_med=np.asarray(age_med), final_long=final["long"],
                overlap=np.asarray(overlap), full=np.asarray(full, bool), closures=r.closures, n_reg=n_reg,
                verified=ok, ckpt_t=r.ckpt_t)


def overlap_summary(rep: dict) -> dict:
    """重合比例：所有批次的均值、记忆已满之后的均值与最小值。"""
    ov, full = rep["overlap"], rep["full"]
    return dict(mean_all=float(np.nanmean(ov)) if len(ov) else float("nan"),
                mean_full=float(np.nanmean(ov[full])) if full.any() else float("nan"),
                min_full=float(np.nanmin(ov[full])) if full.any() else float("nan"), n=int(len(ov)))


def regime_leftover(rep: dict, d) -> dict:
    """每个已结束的评估阶段 j（boundaries[j] 之前的 j）在阶段结束之后的记忆里还剩多少该阶段的行。

    对每个 j 返回 dict：lt5 = 阶段结束之后各次更新里，整个记忆（长期加短期）含 j 的行少于 5 行的比例；
    avg_long / min_long = 阶段结束之后长期池里 j 的行数的均值 / 最小值；share_end = 阶段刚结束那次更新时
    长期池里 j 的行所占比例（长期池为空时 nan）。截断运行里还没结束的阶段不出现。"""
    ts, out = rep["ts"], {}
    for j, e in enumerate(int(b) for b in d["boundaries"]):
        after = ts > e
        if not after.any():
            continue
        at = np.flatnonzero(ts == e)
        share = float(rep["comp"][j, at[0]] / rep["long_len"][at[0]]) if len(at) and rep["long_len"][at[0]] > 0 else float("nan")
        out[j] = dict(lt5=float((rep["mem_comp"][j, after] < 5).mean()), avg_long=float(rep["comp"][j, after].mean()),
                      min_long=int(rep["comp"][j, after].min()), share_end=share)
    return out


def class_coverage(rep: dict, d) -> dict:
    """流末长期池对每个已结束阶段 j 的类别覆盖：n_rows = 池里属于 j 的行数；per_class = 该阶段留出集里出现的各类在池里的行数
    （只含留出集里有的类）；covered = 其中行数 > 0 的类数；n_classes = 留出集里的类数。"""
    L = rep["final_long"]
    y, regime = d["y"], d["regime"]
    out = {}
    for j in range(len(d["boundaries"])):
        hold_cls = np.unique(d["y_hold"][d["regime_hold"] == j])
        sel = L[regime[L] == j]
        cnt = np.array([int((y[sel] == c).sum()) for c in hold_cls], dtype=int)
        out[j] = dict(n_rows=int(len(sel)), per_class=cnt, covered=int((cnt > 0).sum()), n_classes=int(len(hold_cls)))
    return out


def age_summary(rep: dict) -> float:
    """记忆已满之后（full）各次更新里长期池行年龄中位数的均值（行）；没有满的更新时 nan。"""
    m = rep["full"] & np.isfinite(rep["age_med"])
    return float(rep["age_med"][m].mean()) if m.any() else float("nan")


def final_pool_by_position(rep: dict, d) -> dict:
    """渐变段：流末长期池的行按流内位置分类（zone = [a, b)）：pure_first = 过渡区之前（纯 35 °C），t1/t2/t3 = 过渡区三等分
    （切法同 zone_metrics），pure_last = 过渡区之后（纯 23 °C）；classes_first = 过渡区之前的行覆盖的类数。"""
    z = _zone(d)
    if len(z) == 0:
        return {}
    a, b = int(z[0, 0]), int(z[0, 1])
    c = [a + (b - a) * k // 3 for k in range(4)]
    L = rep["final_long"]
    return dict(pure_first=int((L < a).sum()), t1=int(((L >= c[0]) & (L < c[1])).sum()),
                t2=int(((L >= c[1]) & (L < c[2])).sum()), t3=int(((L >= c[2]) & (L < c[3])).sum()),
                pure_last=int((L >= b).sum()), total=int(len(L)),
                classes_first=int(len(np.unique(d["y"][L[L < a]]))))


def last_present_time(rep: dict, j: int) -> "int | None":
    """长期池里最后一次含评估阶段 j 的行的更新时刻（已揭示行数）；从未含过返回 None。"""
    idx = np.flatnonzero(rep["comp"][j] > 0)
    return int(rep["ts"][idx[-1]]) if len(idx) else None


def adwin_alarm_info(d, rep: dict | None = None) -> dict:
    """ADWIN 运行的报警信息。alarms = closures（报警行之后的下一行，即新阶段第一行）。

    首次报警相对过渡区起点 / 相对文档变点 boundaries[0] 的带符号偏移（行数，负 = 在其之前）；
    further = 之后又报警几次；r0_share = 最终长期记忆里评估阶段 0 的行所占比例（需要重放结果）。"""
    al = np.asarray(d["closures"]).astype(int)
    z = _zone(d)
    first = int(al[0]) if len(al) else None
    out = dict(alarms=al.tolist(), n_alarms=len(al), first=first, further=max(len(al) - 1, 0),
               off_zone=None, off_cp=None, r0_share=float("nan"))
    if first is not None:
        if len(z):
            out["off_zone"] = first - int(z[0, 0])
        if len(d["boundaries"]):
            out["off_cp"] = first - int(d["boundaries"][0])
    if rep is not None and rep["long_len"][-1] > 0:
        out["r0_share"] = float(rep["comp"][0, -1] / rep["long_len"][-1])
    return out


def rolling_ba(y, pred, win: int = 500, step: int = 50):
    """滑动平衡准确率：窗口 (t − win, t] 内已预测的行。返回 (t 数组, 准确率数组)。"""
    ts, acc = [], []
    for t in range(step, len(y) + 1, step):
        lo = max(0, t - win)
        m = pred[lo:t] >= 0
        if m.sum() >= win // 2:
            ts.append(t)
            acc.append(balanced_accuracy(y[lo:t][m], pred[lo:t][m]))
    return np.asarray(ts), np.asarray(acc)


# ------------------------------------------------------------------ --quick

def expected_checkpoints(n: int, start: int, stride: int, ckpt_every: int, boundaries, mem_boundaries) -> list:
    """按 run_stream 的批次规则推出应有的检查点时刻（不运行模型）。"""
    bnds = sorted(int(b) for b in boundaries if 0 < b < n)
    cuts = sorted(set(bnds) | {int(b) for b in mem_boundaries if 0 < b < n})
    nxt_ck = (start // ckpt_every + 1) * ckpt_every
    t, out = start, []
    while t < n:
        nxt = [b for b in cuts if b > t]
        end = min(t + stride, n, nxt[0] if nxt else n)
        if end >= nxt_ck or end in bnds or end == n:
            out.append(end)
            while nxt_ck <= end:
                nxt_ck += ckpt_every
        t = end
    return out


def quick_check(d, floor_margin: float = 0.10) -> dict:
    """单个运行文件的快检。返回 dict(lines=中文说明行, reasons=需要看的原因列表, notes=提醒, metrics=指标)。"""
    from src.eval.stream_runner import needs_detector

    ds = str(d["dataset"][0]) if "dataset" in d else "insects"
    method = str(d["method"][0])
    seed = int(d["seed"][0])
    y, pred = d["y"], d["pred"]
    start, stride, ckpt_every = int(d["start"][0]), int(d["stride"][0]), int(d["ckpt_every"][0])
    n_end = effective_rows(d)
    msr = int(d["max_stream_rows"][0])
    K = len(d["class_ids"]) if "class_ids" in d else int(y.max()) + 1
    reasons, notes, lines = [], [], []

    complete = n_end > start and not (pred[start:n_end] < 0).any()
    n_pred = int((pred >= 0).sum())
    if not complete:
        reasons.append(f"行数不完整（已预测 {n_pred} 行，应为 {max(n_end - start, 0)}）")
    device = str(d["device"][0])
    if device == "cpu" and msr < 0:
        reasons.append("完整运行却在 CPU 上跑（device=cpu）")
    n_fits = int(d["n_fits"][0])
    if n_fits <= 0:
        reasons.append("n_fits = 0")
    m = run_metrics(d)
    floor = 1.0 / K + floor_margin
    if np.isfinite(m["ADAPT"]) and m["ADAPT"] < floor:
        reasons.append(f"ADAPT {m['ADAPT'] * 100:.2f} 低于 1/K+10pp = {floor * 100:.1f}")
    if not np.isfinite(m["ADAPT"]) or not np.isfinite(m["ADAPT_masked"]):
        reasons.append("ADAPT 为 nan")
    n_bnd = sum(1 for b in d["boundaries"] if 0 < b < n_end)
    if n_bnd > 0 and not np.isfinite(m["RET"]):
        reasons.append(f"RET 为 nan，但已有 {n_bnd} 个评估阶段结束，应可算")
    if n_bnd == 0 and np.isfinite(m["RET"]):
        reasons.append("没有评估阶段结束却算出了 RET")
    adwin = needs_detector(method)
    mb = [int(b) for b in _mem_boundaries(d) if 0 < b < n_end]
    clos = [int(c) for c in d["closures"]]
    if not adwin and clos != mb:
        reasons.append(f"closures {clos} 与 mem_boundaries {mb} 不一致")
    z = _zone(d).tolist()
    want_zone = GRADUAL_ZONE if ds in GRADUAL_DATASETS else []
    if z != want_zone:
        reasons.append(f"zone {z} 不符（应为 {want_zone}）")
    if int(d["max_ctx"][0]) > int(d["budget"][0]):
        reasons.append(f"max_ctx {int(d['max_ctx'][0])} > budget {int(d['budget'][0])}")
    if int(d["leak_count"][0]) != 0:
        reasons.append(f"leak_count = {int(d['leak_count'][0])}")
    exp_ck = expected_checkpoints(n_end, start, stride, ckpt_every, d["boundaries"], _mem_boundaries(d))
    got_ck = [int(t) for t in d["ckpt_t"]]
    if got_ck != exp_ck:
        reasons.append(f"检查点 {len(got_ck)} 个，与按规则推出的 {len(exp_ck)} 个不一致")
    if len(d["digests"]) != len(got_ck):
        reasons.append(f"digests {len(d['digests'])} 个 ≠ 检查点 {len(got_ck)} 个")
    if adwin and not clos:
        notes.append("ADWIN 没有报警（该 seed 的记忆等同 dual*，预注册允许保留）")

    head = (f"[{ds}/{method} seed {seed}] 行数 {'完整' if complete else '不完整'}"
            f"（{n_pred}/{max(n_end - start, 0)}{'，截断运行' if msr >= 0 else ''}）；device={device}；n_fits={n_fits}；"
            f"elapsed={float(d['elapsed'][0]) / 60:.1f} min；max_ctx={int(d['max_ctx'][0])}/{int(d['budget'][0])}；"
            f"leak={int(d['leak_count'][0])}；检查点 {len(got_ck)} 个")
    lines.append(head)
    fmt = lambda v: "nan" if not np.isfinite(v) else f"{v * 100:.2f}"      # noqa: E731
    per = "、".join(f"R{j}={fmt(m[f'RET_R{j}'])}" for j in range(len(d["boundaries"])) if f"RET_R{j}" in m)
    lines.append(f"    ADAPT={fmt(m['ADAPT'])}  ADAPT(masked)={fmt(m['ADAPT_masked'])}  RET={fmt(m['RET'])}"
                 + (f"  各阶段 RET：{per}" if per else "  （尚无评估阶段结束）"))
    if adwin:
        lines.append(f"    ADWIN 报警（新阶段第一行）：{clos}")
    else:
        lines.append(f"    closures={clos}（mem_boundaries 内应有 {mb}，{'一致' if clos == mb else '不一致'}）；zone={z}")
    return dict(lines=lines, reasons=reasons, notes=notes, metrics=m, dataset=ds, method=method, seed=seed,
                n_pred=n_pred)
