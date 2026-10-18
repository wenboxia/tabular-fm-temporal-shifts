"""第四轮分析用的纯函数（短期占比 0.65 / 0.75 / 0.85 的探索性分析：方法名解析、跨轮载入与一致性、容量检查、配对差）。

此模块属于第四轮（results/phase56_prereg.md 末尾 "Exploratory short/long ratio analysis"）的分析部分，
只读已有的 npz，不调用 TabPFN。第三轮的函数（载入、指标、配对差、标签重放等）原样复用 src/eval/round3_analysis.py，
这里不改动它的任何行为；第三轮的 parse_method 不认识 _s65 / _s85，所以第四轮另写 parse_method4。

使用流程（scripts/analyze_round4.py 与 scripts/plot_round4.py 调用）：
    runs, origin, probs = load_all4(dir4, dir3, dir2, n_estimators=4, tag="")   # 键 (dataset, method, seed)
    probs += check_consistency4(runs, origin)                                   # 跨轮允许 commit 不同，数据必须相同
    role, share = parse_method4("dual200_rb_s85")                               # ('ours', 0.85)
    S = {k: run_metrics(d) for k, d in runs.items()}                            # round3_analysis.run_metrics
    pts = diff_points(S, seeds, DATASETS4, "RET")                               # ours − KDD 随占比的配对差（pp）
    q = quick_check4(runs[("gas", "dual200_s65", 0)])                           # --quick 的逐文件检查（含重放核对容量）
    cal = calib4_check(dir4, runs, origin, n_estimators=4)                      # calib4 的 npz 与第三轮同名文件逐位比较

约定：0.75 列 = 第三轮的 dual200 / dual200_rb（commit 不同于第四轮；calib4 只重跑了 electricity / dual200_rb / seed 0 一个运行，
并在那个运行上逐位复现第三轮，calib4_check 用 npz 再核对一次）；
sw200 作参照（insects 来自第二轮，其余来自第三轮）。"ours" = 阶段均衡双记忆（*_rb），"KDD" = 论文的双记忆。
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

import src.eval.round3_analysis as R3
from src.eval.round3_analysis import load_runs, mean_sd_ci, paired

# ------------------------------------------------------------------ 方法与数据集

DATASETS4 = ("gas", "electricity", "insects_gradual", "insects")      # 画图与报告的顺序
SHARES = (0.65, 0.75, 0.85)
BUDGET = 200
# 独立写出的预期容量（短期, 长期）；不从代码里推，这样 make_memory 被改坏时检查得出来
EXPECTED_CAPS = {0.65: (130, 70), 0.75: (150, 50), 0.85: (170, 30)}
R4_METHODS = ("dual200_s65", "dual200_s85", "dual200_rb_s65", "dual200_rb_s85")
_METHOD_RE4 = re.compile(r"^(?:(sw200)|dual200(_rb)?(?:_s(65|85))?)$")


def parse_method4(m: str):
    """'dual200_rb_s85' -> ('ours', 0.85)；'dual200' -> ('kdd', 0.75)；'sw200' -> ('sw', None)。

    只认 200 行的 sw200 / dual200 / dual200_rb 及其 _s65 / _s85 变体；写 _s75 之类（第三轮就是 0.75，没有后缀）报错。"""
    g = _METHOD_RE4.match(m)
    if not g:
        raise ValueError(f"无法解析方法名 {m!r}")
    if g.group(1):
        return "sw", None
    return ("ours" if g.group(2) else "kdd"), (int(g.group(3)) / 100 if g.group(3) else 0.75)


def method_name(role: str, share) -> str:
    """parse_method4 的反函数：('ours', 0.65) -> 'dual200_rb_s65'；0.75 没有后缀；'sw' 忽略 share。"""
    if role == "sw":
        return "sw200"
    base = "dual200_rb" if role == "ours" else "dual200"
    s = int(round(share * 100))
    assert s in (65, 75, 85), f"不支持的占比 {share}"
    return base if s == 75 else f"{base}_s{s}"


def expected_runs(datasets=DATASETS4) -> list:
    """预期的 (数据集, 方法, 来源) 清单：第四轮 4 个方法；0.75 列（第三轮）；sw200（insects 来自第二轮，其余第三轮）。"""
    out = []
    for ds in datasets:
        out += [(ds, m, "r4") for m in R4_METHODS]
        out += [(ds, m, "r3") for m in ("dual200", "dual200_rb")]
        out.append((ds, "sw200", "r2" if ds == "insects" else "r3"))
    return out


# ------------------------------------------------------------------ 载入与一致性

def load_all4(dir4, dir3, dir2, n_estimators: int = 4, tag: str = "", tag3: str = "", datasets=DATASETS4):
    """载入三个来源。返回 (runs, origin, problems)：runs 键 (dataset, method, seed)，origin 同键 -> 'r2' | 'r3' | 'r4'。

    r4 = dir4 里的四个新方法（文件名后缀 tag）；r3 = dir3 里的 dual200 / dual200_rb（所有数据集）与 sw200（insects 除外）
    （后缀 tag3，正式结果为空）；r2 = dir2 里 insects 的 sw200（后缀 ''）。"""
    keep4 = {(ds, m) for ds in datasets for m in R4_METHODS}
    keep3 = {(ds, m) for ds in datasets for m in ("dual200", "dual200_rb")} | \
            {(ds, "sw200") for ds in datasets if ds != "insects"}
    keep2 = {("insects", "sw200")} if "insects" in datasets else set()
    runs, origin, probs = {}, {}, []
    for dir_, tg, keep, name in ((dir4, tag, keep4, "r4"), (dir3, tag3, keep3, "r3"), (dir2, "", keep2, "r2")):
        if not keep:
            continue
        r, p = load_runs([dir_], n_estimators, 50, keep=keep, tags=[tg])
        probs += p
        for k, d in r.items():
            runs[k] = d
            origin[k] = name
    if not any(v == "r4" for v in origin.values()):
        probs.append(_no_r4_hint(dir4, n_estimators, tag))
    ds4 = {k[0] for k, v in origin.items() if v == "r4"}
    if ds4 and not any(v == "r3" for v in origin.values()):
        probs.append(f"!! 第三轮目录 {dir3} 里没有载入任何 dual200 / dual200_rb / sw200 文件：0.75 列与 sw200 缺失，没有做跨轮核对"
                     "（检查 --round3_dir、--round3_tag）")
    if "insects" in ds4 and ("insects", "sw200") in keep2 and not any(v == "r2" for v in origin.values()):
        probs.append(f"!! 第二轮目录 {dir2} 里没有载入 insects 的 sw200 文件：insects 缺同规模滑窗参照，没有做跨轮核对"
                     "（检查 --round2_dir）")
    return runs, origin, probs


def _no_r4_hint(dir4, n_estimators, tag) -> str:
    """目录里没有第四轮 npz 时的提示：区分"目录不存在"和"目录在，但 npz 在子目录里"（并指出子目录）。"""
    if not os.path.isdir(dir4):
        return f"!! 第四轮目录不存在：{dir4}（检查 --dir）"
    pat = os.path.join(dir4, "**", f"r2_*_n{n_estimators}_S50_seed*{tag}.npz")
    r4_name = re.compile(r"_dual200(_rb)?_s(65|85)_n")
    hits = sorted(f for f in glob.glob(pat, recursive=True) if r4_name.search(os.path.basename(f)) and "calib" not in f)
    if hits:
        return (f"!! 第四轮目录 {dir4} 本身没有第四轮 npz，但在 {os.path.dirname(hits[0])} 里找到了（共 {len(hits)} 个），"
                "请用 --dir 指向它")
    return (f"!! 第四轮目录 {dir4} 存在，但它和它的子目录里都没有 r2_*_n{n_estimators}_S50_seed*{tag}.npz 形式的第四轮文件"
            "（检查 --n_estimators、--tag）")


def check_consistency4(runs, origin) -> list:
    """跨轮一致性。返回问题列表（空 = 通过）。

    1. y / hold_idx / stream_idx / boundaries / zone / mem_boundaries 在同一 (数据集, seed) 的所有方法（含第三轮的 0.75 列与 sw200）
       之间必须相同，n_estimators / 截断长度在数据集内一致，leak_count = 0，上下文不超预算，完整运行没有 -1
       （即第三轮的 check_consistency，但把 commit / device 先抹平：跨轮 commit 本来就不同）；
    2. 每个 (数据集, 来源) 内部 commit / device 必须唯一；第四轮所有文件的 commit 必须唯一（同一套代码跑的）。"""
    flat = {k: dict(d, commit=np.array(["-"]), device=np.array(["-"])) for k, d in runs.items()}
    problems = list(R3.check_consistency(flat))
    ids = {}
    for k, d in runs.items():
        ids.setdefault((k[0], origin[k]), set()).add((str(d["commit"][0]), str(d["device"][0])))
    for (ds, org), s in sorted(ids.items()):
        if len(s) > 1:
            problems.append(f"{ds}（来源 {org}）: 代码版本 / 设备不一致 {sorted(s)}")
    c4 = {str(d["commit"][0]) for k, d in runs.items() if origin[k] == "r4"}
    if len(c4) > 1:
        problems.append(f"第四轮文件的 commit 不唯一 {sorted(c4)}")
    return problems


def capacity_problems(d) -> list:
    """一个 200 行文件的**代码级**容量检查：预算 = 200；用方法名构造出来的记忆（本地代码里的 make_memory）
    的短期 / 长期容量与 EXPECTED_CAPS（130/70、150/50、170/30）一致；max_ctx ≤ 预算。sw200 只查窗口 = 200。

    注意：这只检查本地代码和方法名对不对，**不看 npz 实际是用什么容量跑出来的**。npz 实际用的记忆要靠
    replay_check（重放 digest 与 npz 比对）来证明。"""
    from src.memory.context_memory import make_memory

    method = str(d["method"][0])
    role, share = parse_method4(method)
    out = []
    budget = int(d["budget"][0])
    if budget != BUDGET:
        out.append(f"budget {budget} ≠ {BUDGET}")
    if int(d["max_ctx"][0]) > budget:
        out.append(f"max_ctx {int(d['max_ctx'][0])} > budget {budget}")
    y = d["y"]
    K = len(d["class_ids"]) if "class_ids" in d else int(y.max()) + 1
    mem = make_memory(method, np.zeros((len(y), 1), np.float32), y, K, 0)
    if role == "sw":
        if mem.window != BUDGET:
            out.append(f"{method} 的窗口 {mem.window} ≠ {BUDGET}")
        return out
    want = EXPECTED_CAPS[share]
    got = (mem.short_cap, mem.long_cap)
    if got != want:
        out.append(f"{method} 的短期 / 长期容量 {got[0]}/{got[1]} ≠ 预期 {want[0]}/{want[1]}")
    if mem.max_context() != BUDGET:
        out.append(f"{method} 的记忆总容量 {mem.max_context()} ≠ {BUDGET}")
    return out


def replay_check(d) -> "tuple[bool, str]":
    """用方法名对应的记忆重放这个 npz（run_stream + 标签，不跑模型），把每个检查点的 digest、closures、检查点时刻与 npz 比对。

    通过 = npz 实际用的记忆（含短期 / 长期容量）就是方法名所说的那个；失败返回 (False, 原因)。
    例如文件名写 _s65、实际按 0.75 跑的文件会在 digests 上失败。只适用于 dual* 方法。"""
    try:
        R3.replay_run(d, str(d["method"][0]), verify=True)
    except AssertionError as e:
        return False, str(e)
    except Exception as e:                                       # noqa: BLE001 —— 文件残缺等，当作重放失败报告
        return False, f"重放出错：{type(e).__name__}: {e}"
    return True, ""


def quick_check4(d, replay: bool = True) -> dict:
    """第三轮 quick_check（行数、device、closures == mem_boundaries、zone、检查点、leak、ADAPT / RET）再加：
    1. 代码级容量检查（capacity_problems：本地代码里的容量是否与方法名一致）；
    2. 重放核对（replay_check：npz 实际用的记忆是否与方法名一致；replay=False 时跳过）。
    两者的不符都进 reasons。"""
    q = R3.quick_check(d)
    cap = capacity_problems(d)
    q["reasons"] = list(q["reasons"]) + cap
    q["lines"] = list(q["lines"]) + [f"    容量（本地代码里）：{'与方法名一致' if not cap else '不符 —— ' + '；'.join(cap)}"]
    do_replay = replay and parse_method4(str(d["method"][0]))[0] != "sw"
    q["replay_ok"] = None
    if do_replay:
        ok, why = replay_check(d)
        q["replay_ok"] = ok
        if ok:
            q["lines"].append("    重放核对（digests / closures / 检查点时刻与 npz 比对）：通过，npz 实际用的记忆与方法名一致")
        else:
            q["lines"].append(f"    重放核对：**失败** —— {why}")
            q["reasons"].append(f"npz 实际用的记忆可能与方法名 {str(d['method'][0])} 不符 —— {why}")
    return q


def calib4_check(dir4, runs, origin, n_estimators: int = 4, tag: str = "") -> list:
    """calib4 的 npz（新代码重跑 electricity / dual200_rb / seed 0）与第三轮同名运行逐位比较：pred、digests、ckpt_pred。

    load_runs 会跳过名字含 calib 的文件，所以这里单独读 dir4 里的 r2_*_n{n}_S50_seed*_calib4{tag}.npz。
    返回 [dict(name, key, status, detail)]，status ∈ 'identical' | 'differs' | 'no_r3'；没找到 calib4 文件返回 []。"""
    out = []
    if not dir4 or not os.path.isdir(dir4):
        return out
    for f in sorted(glob.glob(os.path.join(dir4, f"r2_*_n{n_estimators}_S50_seed*_calib4{tag}.npz"))):
        with np.load(f, allow_pickle=True) as z:
            g = {k: z[k] for k in z.files}
        key = (str(g["dataset"][0]) if "dataset" in g else "insects", str(g["method"][0]), int(g["seed"][0]))
        name = os.path.basename(f)
        r = runs.get(key)
        if r is None or origin.get(key) != "r3":
            out.append(dict(name=name, key=key, status="no_r3", detail="第三轮同名运行没有载入，无法比较"))
            continue
        diff = []
        for k in ("pred", "digests", "ckpt_pred"):
            if k not in g or k not in r or not np.array_equal(np.asarray(g[k]), np.asarray(r[k])):
                diff.append(k)
        out.append(dict(name=name, key=key, status="differs" if diff else "identical",
                        detail=("以下字段与第三轮不同：" + "、".join(diff)) if diff else
                        "pred、digests、ckpt_pred 与第三轮同名运行逐位相同（np.array_equal）"))
    return out


# ------------------------------------------------------------------ 随占比变化的统计（报告与画图共用）

def metric_vals(S, ds, method, key, seeds) -> np.ndarray:
    """按 seeds 取一列指标，缺的是 nan（配对时按 seed 对齐）。"""
    return np.array([S.get((ds, method, s), {}).get(key, np.nan) for s in seeds], float)


def level_points(S, seeds, datasets, key) -> list:
    """每个 (数据集, 角色 ∈ {kdd, ours}, 占比) 一个点：该指标在 seed 间的均值与 95% t 区间（单位 pp）。"""
    out = []
    for ds in datasets:
        for role in ("kdd", "ours"):
            for share in SHARES:
                st = mean_sd_ci(metric_vals(S, ds, method_name(role, share), key, seeds))
                out.append(dict(dataset=ds, role=role, share=share, n=st["n"], mean=100 * st["mean"],
                                lo=100 * st["lo"], hi=100 * st["hi"]))
    return out


def sw_reference(S, seeds, datasets, key) -> dict:
    """同规模滑窗 sw200 的均值（pp），每个数据集一个；缺失为 nan。"""
    return {ds: 100 * mean_sd_ci(metric_vals(S, ds, "sw200", key, seeds))["mean"] for ds in datasets}


def diff_points(S, seeds, datasets, key) -> list:
    """ours − KDD 的配对差随占比：每个 (数据集, 占比) 一个点（pp）：均值、95% t 区间、n、正号 seed 数、逐 seed 差。"""
    out = []
    for ds in datasets:
        for share in SHARES:
            a = metric_vals(S, ds, method_name("ours", share), key, seeds)
            b = metric_vals(S, ds, method_name("kdd", share), key, seeds)
            r = paired(a, b)
            m = np.isfinite(a) & np.isfinite(b)
            out.append(dict(dataset=ds, share=share, n=r["n"], mean=100 * r["mean"], lo=100 * r["lo"], hi=100 * r["hi"],
                            npos=r["npos"], per_seed=100 * (a[m] - b[m])))
    return out
