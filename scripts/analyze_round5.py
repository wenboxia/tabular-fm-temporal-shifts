"""第五、六轮分析（审阅后补做）：固定分段的 RET（与阶段无关）、ADWIN 版补齐、CBRS 双记忆、阶段编号特征消融（修订 4），
阶段均衡的蓄水池双记忆（修订 5）。

    python scripts/analyze_round5.py                       # 写 results/round5_analysis.md 与 results/round5_per_seed.csv
    python scripts/analyze_round5.py --quick               # 只看 seed 0，打印到终端，不写文件
    # npz 在别的目录时：--round2_dir / --round3_dir / --round5_dir / --round6_dir

只读 npz，不调用 TabPFN。收集运行的规则（上一次只读重算把每批 10 行的敏感性运行混了进去，所以这里写死）：
  * 只取 stride == 50 且 n_estimators == 4 的完整运行（max_stream_rows == -1）；
  * 跳过文件名含 calib / smoke / pilot 的文件，跳过缺 method 或 ckpt_pred 字段的文件（如 r2_transfer_n4.npz）；
  * 按 (dataset, method, seed) 去重；同一个键出现两次且内容不同时报错退出。
指标：ADAPT、官方 RET（按官方阶段）、RET_seg2000（主）、RET_seg1000、RET_seg5000、RET_seg2000_lag5000
（定义见 src/eval/round2_metrics.py 的 ret_segments）、RET_worst（各被打分阶段 RET_j 的最小值；修订 5 登记，
对第二至五轮的方法是探索性的）。
比较族（各族 = 4 流 × 2 规模，族内 Holm，双侧配对 t）：A ours_adwin − cbdm；B ours − cbrs；C1 rid − cbdm；C2 ours − rid；
E1 rbr − cbrs；E2 rbr − ours。另报（描述性）：ours − cbdm、ours − sw、cbrs − cbdm、rbr − cbdm、rbr − sw。
还没有结果的方法自动跳过。完整模式另报 ADWIN 报警统计（e 节）与阶段编号列的重放计数（f 节，修订 4 澄清 1，全部 seed）。
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.round2_metrics import adapt, ret, ret_per_regime, ret_segments, ret_segments_per_segment  # noqa: E402
from src.eval.round3_analysis import check_consistency, holm, mean_sd_ci, paired, replay_run  # noqa: E402

NAN = float("nan")
STREAMS = ("insects", "gas", "electricity", "insects_gradual")
SIZES = (200, 1000)
ROLES = {                       # 角色 -> 方法名模板
    "sw": "sw{n}",
    "cbdm": "dual{n}",          # class-balanced dual memory（Lourenço et al.）
    "ours": "dual{n}_rb",       # regime-balanced dual memory（官方边界）
    "ours_adwin": "dual{n}_rb_adwin",
    "cbrs": "dual{n}_cbrs",
    "rid": "dual{n}_rid",
    "rbr": "dual{n}_rbr",       # 修订 5：阶段均衡的蓄水池双记忆
}
ROLE_NAMES = {"sw": "滑窗", "cbdm": "cbdm（类别均衡双记忆）", "ours": "ours（阶段均衡，官方边界）",
              "ours_adwin": "ours_adwin（阶段均衡，ADWIN 边界）", "cbrs": "cbrs（CBRS 长期池）",
              "rid": "rid（cbdm + 阶段编号特征）", "rbr": "rbr（阶段均衡蓄水池，修订 5）"}
METHOD_ROLE = {tpl.format(n=n): (role, n) for role, tpl in ROLES.items() for n in SIZES}
METRICS = ("ADAPT", "RET", "RET_seg2000", "RET_seg1000", "RET_seg5000", "RET_seg2000_lag5000", "RET_worst")
SEG_SPECS = {"RET_seg2000": (2000, 0), "RET_seg1000": (1000, 0), "RET_seg5000": (5000, 0),
             "RET_seg2000_lag5000": (2000, 5000)}
PRIMARY = "RET_seg2000"
FAMILIES = [  # (标签, a, b, 是否预先登记的比较族)
    ("A", "ours_adwin", "cbdm", True),
    ("B", "ours", "cbrs", True),
    ("C1", "rid", "cbdm", True),
    ("C2", "ours", "rid", True),
    ("E1", "rbr", "cbrs", True),
    ("E2", "rbr", "ours", True),
    ("D1", "ours", "cbdm", False),
    ("D2", "ours", "sw", False),
    ("D3", "cbrs", "cbdm", False),
    ("D4", "rbr", "cbdm", False),
    ("D5", "rbr", "sw", False),
]
SKIP_WORDS = ("calib", "smoke", "pilot")
CONTENT_KEYS = ("y", "pred", "y_hold", "hold_idx", "stream_idx", "boundaries", "regime_hold", "ckpt_t", "ckpt_pred")

# 第二、三轮报告里的官方数字（results/phase56_round2.md 主表、phase56_round3.md 主表与 2.1 / 2.2 节、第 4 节 (c) 臂）
EXPECTED_MEAN = {  # (流, 方法, 指标) -> 报告里的两位小数
    **{(ds, m, "RET"): v for ds, row in {
        "insects": ("35.08", "42.72", "44.17", "39.76", "45.86", "47.27"),
        "gas": ("78.67", "78.85", "96.85", "81.74", "82.02", "97.76"),
        "electricity": ("64.48", "64.48", "68.39", "65.68", "65.38", "68.59"),
        "insects_gradual": ("27.52", "25.74", "36.89", "31.89", "31.36", "46.99"),
    }.items() for m, v in zip(("sw200", "dual200", "dual200_rb", "sw1000", "dual1000", "dual1000_rb"), row)},
    **{(ds, m, "ADAPT"): v for ds, row in {
        "insects": ("75.68", "74.76", "74.76", "76.99", "76.65", "76.84"),
        "gas": ("98.54", "98.55", "98.64", "99.02", "99.03", "99.09"),
        "electricity": ("82.33", "82.58", "80.48", "77.43", "77.68", "77.17"),
        "insects_gradual": ("68.80", "72.27", "71.88", "74.48", "76.25", "76.01"),
    }.items() for m, v in zip(("sw200", "dual200", "dual200_rb", "sw1000", "dual1000", "dual1000_rb"), row)},
    ("insects", "dual1000_rb_adwin", "RET"): "46.45",
    ("insects", "dual1000_rb_adwin", "ADAPT"): "76.88",
    ("insects_gradual", "dual200_rb_adwin", "RET"): "34.89",
    ("insects_gradual", "dual1000_rb_adwin", "RET"): "43.04",
}
EXPECTED_DIFF = {  # (流, 规模, a 角色, b 角色) -> RET 配对差（pp，两位小数）；ours − cbdm 另有未校正 p
    **{(ds, n, "ours", "cbdm"): v for (ds, n), v in {
        ("insects", 200): "+1.45", ("insects", 1000): "+1.42", ("gas", 200): "+18.00", ("gas", 1000): "+15.74",
        ("electricity", 200): "+3.91", ("electricity", 1000): "+3.21",
        ("insects_gradual", 200): "+11.15", ("insects_gradual", 1000): "+15.63"}.items()},
    **{(ds, n, "ours", "sw"): v for (ds, n), v in {
        ("insects", 200): "+9.09", ("insects", 1000): "+7.51", ("gas", 200): "+18.18", ("gas", 1000): "+16.03",
        ("electricity", 200): "+3.91", ("electricity", 1000): "+2.91",
        ("insects_gradual", 200): "+9.37", ("insects_gradual", 1000): "+15.10"}.items()},
}
EXPECTED_P = {  # phase56_round3.md 2.1 节：(流, 规模) -> (未校正 p, Holm p)，ours − cbdm 的官方 RET
    ("insects", 200): ("5.26e-07", "1.05e-06"), ("insects", 1000): ("8.65e-07", "1.05e-06"),
    ("gas", 200): ("6.08e-13", "4.87e-12"), ("gas", 1000): ("2.42e-10", "1.21e-09"),
    ("electricity", 200): ("2.41e-09", "9.66e-09"), ("electricity", 1000): ("1.67e-08", "5.02e-08"),
    ("insects_gradual", 200): ("2.02e-11", "1.21e-10"), ("insects_gradual", 1000): ("6.87e-13", "4.87e-12"),
}


class DuplicateRunError(RuntimeError):
    pass


def parse_seeds(s: str) -> list:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def _field(d, k, default=None):
    return d[k][0] if k in d and np.ndim(d[k]) > 0 and len(d[k]) else (d[k].item() if k in d else default)


def same_content(a, b) -> bool:
    return all((k in a) == (k in b) and (k not in a or np.array_equal(a[k], b[k])) for k in CONTENT_KEYS)


def collect_runs(dirs, seeds=None, stride=50, n_estimators=4):
    """读入若干目录里的 r2_*.npz，返回 (runs, sources, notes)。runs 的键是 (dataset, method, seed)，
    只含 STREAMS × METHOD_ROLE 里的方法。同键重复且内容相同时保留先读到的并记一条；内容不同时抛 DuplicateRunError。"""
    runs, sources, notes = {}, {}, []
    skipped = {"名字含 calib/smoke/pilot": [], "缺 method 或 ckpt_pred": [], "stride ≠ 50 或 n_estimators ≠ 4": [],
               "截断运行（max_stream_rows ≥ 0）": []}
    ignored = 0
    for dir_ in dirs:
        if not dir_ or not os.path.isdir(dir_):
            if dir_:
                notes.append(f"目录 {dir_} 不存在，跳过")
            continue
        for f in sorted(glob.glob(os.path.join(dir_, "r2_*.npz"))):
            base = os.path.basename(f)
            if any(w in base.lower() for w in SKIP_WORDS):
                skipped["名字含 calib/smoke/pilot"].append(base)
                continue
            with np.load(f, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            if "method" not in d or "ckpt_pred" not in d:
                skipped["缺 method 或 ckpt_pred"].append(base)
                continue
            if int(_field(d, "stride", -1)) != stride or int(_field(d, "n_estimators", -1)) != n_estimators:
                skipped["stride ≠ 50 或 n_estimators ≠ 4"].append(base)
                continue
            if int(_field(d, "max_stream_rows", -1)) >= 0:
                skipped["截断运行（max_stream_rows ≥ 0）"].append(base)
                continue
            ds = str(_field(d, "dataset", "insects"))
            m = str(_field(d, "method"))
            s = int(_field(d, "seed"))
            if ds not in STREAMS or m not in METHOD_ROLE or (seeds is not None and s not in seeds):
                ignored += 1
                continue
            key = (ds, m, s)
            if key in runs:
                if not same_content(runs[key], d):
                    raise DuplicateRunError(f"{key} 在 {sources[key]} 与 {f} 里内容不同")
                notes.append(f"{key} 在 {sources[key]} 与 {f} 里都有，内容相同，用前者")
                continue
            runs[key] = d
            sources[key] = f
    for why, lst in skipped.items():
        if lst:
            notes.append(f"跳过 {len(lst)} 个文件（{why}）：" + "、".join(lst[:6]) + (" …" if len(lst) > 6 else ""))
    notes.append(f"不属于本分析的方法或流（例如第二轮的 arch_* / sw300、insects_gradual_start）{ignored} 个，未读入")
    return runs, sources, notes


def _stream_len(d, b, c) -> int:
    return len(d["y"]) if "y" in d else int(max(c.max(initial=0), b.max(initial=0))) + 1


def alarm_stats(d, near: int = 1000) -> dict:
    """ADWIN 报警（记忆关闭阶段的位置 closures）与官方变点 boundaries 的对照：报警次数、变点之后 near 行内
    有报警的变点比例、离任何变点都超过 near 行的报警数；检出率 = 下一个变点（或流末）之前有报警的变点比例，
    延迟 = 变点到它之后第一次报警的行数（只算检出的变点，取中位数）。"""
    b = np.asarray(d["boundaries"], dtype=np.int64)
    c = np.asarray(d["closures"], dtype=np.int64)
    return _alarm_core(b, c, _stream_len(d, b, c), near)


def _alarm_core(b, c, n: int, near: int) -> dict:
    hit = [bool(np.any((c > x) & (c <= x + near))) for x in b]
    far = int(sum(1 for x in c if not np.any(np.abs(b - x) <= near)))
    delays = []
    for x, nxt in zip(b, list(b[1:]) + [n]):
        after = c[(c > x) & (c <= nxt)]
        if len(after):
            delays.append(int(after.min() - x))
    return dict(n_alarms=len(c), n_cps=len(b), cp_hit=float(np.mean(hit)) if len(b) else NAN, far_alarms=far,
                detected=len(delays) / len(b) if len(b) else NAN,
                median_delay=float(np.median(delays)) if delays else NAN)


def alarm_chance(d, near: int = 1000, n_shift: int = 200) -> dict:
    """ADWIN 报警的机会基线（看过第五轮结果之后加的检查，未预先登记）：把一个运行的全部报警位置在流上
    整体循环平移（n_shift 个等距平移量，报警之间的间距不变），对每个平移量重算检出率、“变点后 near 行内
    有报警”的比例和延迟中位数，再对平移量取平均（延迟取中位数）。真实报警若跟随变点，检出应比平移后的
    更早、更集中；与平移后的差不多，就说明报警与变点无关。"""
    b = np.asarray(d["boundaries"], dtype=np.int64)
    c = np.asarray(d["closures"], dtype=np.int64)
    if not len(b) or not len(c):
        return dict(chance_detected=NAN, chance_hit=NAN, chance_delay=NAN)
    n = _stream_len(d, b, c)
    det, hit, dly = [], [], []
    for k in range(1, n_shift + 1):
        st = _alarm_core(b, np.sort((c + int(round(k * n / (n_shift + 1)))) % n), n, near)
        det.append(st["detected"])
        hit.append(st["cp_hit"])
        if np.isfinite(st["median_delay"]):
            dly.append(st["median_delay"])
    return dict(chance_detected=float(np.mean(det)), chance_hit=float(np.mean(hit)),
                chance_delay=float(np.median(dly)) if dly else NAN)


def run_metrics5(d) -> dict:
    out = dict(ADAPT=adapt(d), ADAPT_masked=adapt(d, masked=True), RET=ret(d))
    for k, (L, lag) in SEG_SPECS.items():
        per = ret_segments_per_segment(d, L, lag)
        out[k] = float(np.mean(list(per.values()))) if per else NAN
        out[f"n_{k}"] = len(per)
    rj = ret_per_regime(d)
    out["RET_worst"] = float(min(rj.values())) if rj else NAN
    if str(_field(d, "method", "")).endswith("_adwin") and "closures" in d:
        out.update(alarm_stats(d))
        out.update(alarm_chance(d))
    return out


def id_column_counts(mem_comp, ts, ckpt_t, boundaries, regime_hold) -> dict:
    """阶段编号列在检查点上的类型（修订 4 澄清 1）。只看至少有一个已结束阶段被打分的检查点（t > 该阶段的终点）。
    mem_comp：n_reg × T，第 k 次记忆更新之后整个上下文（长期 + 短期）里各评估阶段的行数；ts：每次更新之后已揭示的行数。
    TabPFN 6.4.1 按上下文推断列类型：只有 1 种编号 = 常数列（被删），2–3 种 = 类别列，≥ 4 种 = 数值列。
    返回 const / cat / num（检查点个数）与 absent（被打分的旧阶段留出行中，编号不在上下文里的比例，检查点平均）。"""
    b = np.asarray(boundaries, dtype=np.int64)
    ts = np.asarray(ts)
    rh = np.asarray(regime_hold)
    out = dict(const=0, cat=0, num=0)
    shares = []
    for t in np.asarray(ckpt_t):
        ended = np.flatnonzero(b < t)
        at = np.flatnonzero(ts == t)
        if len(ended) == 0 or len(at) == 0:
            continue
        col = mem_comp[:, at[-1]]
        k = int((col > 0).sum())
        out["const" if k <= 1 else "cat" if k <= 3 else "num"] += 1
        hold = np.array([(rh == j).sum() for j in ended], dtype=float)
        absent = np.array([col[j] == 0 for j in ended])
        shares.append(hold[absent].sum() / hold.sum())
    out["absent"] = float(np.mean(shares)) if shares else NAN
    return out


def rid_replay_counts(d) -> dict:
    """*_rid 的记忆就是 dual*（与模型预测无关）：用标签重放（digest 必须与 npz 一致），再数阶段编号列的类型。"""
    rep = replay_run(d, str(_field(d, "method")), verify=True)
    return id_column_counts(rep["mem_comp"], rep["ts"], d["ckpt_t"], d["boundaries"], d["regime_hold"])


REPLAY_ROLES = ("cbrs", "ours_adwin", "rbr")      # *_rid 已在 rid_replay_counts 里带 verify 重放


def replay_check(d) -> bool:
    """用标签重放（verify=True）：每个检查点的记忆 digest、closures 与检查点时刻都与 npz 一致返回 True。"""
    try:
        replay_run(d, str(_field(d, "method")), verify=True)
        return True
    except AssertionError:
        return False


def method_of(role, n):
    return ROLES[role].format(n=n)


def vals(S, ds, m, key, seeds):
    return np.array([S.get((ds, m, s), {}).get(key, NAN) for s in seeds], float)


def compare(S, seeds, a_role, b_role, metric):
    """对 4 流 × 2 规模算 a − b 的配对差；返回 {(ds, n): paired 结果（另加 p_holm）}，只含两边都有数据的格子；族内 Holm。"""
    cells = {}
    for ds in STREAMS:
        for n in SIZES:
            a = vals(S, ds, method_of(a_role, n), metric, seeds)
            b = vals(S, ds, method_of(b_role, n), metric, seeds)
            pr = paired(a, b)
            if pr["n"]:
                cells[(ds, n)] = pr
    if cells:
        adj = holm([c["p"] for c in cells.values()])
        for c, q in zip(cells.values(), adj):
            c["p_holm"] = float(q)
    return cells


# ------------------------------------------------------------------ 格式化

def pp(v, plus=False):
    if v is None or not np.isfinite(v):
        return "nan"
    return f"{v * 100:+.2f}" if plus else f"{v * 100:.2f}"


def fp(p):
    return "nan" if p is None or not np.isfinite(p) else f"{p:.3g}"


def table(P, header, rows):
    P("| " + " | ".join(header) + " |")
    P("|" + "---|" * len(header))
    for r in rows:
        P("| " + " | ".join(str(x) for x in r) + " |")
    P("")


def dcell(pr, with_p=False):
    if pr is None or pr["n"] == 0:
        return "—"
    ci = "" if not np.isfinite(pr["lo"]) else f" [{pp(pr['lo'], True)}, {pp(pr['hi'], True)}]"
    s = f"{pp(pr['mean'], True)}{ci} ({pr['npos']}/{pr['n']})"
    if with_p:
        s += f" p={fp(pr['p'])}, Holm {fp(pr.get('p_holm', NAN))}"
    return s


# ------------------------------------------------------------------ 官方 RET 核对

def read_csv_rows(path, default_dataset=None):
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ds = r.get("dataset", default_dataset)
            out[(ds, r["method"], int(r["seed"]))] = r
    return out


def verify_official(S, seeds, csv_paths):
    """返回 (表格行, 是否全部通过, 逐 seed 核对文字)。报告里的均值与配对差只在 seeds = 0–9 时才比。"""
    rows, ok = [], True
    full = list(seeds) == list(range(10))
    for (ds, m, metric), exp in (EXPECTED_MEAN.items() if full else []):
        v = vals(S, ds, m, metric, seeds)
        v = v[np.isfinite(v)]
        if len(v) != len(seeds):
            rows.append((ds, m, metric + " 均值", exp, f"缺数据（n={len(v)}）", "—"))
            continue
        got = f"{v.mean() * 100:.2f}"
        good = got == exp
        ok &= good
        rows.append((ds, m, metric + " 均值", exp, got, "一致" if good else "**不一致**"))
    for (ds, n, a, b), exp in (EXPECTED_DIFF.items() if full else []):
        pr = paired(vals(S, ds, method_of(a, n), "RET", seeds), vals(S, ds, method_of(b, n), "RET", seeds))
        if pr["n"] != len(seeds):
            rows.append((ds, n, f"RET {a} − {b}", exp, f"缺数据（n={pr['n']}）", "—"))
            continue
        got = pp(pr["mean"], True)
        good = got == exp
        ok &= good
        rows.append((ds, n, f"RET {a} − {b}", exp, got, "一致" if good else "**不一致**"))
    fam = compare(S, seeds, "ours", "cbdm", "RET")
    if full and len(fam) == len(EXPECTED_P) and all(c["n"] == len(seeds) for c in fam.values()):
        for (ds, n), (ep, eh) in EXPECTED_P.items():
            c = fam[(ds, n)]
            got = f"{fp(c['p'])} / {fp(c['p_holm'])}"
            good = got == f"{ep} / {eh}"
            ok &= good
            rows.append((ds, n, "RET ours − cbdm p / Holm p", f"{ep} / {eh}", got, "一致" if good else "**不一致**"))
    # 逐 seed：与第二、三轮的 per-seed CSV（6 位小数）比
    ref = {**read_csv_rows(csv_paths[0], "insects"), **read_csv_rows(csv_paths[1])}
    n_cmp, n_bad, worst = 0, 0, 0.0
    for key, m in S.items():
        r = ref.get(key)
        if r is None:
            continue
        for metric in ("RET", "ADAPT"):
            if r.get(metric) in (None, "", "nan"):
                continue
            diff = abs(float(r[metric]) - m[metric] * 100)
            n_cmp += 1
            worst = max(worst, diff)
            if diff > 5e-7 + 1e-9:
                n_bad += 1
    ok &= n_bad == 0
    txt = (f"逐 seed 核对：与 `results/round2_per_seed.csv`、`results/round3_per_seed.csv`（6 位小数）共比 {n_cmp} 个数"
           f"（RET 与 ADAPT），最大绝对差 {worst:.2e} pp，超出舍入误差（5e-7）的 {n_bad} 个。"
           + ("" if full else "（seed 不是 0–9，报告里的均值与配对差没有比。）"))
    return rows, ok, txt


# ------------------------------------------------------------------ 报告

def build_report(S, sources, notes, problems, seeds, args, rid_counts=None, replay_ok=None):
    lines = []
    P = lines.append
    P("# 第五轮自动分析报告（scripts/analyze_round5.py 生成，请勿手改）")
    P("")
    P(f"目录：第二轮 `{args.round2_dir}`，第三轮 `{args.round3_dir}`，第五轮 `{args.round5_dir}`，第六轮 `{args.round6_dir}`；"
      f"seed {seeds[0]}–{seeds[-1]}"
      f"（{len(seeds)} 个）。只取 stride = 50、n_estimators = 4 的完整运行。准确率是平衡准确率，单位 %；差值单位 pp；"
      "± 是 seed 间 SD；差值同 seed 配对，括号内是差值为正的 seed 数 / 配对数；区间为双侧 95 % t 区间；p 为双侧配对 t。")
    P("")
    P("## a. 载入")
    P("")
    for n_ in notes:
        P(f"- {n_}")
    P(f"- 读入 {len(S)} 个运行。一致性检查（同一流、同一 seed 的 y / 留出行 / 边界相同，泄漏 = 0，上下文 ≤ 预算，运行完整）："
      + ("通过。" if not problems else f"{len(problems)} 个问题："))
    for p in problems:
        P(f"  - {p}")
    P("")
    rows = []
    for ds in STREAMS:
        for n in SIZES:
            cnt = [f"{r} {sum((ds, method_of(r, n), s) in S for s in seeds)}" for r in ROLES]
            rows.append((ds, n, "，".join(cnt)))
    table(P, ["流", "规模", "各角色的 seed 数"], rows)
    segs = {}
    for (ds, m, s), v in S.items():
        segs.setdefault(ds, {k: v[f"n_{k}"] for k in SEG_SPECS})
    P("各指标实际用到的段数（同一流的所有运行相同）：" + "；".join(
        f"{ds} " + " / ".join(f"{k.replace('RET_', '')} {v}" for k, v in segs[ds].items()) for ds in STREAMS if ds in segs) + "。")
    P("")

    P("## b. 官方 RET 核对（应与 phase56_round2.md / phase56_round3.md 逐位一致）")
    P("")
    vrows, ok, txt = verify_official(S, seeds, (args.round2_csv, args.round3_csv))
    P(f"**结论：{'全部一致' if ok else '有不一致，见表'}。** {txt}")
    P("")
    table(P, ["流", "方法 / 规模", "量", "报告值", "本脚本", "判定"], vrows)

    P("## c. 各方法均值 ± SD（%）")
    P("")
    for n in SIZES:
        P(f"### {n} 行")
        P("")
        rows = []
        for ds in STREAMS:
            for role in ROLES:
                m = method_of(role, n)
                if not any((ds, m, s) in S for s in seeds):
                    continue
                cells = []
                for k in METRICS:
                    st = mean_sd_ci(vals(S, ds, m, k, seeds))
                    cells.append("—" if st["n"] == 0 else (pp(st["mean"]) if st["n"] < 2 else f"{pp(st['mean'])} ± {pp(st['sd'])}"))
                rows.append((ds, m, *cells))
        table(P, ["流", "方法", *METRICS], rows)

    P("## d. 配对差")
    P("")
    P(f"主指标是 {PRIMARY}。A、B、C1、C2 是比较族（每族 4 流 × 2 规模，Holm 在族内、按指标分别校正）；"
      "D1–D3 是另报的描述性比较（同样在 8 格内做 Holm，只作参考）。还没有结果的方法整族跳过。")
    P("")
    for lab, a, b, fam in FAMILIES:
        name = f"{lab}：{a} − {b}" + ("" if fam else "（描述性）")
        res = {k: compare(S, seeds, a, b, k) for k in METRICS}
        if not res[PRIMARY] and not res["RET"]:
            P(f"### {name}")
            P("")
            P(f"尚无数据（缺 {ROLE_NAMES[a] if not any(method_of(a, n) in {k[1] for k in S} for n in SIZES) else ROLE_NAMES[b]}）。")
            P("")
            continue
        P(f"### {name}")
        P("")
        rows = []
        for ds in STREAMS:
            for n in SIZES:
                if (ds, n) not in res[PRIMARY] and (ds, n) not in res["RET"]:
                    continue
                rows.append((ds, n, dcell(res[PRIMARY].get((ds, n)), True), dcell(res["RET"].get((ds, n)), True),
                             dcell(res["ADAPT"].get((ds, n)), True)))
        table(P, ["流", "规模", f"{PRIMARY}", "RET（官方）", "ADAPT"], rows)
        rows = []
        for ds in STREAMS:
            for n in SIZES:
                if (ds, n) in res["RET_worst"]:
                    rows.append((ds, n, dcell(res["RET_worst"][(ds, n)], True)))
        if rows:
            P("最差阶段的保留 RET_worst（修订 5 登记；对第二至五轮的方法是探索性的；insects_gradual 只有一个被打分的阶段）：")
            P("")
            table(P, ["流", "规模", "RET_worst"], rows)
        rows = []
        for ds in STREAMS:
            for n in SIZES:
                if (ds, n) not in res[PRIMARY]:
                    continue
                rows.append((ds, n, *[dcell(res[k].get((ds, n))) for k in ("RET_seg1000", "RET_seg5000", "RET_seg2000_lag5000")]))
        if rows:
            P("敏感性（同一比较，其他分段口径）：")
            P("")
            table(P, ["流", "规模", "RET_seg1000", "RET_seg5000", "RET_seg2000_lag5000"], rows)

    rows = []
    for ds in STREAMS:
        for n in SIZES:
            v = [S[(ds, method_of("ours_adwin", n), s)] for s in seeds if (ds, method_of("ours_adwin", n), s) in S]
            if v and "n_alarms" in v[0]:
                na = np.array([x["n_alarms"] for x in v])
                dl = [x["median_delay"] for x in v if np.isfinite(x["median_delay"])]
                cd = [x["chance_delay"] for x in v if np.isfinite(x.get("chance_delay", NAN))]
                rows.append((ds, n, len(v), v[0]["n_cps"], f"{na.mean():.1f}（{na.min()}–{na.max()}）",
                             f"{np.mean([x['cp_hit'] for x in v]) * 100:.0f} %", f"{np.mean([x['far_alarms'] for x in v]):.1f}",
                             f"{np.mean([x['detected'] for x in v]) * 100:.0f} %", f"{np.median(dl):.0f}" if dl else "—",
                             f"{np.mean([x['chance_hit'] for x in v]) * 100:.0f} %",
                             f"{np.mean([x['chance_detected'] for x in v]) * 100:.0f} %",
                             f"{np.median(cd):.0f}" if cd else "—"))
    if rows:
        P("## e. ADWIN 报警统计（ours_adwin）")
        P("")
        P("报警 = 记忆关闭阶段的位置；“变点后 1000 行内有报警”按官方变点计；“离变点 > 1000 行的报警”视为与官方变点无关的报警。"
          "检出 = 下一变点（或流末）之前有报警（不论报警原因）；延迟 = 每个运行检出变点的延迟中位数，再对 seed 取中位数。"
          "机会基线（看过结果后加的检查）：同一运行的报警整体循环平移 200 个等距量（间距不变）后重算，平移量上取平均（延迟取中位数）。")
        P("")
        table(P, ["流", "规模", "seed 数", "官方变点", "报警次数 均值（范围）", "变点后 1000 行内有报警", "离变点 > 1000 行的报警",
                  "检出（下一变点前有报警）", "延迟中位数（行）", "平移后：1000 行内", "平移后：检出", "平移后：延迟中位数"], rows)

    if rid_counts:
        P("## f. 阶段编号列的重放计数（修订 4 澄清 1，全部 seed）")
        P("")
        P("只看至少有一个已结束阶段被打分的检查点；常数 / 类别 / 数值列 = 上下文里阶段编号有 1 / 2–3 / ≥ 4 种；"
          "缺失比例 = 被打分的旧阶段留出行中编号不在上下文里的比例（检查点平均，再对 seed 平均）。记忆按标签重放，digest 与 npz 一致。")
        P("")
        rows = []
        for ds in STREAMS:
            for n in SIZES:
                v = [rid_counts[(ds, method_of("rid", n), s)] for s in seeds if (ds, method_of("rid", n), s) in rid_counts]
                if not v:
                    continue
                s0 = rid_counts.get((ds, method_of("rid", n), seeds[0]))
                rows.append((ds, n, len(v),
                             f"{np.mean([x['const'] for x in v]):.1f} / {np.mean([x['cat'] for x in v]):.1f} / {np.mean([x['num'] for x in v]):.1f}",
                             f"{np.mean([x['absent'] for x in v]) * 100:.0f} %",
                             f"{s0['const']} / {s0['cat']} / {s0['num']}，{s0['absent'] * 100:.0f} %" if s0 else "—"))
        table(P, ["流", "规模", "seed 数", "常数 / 类别 / 数值（均值）", "缺失比例", f"seed {seeds[0]}"], rows)

    if replay_ok:
        P("")
        P("## g. 标签重放核对（第五、六轮比较用到的全部双记忆运行）")
        P("")
        P("每个运行用标签重放（哑预测器，ADWIN 版用 npz 里的报警位置），每个检查点的记忆 digest、报警位置与检查点时刻"
          "都必须与 npz 逐位一致；*_rid 的重放见 f 段（同样带核对）。")
        P("")
        rows = []
        for role in (*REPLAY_ROLES, "rid"):
            for n in SIZES:
                m = method_of(role, n)
                if role == "rid":
                    ks = [k for k in (rid_counts or {}) if k[1] == m]
                    rows.append((m, len(ks), len(ks)))
                else:
                    ks = [k for k in replay_ok if k[1] == m]
                    rows.append((m, len(ks), sum(replay_ok[k] for k in ks)))
        table(P, ["方法", "运行数", "digest 一致"], rows)
    return "\n".join(lines) + "\n"


def write_csv(path, S, sources, seeds):
    cols = ["dataset", "method", "role", "size", "seed", "ADAPT", "ADAPT_masked", "RET",
            *SEG_SPECS, "RET_worst", *[f"n_{k}" for k in SEG_SPECS]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for key in sorted(S, key=lambda k: (STREAMS.index(k[0]), k[1], k[2])):
            ds, m, s = key
            if s not in seeds:
                continue
            role, n = METHOD_ROLE[m]
            v = S[key]
            w.writerow([ds, m, role, n, s,
                        *[f"{v[k] * 100:.6f}" for k in ("ADAPT", "ADAPT_masked", "RET", *SEG_SPECS, "RET_worst")],
                        *[v[f"n_{k}"] for k in SEG_SPECS]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round2_dir", default="results/archive/rog_round2/npz")
    ap.add_argument("--round3_dir", default="results/archive/rog_round3/npz")
    ap.add_argument("--round5_dir", default="results/archive/rog_round5/npz")
    ap.add_argument("--round6_dir", default="results/archive/rog_round6/npz")
    ap.add_argument("--round2_csv", default="results/round2_per_seed.csv")
    ap.add_argument("--round3_csv", default="results/round3_per_seed.csv")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default=None, help="报告路径（默认 results/round5_analysis.md；--quick 时默认不写）")
    ap.add_argument("--csv", default=None, help="逐 seed CSV 路径（默认 results/round5_per_seed.csv；--quick 时默认不写）")
    ap.add_argument("--quick", action="store_true", help="只看 seed 0，打印到终端")
    args = ap.parse_args(argv)
    seeds = [0] if args.quick else parse_seeds(args.seeds)
    if not args.quick:
        args.out = args.out or "results/round5_analysis.md"
        args.csv = args.csv or "results/round5_per_seed.csv"

    runs, sources, notes = collect_runs([args.round2_dir, args.round3_dir, args.round5_dir, args.round6_dir], seeds=set(seeds))
    problems = check_consistency(runs)
    S = {k: run_metrics5(d) for k, d in runs.items()}
    rid_counts = None if args.quick else {k: rid_replay_counts(d) for k, d in runs.items() if METHOD_ROLE[k[1]][0] == "rid"}
    replay_ok = None if args.quick else {k: replay_check(d) for k, d in runs.items()
                                         if METHOD_ROLE[k[1]][0] in REPLAY_ROLES and "digests" in d and "regime" in d}
    rep = build_report(S, sources, notes, problems, seeds, args, rid_counts, replay_ok)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(rep)
        print(f"报告写到 {args.out}")
    else:
        print(rep)
    if args.csv:
        write_csv(args.csv, S, sources, seeds)
        print(f"逐 seed CSV 写到 {args.csv}")
    return S


if __name__ == "__main__":
    main()
