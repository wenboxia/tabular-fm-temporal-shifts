"""前期探索（round 1）的汇总：报警后的动作有没有用、三层系统本身的代价、检测器在哪里报警。

    python scripts/analyze_round1.py --dir results

输入是 scripts/reproduce_round1.sh 产出的 npz（seed 42，标签方案 pair_A_vs_B）。
所有比较都在同一条数据流、同一批行上逐行配对；p 值是精确的 McNemar 检验（只看两边结果不同的行，
把各行当作独立样本，对时间序列偏乐观）。只有一个 seed，几个 errors 以内的差异应当视为噪声。
"""
import argparse
import glob
import os
import re

import numpy as np
from scipy.stats import binomtest

ACTIONS = ("context_reset", "route_adapter", "buffer_clear")
PAT = re.compile(r"multiseed_(phase1|phase4a)_real_insects_(.+)_(d\d_[a-z0-9]+)_seed42\.npz$")


def load(dir_, suffix=""):
    """suffix 非空时只读取标签以它结尾的运行（例如 SMOKE 的 _smoke），并在匹配前去掉它。"""
    runs = {}
    for f in glob.glob(os.path.join(dir_, "multiseed_*_real_insects_*_seed42.npz")):
        m = PAT.search(os.path.basename(f))
        if not m:
            continue
        tag = m.group(2)
        if suffix:
            if not tag.endswith(suffix):
                continue
            tag = tag[: -len(suffix)]
        runs[(m.group(1), tag, m.group(3))] = dict(np.load(f, allow_pickle=True))
    return runs


def wrong(d):
    return d["predictions"] != d["labels"]


def mcnemar(wa, wb):
    """a 比 b 少错多少（正 = a 更好）、两边不同的行数、精确双侧 p。"""
    b_only = int((wb & ~wa).sum())      # 只有 b 错
    a_only = int((wa & ~wb).sum())      # 只有 a 错
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return b_only - a_only, n, p


def alarms(d):
    a = np.asarray(d.get("alarm_events", []), dtype=np.int64).ravel()
    return a[a >= 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results")
    ap.add_argument("--suffix", default="", help="只分析标签以此结尾的运行，例如 _smoke")
    args = ap.parse_args()
    runs = load(args.dir, args.suffix)
    if not runs:
        raise SystemExit(f"{args.dir} 里没有 round 1 的结果文件（先运行 scripts/reproduce_round1.sh）")

    segs = sorted({s for _, _, s in runs})
    print(f"# Round 1 summary ({len(runs)} runs, segments: {', '.join(segs)})\n")

    print("## Action vs. no action (same trigger, same segment; + = errors avoided)\n")
    print("| trigger | action | segment | alarms | Δ errors | differing rows | p |")
    print("|---|---|---|---|---|---|---|")
    cells = []
    for (ph, tag, seg), d in sorted(runs.items()):
        m = re.fullmatch(r"b2_(oracle|detector_[a-z0-9_]+?)_(context_reset|route_adapter|buffer_clear)", tag)
        if ph != "phase4a" or not m:
            continue
        trig, act = m.group(1), m.group(2)
        base = runs.get(("phase4a", f"b2_{trig}_none", seg))
        if base is None:
            continue
        al = alarms(d)
        gain, n, p = mcnemar(wrong(d), wrong(base))
        cells.append(dict(trig=trig, act=act, seg=seg, n_alarm=len(al), gain=gain, n=n, p=p))
        print(f"| {trig} | {act} | {seg} | {len(al)} | {gain:+d} | {n} | {p:.3f} |")
    fired = [c for c in cells if c["n_alarm"] > 0]
    orc = [c for c in fired if c["trig"] == "oracle"]
    if fired:
        print(f"\nCells with at least one alarm: {len(fired)}; Δ errors {min(c['gain'] for c in fired):+d} to "
              f"{max(c['gain'] for c in fired):+d}; smallest p = {min(c['p'] for c in fired):.3f}")
    if orc:
        print(f"Oracle trigger only: {len(orc)} cells; Δ errors {min(c['gain'] for c in orc):+d} to "
              f"{max(c['gain'] for c in orc):+d}; smallest p = {min(c['p'] for c in orc):.3f}")

    print("\n## Cost of the three-level system (no action vs. TabPFN alone with a sliding window)\n")
    print("| segment | errors, TabPFN alone | errors, three-level | extra errors | differing rows | p |")
    print("|---|---|---|---|---|---|")
    for seg in segs:
        base = runs.get(("phase1", "v2AvsB_base", seg))
        none = next((runs[k] for k in sorted(runs) if k[0] == "phase4a" and k[2] == seg and k[1].endswith("_none")), None)
        if base is None or none is None:
            continue
        gain, n, p = mcnemar(wrong(base), wrong(none))
        print(f"| {seg} | {int(wrong(base).sum())} | {int(wrong(none).sum())} | {gain:+d} | {n} | {p:.4f} |")

    print("\n## Detector alarms (runs with action = none)\n")
    print("| detector input | segment | official change points | alarms |")
    print("|---|---|---|---|")
    for (ph, tag, seg), d in sorted(runs.items()):
        m = re.fullmatch(r"b2_detector_(.+)_none", tag)
        if ph == "phase4a" and m:
            dp = np.asarray(d["drift_points"]).ravel().tolist()
            print(f"| {m.group(1)} | {seg} | {dp if dp else '—'} | {alarms(d).tolist() or 'none'} |")


if __name__ == "__main__":
    main()
