"""Round 2 分析：汇总各方法 × seed 的指标，按预注册的 H1–H8 判定，画适应-遗忘前沿图。

    python scripts/analyze_round2.py --dir results/round2 --n_estimators 4 --out results/phase56_round2_summary.md
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.round2_metrics import fgt_per_regime, summarize  # noqa: E402

WINDOWS = [100, 200, 300, 400, 600, 1000, 1500, 2000]
OURS, BASE = "dual1000_rb", "sw200"   # 修订 1（2026-10-03）：原为 arch_routed


def load(dir_, n, S, tag):
    runs = {}
    pat = re.compile(rf"r2_(.+)_n{n}_S{S}_seed(\d+){re.escape(tag)}\.npz$")
    for f in sorted(glob.glob(os.path.join(dir_, f"r2_*_n{n}_S{S}_seed*{tag}.npz"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        d = dict(np.load(f, allow_pickle=True))
        runs[(m.group(1), int(m.group(2)))] = d
    return runs


def consistency(runs):
    problems = []
    by_seed = {}
    for (m, s), d in runs.items():
        by_seed.setdefault(s, []).append((m, d))
    for s, lst in by_seed.items():
        m0, d0 = lst[0]
        for m, d in lst[1:]:
            for k in ("y", "hold_idx", "stream_idx", "boundaries"):
                if not np.array_equal(d[k], d0[k]):
                    problems.append(f"seed {s}: {m} 与 {m0} 的 {k} 不同")
    metas = {(str(d["commit"][0]), str(d["device"][0]), int(d["n_estimators"][0]), int(d["max_stream_rows"][0]))
             for d in runs.values()}
    if len(metas) > 1:
        problems.append(f"结果来自不同的代码版本/设备/配置：{sorted(metas)}")
    for (m, s), d in runs.items():
        if int(d["leak_count"][0]) != 0:
            problems.append(f"{m} seed {s}: leak_count = {int(d['leak_count'][0])}")
        if int(d["max_ctx"][0]) > int(d["budget"][0]):
            problems.append(f"{m} seed {s}: 上下文超预算")
    return problems


def paired(a, b):
    """返回 (均值差, 单侧 p(差>0), 单侧 95% 下界, 正号的 seed 数, n)。"""
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    n = len(d)
    if n < 2:
        return float(d.mean()) if n else float("nan"), float("nan"), float("nan"), int((d > 0).sum()), n
    p = float(stats.ttest_rel(a, b, alternative="greater").pvalue)
    lower = float(d.mean() - stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n))
    return float(d.mean()), p, lower, int((d > 0).sum()), n


def baselines(d):
    """不用 TabPFN 的两个基线：No-Change（上一个标签）与最近 200 行的多数类。"""
    from src.eval.round2_metrics import balanced_accuracy
    y = d["y"]; done = np.flatnonzero(d["pred"] >= 0)
    nochange = y[done - 1]
    maj = np.array([np.bincount(y[max(0, t - 200):t], minlength=6).argmax() for t in done])
    return balanced_accuracy(y[done], nochange), balanced_accuracy(y[done], maj)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round2")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    runs = load(args.dir, args.n_estimators, args.stride, args.tag)
    if not runs:
        raise SystemExit("没有找到结果文件")
    lines = []
    P = lambda *a: lines.append(" ".join(str(x) for x in a))   # noqa: E731
    probs = consistency(runs)
    P(f"# Round 2 summary (n_estimators={args.n_estimators}, S={args.stride}, tag='{args.tag}')\n")
    P(f"runs: {len(runs)}；一致性检查：{'OK' if not probs else '; '.join(probs)}\n")

    S = {k: summarize(d) for k, d in runs.items()}
    methods = sorted({m for m, _ in runs}, key=lambda m: (not m.startswith("sw"), int(m[2:]) if m.startswith("sw") else 0, m))
    seeds = sorted({s for _, s in runs})
    keys = ["ADAPT", "ADAPT_masked", "RET"] + [f"POST_{k}" for k in range(5)]
    P("| method | n | " + " | ".join(keys) + " |")
    P("|---|---|" + "---|" * len(keys))
    for m in methods:
        vals = [[S[(m, s)][k] for s in seeds if (m, s) in S] for k in keys]
        cells = [f"{np.mean(v) * 100:.2f} ± {np.std(v, ddof=1) * 100:.2f}" if len(v) > 1 else f"{np.mean(v) * 100:.2f}"
                 for v in vals]
        P(f"| {m} | {len(vals[0])} | " + " | ".join(cells) + " |")

    def col(m, k):
        return [S[(m, s)][k] for s in seeds if (m, s) in S]

    def common(m1, m2, k):
        ss = [s for s in seeds if (m1, s) in S and (m2, s) in S]
        return [S[(m1, s)][k] for s in ss], [S[(m2, s)][k] for s in ss]

    P("\n## 预注册假设（主要假设需要 10 个种子才给正式结论）\n")
    from scipy import stats

    def final(n):
        return n == 10

    if not any((OURS, s) in S for s in seeds):
        P(f"（没有 {OURS} 的结果，跳过假设检验）")
    else:
        prim = []      # (名称, 文本, p, 其他通过条件)
        for name, other, key, thr in [("H1", BASE, "RET", 0.05), ("H3", "dual1000", "RET", 0.02)]:
            if any((other, s) in S for s in seeds):
                a, b = common(OURS, other, key)
                diff, p, lower, npos, n = paired(a, b)
                prim.append((name, f"{key}({OURS}) − {key}({other}) = {diff * 100:+.2f} pp（阈值 ≥ {thr * 100:.0f} pp；正号 {npos}/{n}）",
                             p, diff >= thr and npos >= int(np.ceil(0.9 * n)), n))
        if any((BASE, s) in S for s in seeds):
            ps, txt, ok_all, n2 = [], [], True, 0
            for key in ("ADAPT", "ADAPT_masked"):          # 非劣效：d > −0.5 pp
                a, b = common(OURS, BASE, key)
                d = np.asarray(a) - np.asarray(b); n2 = len(d)
                p = float(stats.ttest_1samp(d + 0.005, 0, alternative="greater").pvalue) if n2 >= 2 else float("nan")
                lower = float(d.mean() - stats.t.ppf(0.95, n2 - 1) * d.std(ddof=1) / np.sqrt(n2)) if n2 >= 2 else float("nan")
                ps.append(p); ok_all &= int((d > -0.005).sum()) >= int(np.ceil(0.9 * n2))
                txt.append(f"{key} 差 {d.mean() * 100:+.2f} pp（单侧 95% 下界 {lower * 100:+.2f}）")
            prim.append(("H2", "；".join(txt) + "（需 > −0.50 pp，两个版本都要）", max(ps), ok_all, n2))
        adj = holm([r[2] if np.isfinite(r[2]) else 1.0 for r in prim])
        for (name, txt, p, cond, n), pa in zip(prim, adj):
            if n < 2:
                verdict = "单个种子，不做检验"
            elif not final(n):
                verdict = f"{'PASS' if (cond and pa < 0.05) else 'FAIL'}（只有 {n} 个种子，非正式）"
            else:
                verdict = "PASS" if (cond and pa < 0.05) else "FAIL"
            P(f"- **{name}** {txt}；p_holm = {pa:.3g} → **{verdict}**")

        multi = len(seeds) >= 2
        tag = (lambda ok: "PASS" if ok else "FAIL") if multi else (lambda ok: "（单个种子，仅供参考）")
        for k, (kind, thr) in {"POST_1": ("gain", 0.05), "POST_3": ("gain", 0.05),
                               "POST_0": ("ctrl", 0.02), "POST_4": ("ctrl", 0.02)}.items():
            if any((BASE, s) in S for s in seeds):
                a, bb = common(OURS, BASE, k)
                diff = np.mean(a) - np.mean(bb)
                ok = diff >= thr if kind == "gain" else abs(diff) <= thr
                P(f"- H4 {k}: {diff * 100:+.2f} pp（{'需 ≥ +5' if kind == 'gain' else '阴性对照，需在 ±2 内'}）→ {tag(ok)}")
        ours_a, ours_r = np.mean(col(OURS, "ADAPT")), np.mean(col(OURS, "RET"))
        dom = [m for m in methods if m.startswith("sw") and np.mean(col(m, "ADAPT")) >= ours_a
               and np.mean(col(m, "RET")) >= ours_r]
        P(f"- H5 同时在 ADAPT 与 RET 上不差于 {OURS} 的滑窗：{dom or '无'} → {tag(not dom)}")
        if any(("cbfifo400", s) in S for s in seeds):
            a, b = common(OURS, "cbfifo400", "RET"); diff = np.mean(a) - np.mean(b)
            P(f"- H6 RET({OURS}) − RET(cbfifo400) = {diff * 100:+.2f} pp（需 ≥ +2）→ {tag(diff >= 0.02)}")
        if any(("dual1000", s) in S for s in seeds):
            a7, b7 = common(OURS, "dual1000", "ADAPT"); diff = np.mean(a7) - np.mean(b7)
            P(f"- H7 ADAPT({OURS}) − ADAPT(dual1000) = {diff * 100:+.2f} pp（需 ≥ −0.5）→ {tag(diff >= -0.005)}")
        if any((OURS + "_adwin", s) in S for s in seeds) and any((BASE, s) in S for s in seeds):
            g1 = np.mean(col(OURS, "RET")) - np.mean(col(BASE, "RET"))
            a, b = common(OURS + "_adwin", BASE, "RET"); g8 = np.mean(a) - np.mean(b)
            lows = []
            for key in ("ADAPT", "ADAPT_masked"):
                a2, b2 = common(OURS + "_adwin", BASE, key); lows.append(paired(a2, b2)[2])
            ok = g1 > 0 and g8 >= 0.5 * g1 and all(np.isfinite(lows)) and min(lows) > -0.005
            P(f"- H8 ADWIN 版 RET 增益 {g8 * 100:+.2f} pp（官方切点版 {g1 * 100:+.2f}），ADAPT 下界 "
              f"{', '.join(f'{x * 100:+.2f}' for x in lows)} → {tag(ok)}")

    P("\n## 不用 TabPFN 的基线（平衡准确率 %）\n")
    # 每个 seed 的留出集不同，所以数据流也不同：对每个 seed 各算一次再取平均
    ref = BASE if any(k[0] == BASE for k in runs) else next(iter(runs))[0]
    bl = np.array([baselines(runs[k]) for k in sorted(runs) if k[0] == ref])
    nc, mj = bl.mean(0)
    P(f"No-Change {nc * 100:.2f}；最近 200 行多数类 {mj * 100:.2f}（{len(bl)} 个 seed 的均值）")

    P("\n## 漂移后 500 行（剔除单类长段的版本，%）\n")
    P("| method | " + " | ".join(f"POST_{k}_masked" for k in range(5)) + " |")
    P("|---|" + "---|" * 5)
    for m in methods:
        P(f"| {m} | " + " | ".join(f"{np.nanmean(col(m, f'POST_{k}_masked')) * 100:.1f}" for k in range(5)) + " |")

    P("\n## 遗忘量 FGT_j = 最佳 − 最终留出准确率（%，越小越好）\n")
    P("| method | " + " | ".join(f"R{j}" for j in range(5)) + " | 平均 |")
    P("|---|" + "---|" * 6)
    for m in methods:
        vals = [fgt_per_regime(runs[(m, s)]) for s in seeds if (m, s) in runs]
        cells = [np.mean([v[j] for v in vals if j in v]) * 100 for j in range(5)]
        P(f"| {m} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | {np.mean(cells):.1f} |")

    P("\n## 各阶段留出集的保留（RET_Rj，均值）\n")
    P("| method | " + " | ".join(f"R{j}" for j in range(5)) + " |")
    P("|---|" + "---|" * 5)
    for m in methods:
        cells = []
        for j in range(5):
            v = [S[(m, s)].get(f"RET_R{j}") for s in seeds if (m, s) in S]
            v = [x for x in v if x is not None]
            cells.append(f"{np.mean(v) * 100:.1f}" if v else "—")
        P(f"| {m} | " + " | ".join(cells) + " |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sw = [m for m in methods if m.startswith("sw")]
        xs = [np.mean(col(m, "ADAPT")) * 100 for m in sw]; ys = [np.mean(col(m, "RET")) * 100 for m in sw]
        ax.plot(xs, ys, "-o", color="0.5", label="sliding window (size sweep)")
        for m, x, y in zip(sw, xs, ys):
            ax.annotate(m[2:], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for m, mk in [("dual1000", "s"), ("dual400", "s"), ("cbfifo400", "D"), ("arch_union", "^"),
                      ("arch_routed", "v"), ("arch_routed_adwin", "<"), ("dual1000_rb", "*"),
                      ("dual1000_rb_adwin", "P")]:
            if any((m, s) in S for s in seeds):
                ax.plot(np.mean(col(m, "ADAPT")) * 100, np.mean(col(m, "RET")) * 100, mk,
                        markersize=12 if m == OURS else 8, label=m)
        ax.set_xlabel("ADAPT: balanced prequential accuracy (%)")
        ax.set_ylabel("RET: balanced accuracy on past-regime holdouts (%)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=150)
        print(f"[plot] {args.plot}")


if __name__ == "__main__":
    main()
