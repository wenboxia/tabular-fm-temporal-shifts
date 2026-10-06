"""Round 2 结果图（英文标注，PNG + PDF）与逐 seed 汇总 CSV。

    python scripts/plot_round2.py --dir results/round2 --out results/figures_round2

只跑了部分方法或部分 seed 时也能画，缺的方法和图会跳过。

1. fig_frontier         适应–遗忘前沿图（16 方法，10 seeds 均值 ± 标准差）
2. fig_transfer         6×6 阶段迁移矩阵热力图（上下文阶段 × 留出阶段）
3. fig_retention_time   各旧阶段留出集准确率随数据流推进的变化（sw200 / dual1000 / dual1000_rb）
4. fig_regime_retention 各旧阶段的 RET 柱状图
5. fig_pool_composition 长期池中各阶段行数随时间的变化（双记忆 vs 阶段均衡，seed 0，只回放记忆）
另写一份逐 seed 汇总 CSV（公开，读者不需要 npz 就能核对文中的数字）。
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.round2_metrics import fgt_per_regime, holdout_curve, regime_end, summarize  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {
    "sw200": "Sliding window 200 (TabPFN alone)", "dual1000": "Dual memory (Lourenço et al., KDD 2026)",
    "dual1000_rb": "Regime-balanced dual memory (ours)", "dual1000_rb_adwin": "Ours, ADWIN-segmented",
    "dual400": "Dual memory, 400 rows", "cbfifo400": "Class-balanced FIFO 400",
    "arch_union": "Regime archive (all)", "arch_routed": "Regime archive (routed)",
    "arch_routed_adwin": "Regime archive (routed, ADWIN)",
}
C_OURS, C_DUAL, C_SW = "#d62728", "#1f77b4", "#555555"


def load(dir_, n, S):
    runs = {}
    for f in glob.glob(os.path.join(dir_, f"r2_*_n{n}_S{S}_seed*.npz")):
        m = re.search(rf"r2_(.+)_n{n}_S{S}_seed(\d+)\.npz$", os.path.basename(f))
        if m:
            runs[(m.group(1), int(m.group(2)))] = dict(np.load(f, allow_pickle=True))
    return runs


WRITTEN = []


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        WRITTEN.append(f"{name}.{ext}")
    plt.close(fig)


def seeds_of(runs):
    return sorted({s for _, s in runs})


def frontier(runs, S, out):
    methods = sorted({m for m, _ in runs})
    seeds = seeds_of(runs)
    stat = {m: (np.array([S[(m, s)]["ADAPT"] for s in seeds if (m, s) in S]) * 100,
                np.array([S[(m, s)]["RET"] for s in seeds if (m, s) in S]) * 100) for m in methods}
    sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0  # noqa: E731
    fig, ax = plt.subplots(figsize=(7, 5))
    sw = sorted([m for m in methods if m.startswith("sw")], key=lambda m: int(m[2:]))
    xs = [stat[m][0].mean() for m in sw]; ys = [stat[m][1].mean() for m in sw]
    if sw:
        ax.plot(xs, ys, "-o", color=C_SW, label="Sliding window (TabPFN alone; label = window size)", zorder=2)
    for m, x, y in zip(sw, xs, ys):
        ax.annotate(m[2:], (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8, color=C_SW)
    style = {"dual1000": ("s", C_DUAL, 9), "dual400": ("s", "#9ecae1", 7), "cbfifo400": ("D", "#2ca02c", 7),
             "arch_union": ("^", "#9467bd", 8), "arch_routed": ("v", "#8c564b", 8),
             "arch_routed_adwin": ("<", "#c49c94", 8), "dual1000_rb": ("*", C_OURS, 16),
             "dual1000_rb_adwin": ("P", "#ff9896", 10)}
    for m, (mk, c, ms) in style.items():
        if m in stat:
            a, r = stat[m]
            ax.errorbar(a.mean(), r.mean(), xerr=sd(a), yerr=sd(r), fmt=mk, color=c,
                        markersize=ms, capsize=2, label=LABEL[m], zorder=3)
    zoom = [m for m in ("dual1000", "dual1000_rb", "dual1000_rb_adwin") if m in stat]
    if len(zoom) >= 2:   # 放大图：三种双记忆挤在一起时才需要
        ins = ax.inset_axes([0.57, 0.52, 0.23, 0.4])
        for m in zoom:
            a, r = stat[m]
            mk, c, ms = style[m]
            ins.errorbar(a.mean(), r.mean(), xerr=sd(a), yerr=sd(r), fmt=mk, color=c, markersize=ms * 0.8, capsize=2)
        xa = [stat[m][0].mean() for m in zoom]; ya = [stat[m][1].mean() for m in zoom]
        ins.set_xlim(min(xa) - 0.15, max(xa) + 0.15); ins.set_ylim(min(ya) - 1.0, max(ya) + 1.0)
        ins.tick_params(labelsize=7); ins.grid(alpha=0.3)
        ins.set_title("zoom: dual memories", fontsize=7)
        ax.indicate_inset_zoom(ins, edgecolor="0.5")
    ax.set_xlabel("Adaptation: balanced prequential accuracy on the stream (%)")
    ax.set_ylabel("Retention: balanced accuracy on past-regime holdouts (%)")
    ax.set_title(f"Adaptation–retention trade-off (Insects, {len(seeds)} seeds; bars = ±1 SD)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower left")
    save(fig, out, "fig_frontier")


def transfer(dir_, out, n):
    p = os.path.join(dir_, f"r2_transfer_n{n}.npz")
    if not os.path.exists(p):
        return
    M = np.load(p)["matrix"].mean(0) * 100
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=max(70, M.max()))
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] < 40 else "black")
    names = ["R0\n30°C", "R1\n20°C", "R2\n~35°C", "R3", "R4", "R5"]
    ax.set_xticks(range(6)); ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels([n.replace("\n", " ") for n in names], fontsize=8)
    ax.set_xlabel("Evaluated on held-out data of regime"); ax.set_ylabel("Context drawn from regime")
    ax.set_title("Balanced accuracy (%) of TabPFN by context regime")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, out, "fig_transfer")


def retention_time(runs, out):
    seeds = seeds_of(runs)
    meths = [(m, c) for m, c in [("sw200", C_SW), ("dual1000", C_DUAL), ("dual1000_rb", C_OURS)]
             if any((m, s) in runs for s in seeds)]
    if not meths:
        return
    d0 = next(runs[(m, s)] for m, _ in meths for s in seeds if (m, s) in runs)
    n_old = len(d0["boundaries"])
    fig, axes = plt.subplots(1, n_old, figsize=(3.2 * n_old, 3.4), sharey=True, squeeze=False)
    axes = axes[0]
    for j, ax in enumerate(axes):
        for m, c in meths:
            curves = [holdout_curve(runs[(m, s)], j) for s in seeds if (m, s) in runs]
            if not len(curves[0][0]):
                continue
            ts = curves[0][0]
            acc = np.vstack([cv[1] for cv in curves]) * 100
            band = acc.std(0, ddof=1) if len(acc) > 1 else np.zeros(acc.shape[1])
            ax.plot(ts, acc.mean(0), color=c, lw=1.4, label=LABEL[m])
            ax.fill_between(ts, acc.mean(0) - band, acc.mean(0) + band, color=c, alpha=0.15)
        ax.axvline(regime_end(d0, j), color="k", ls="--", lw=0.8)
        for b in d0["boundaries"]:
            ax.axvline(int(b), color="#9ecae1", ls=":", lw=0.8, zorder=0)
        ax.set_title(f"Holdout of regime R{j}", fontsize=10)
        ax.set_xlabel("Stream position (rows)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Accuracy on each regime's holdout along the stream (mean ± SD over {len(seeds)} seeds; black dashed = "
                 "end of that regime, blue dotted = other change points)", fontsize=10)
    save(fig, out, "fig_retention_time")


def regime_bars(S, out):
    seeds = sorted({s for _, s in S})
    meths = [(m, c) for m, c in [("sw200", C_SW), ("dual1000", C_DUAL), ("dual1000_rb", C_OURS)]
             if any((m, s) in S for s in seeds)]
    if not meths:
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.26
    for k, (m, c) in enumerate(meths):
        vals = np.array([[S[(m, s)].get(f"RET_R{j}", np.nan) for j in range(5)] for s in seeds if (m, s) in S]) * 100
        err = np.nanstd(vals, 0, ddof=1) if len(vals) > 1 else None
        ax.bar(np.arange(5) + (k - 1) * w, np.nanmean(vals, 0), w, yerr=err, color=c, capsize=2, label=LABEL[m])
    ax.set_xticks(range(5)); ax.set_xticklabels(["R0 (30°C)", "R1 (20°C)", "R2 (~35°C)", "R3", "R4"])
    ax.set_ylabel("Retention (%)"); ax.set_title(f"Retention per old regime ({len(seeds)} seeds)")
    ax.set_ylim(0, 75); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8, loc="upper left", ncol=1)
    save(fig, out, "fig_regime_retention")


def pool_composition(out, seed=0):
    """长期池（250 行）里各阶段的行数随数据流的变化。只回放记忆本身（标签决定淘汰），不调用 TabPFN。"""
    sys.path.insert(0, os.path.dirname(__file__))
    from run_round2 import START, prepare  # noqa: E402
    from src.eval.stream_runner import run_stream  # noqa: E402
    from src.memory.context_memory import make_memory  # noqa: E402

    d = prepare(seed)
    n_reg = int(d["regime"].max()) + 1
    dummy = lambda Xc, yc, Xq: np.full((len(Xq), d["n_classes"]), 1.0 / d["n_classes"])  # noqa: E731
    colors = ["#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6", "#fb9a99", "#ffff99"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    final = {}
    for ax, m in zip(axes, ["dual1000", "dual1000_rb"]):
        mem = make_memory(m, d["X"], d["y"], d["n_classes"], seed)
        ts, comp = [], []
        orig = mem.update

        def update(idx, mem=mem, orig=orig):
            orig(idx)
            ts.append(int(np.max(idx)) + 1)
            comp.append(np.bincount(d["regime"][np.asarray(mem._long, dtype=np.int64)], minlength=n_reg))
        mem.update = update
        run_stream(mem, dummy, d["X"], d["y"], d["regime"], d["X_hold"], d["regime_hold"], d["boundaries"],
                   stride=50, start=START)
        C = np.vstack(comp).T
        final[m] = C[:, -1]
        ax.stackplot(ts, C, colors=colors, labels=[f"R{j}" for j in range(n_reg)])
        for b in d["boundaries"]:
            ax.axvline(int(b), color="k", ls="--", lw=0.6)
        ax.set_title(LABEL[m], fontsize=10)
        ax.set_xlabel("Stream position (rows)"); ax.set_xlim(0, ts[-1]); ax.set_ylim(0, 250)
    axes[0].set_ylabel("Rows in long-term pool (of 250)")
    axes[1].legend(title="regime of stored row", fontsize=8, title_fontsize=8, loc="center left",
                   bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("What the long-term pool holds over time (seed 0; dashed = official change points)", fontsize=11, y=1.03)
    save(fig, out, "fig_pool_composition")
    print("final long-pool rows per regime:", {k: v.tolist() for k, v in final.items()})


def per_seed_csv(runs, S, path):
    keys = ["ADAPT", "ADAPT_masked", "RET"] + [f"RET_R{j}" for j in range(5)] + \
           [f"POST_{k}" for k in range(5)] + [f"FGT_R{j}" for j in range(5)]
    with open(path, "w") as fh:
        fh.write("method,seed," + ",".join(keys) + "\n")
        for (m, s) in sorted(runs):
            row = dict(S[(m, s)])
            row.update({f"FGT_R{j}": v for j, v in fgt_per_regime(runs[(m, s)]).items()})
            fh.write(f"{m},{s}," + ",".join(f"{row.get(k, float('nan')) * 100:.6f}" for k in keys) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round2", help="run_round2_multiseed.py 的输出目录")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--out", default="results/round2/figures",
                    help="发布的图在 results/figures_round2（reproduce_round2.sh 显式传入）")
    ap.add_argument("--csv", default="results/round2/per_seed.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    runs = load(args.dir, args.n_estimators, args.stride)
    if not runs:
        raise SystemExit(f"{args.dir} 里没有 r2_*_n{args.n_estimators}_S{args.stride}_seed*.npz")
    S = {k: summarize(d) for k, d in runs.items()}
    per_seed_csv(runs, S, args.csv)
    frontier(runs, S, args.out)
    transfer(args.dir, args.out, args.n_estimators)
    retention_time(runs, args.out)
    regime_bars(S, args.out)
    pool_composition(args.out)
    seeds, methods = seeds_of(runs), sorted({m for m, _ in runs})
    print(f"wrote {args.csv} and {sorted(WRITTEN)} to {args.out} ({len(methods)} methods, {len(seeds)} seeds)")


if __name__ == "__main__":
    main()
