"""第三轮结果图（英文标注，PNG + PDF）。

    python scripts/plot_round3.py --out results/figures_round3

数据读取与 analyze_round3.py 相同（第二轮 insects 的 1000 行与滑窗 + 第三轮目录）；缺的方法或数据集自动跳过。

1. fig_adapt_ret         每个数据集的 ADAPT–RET 散点（滑窗 / KDD / ours × 200、1000 行；误差棒 = seed SD）
2. fig_regime_ret        每个数据集的各阶段 RET 柱状图
3. fig_pool_<dataset>    seed 0 的长期池构成（KDD vs ours，各规模；只重放记忆，不跑模型）
4. fig_transition        渐变段过渡区前后的滑动平衡准确率（三个边界臂 + 滑窗 / KDD）
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from analyze_round3 import load_all, parse_seeds  # noqa: E402
from src.eval.round3_analysis import (DATASET_METHODS, HOLM_DATASETS, SIZES, check_consistency,  # noqa: E402
                                      parse_method, replay_run, rolling_ba, run_metrics)

# 颜色与第二轮（scripts/plot_round2.py）一致：滑窗灰、KDD 蓝、ours 红；ADWIN 臂沿用第二轮的粉色 + 'P' 标记；
# (b) 过渡区起点臂是第三轮新增的，另选第二轮没用过的青色 + 'D' 标记
C_SW, C_KDD, C_OURS, C_ADWIN, C_START = "#555555", "#1f77b4", "#d62728", "#ff9896", "#17becf"
# 池构成图里各阶段的样式 (颜色, 阴影)：不含 KDD 蓝、ours 红、ADWIN 粉、(b) 青、过渡区灰；
# 同色系（绿、橙、紫、黄、棕）的阶段用不同深浅，并加阴影线区分，至少 12 种
STAGE_STYLES = [("#b2df8a", ""), ("#fdbf6f", ""), ("#cab2d6", ""), ("#ffff99", ""),
                ("#33a02c", "//"), ("#e5c494", ""), ("#6a3d9a", ""), ("#bcbd22", ""),
                ("#e08214", "//"), ("#8c510a", ""), ("#b15928", ".."), ("#ffd92f", "..")]
TITLE = {"insects": "Insects (abrupt)", "gas": "Gas Sensor Drift", "electricity": "Electricity (NSW)",
         "insects_gradual": "Insects (gradual slice)"}
WRITTEN = []


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        WRITTEN.append(f"{name}.{ext}")
    plt.close(fig)


def vals(S, ds, m, key, seeds):
    return np.array([S.get((ds, m, s), {}).get(key, np.nan) for s in seeds], float)


def mean_sd(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def role_style(m, ds="insects_gradual"):
    role, size, adw = parse_method(m)
    if role == "sw":
        return C_SW, "o", "Sliding window"
    if role == "kdd":
        return C_KDD, "s", "KDD dual memory"
    if adw:
        return C_ADWIN, "P", "Ours, ADWIN boundary (c)"
    return C_OURS, "*", "Ours (regime-balanced)"


def n_seeds_text(counts):
    """实际有限值的 seed 数：全相同写 'n = 10 seeds'，否则写范围。"""
    counts = [c for c in counts if c > 0]
    if not counts:
        return "n = 0 seeds"
    return f"n = {counts[0]} seeds" if min(counts) == max(counts) else f"n = {min(counts)}-{max(counts)} seeds"


def fig_adapt_ret(S, seeds, out):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    counts = []
    # 标签偏移按角色错开：滑窗在下、KDD 居中偏上、ours 在上，避免重合的点标签叠在一起
    offset = {"sw": (-24, -11), "kdd": (5, 4), "ours": (5, 7)}
    zorder = {"sw": 4, "kdd": 3, "ours": 5}                       # 滑窗画在 KDD 上面（两者在 Gas / Electricity 上几乎重合）
    for ax, ds in zip(axes, HOLM_DATASETS):
        arms = [(ds, m) for m in DATASET_METHODS[ds]]
        if ds == "insects_gradual":
            arms += [("insects_gradual_start", m) for m in DATASET_METHODS["insects_gradual_start"]]
        xs_all = []
        for d_, m in arms:
            role, size, adw = parse_method(m)
            c, mk, _ = role_style(m)
            if d_ == "insects_gradual_start":
                c, mk = C_START, "D"
            ax_, sx = mean_sd(vals(S, d_, m, "ADAPT", seeds))
            ay, sy = mean_sd(vals(S, d_, m, "RET", seeds))
            if not (np.isfinite(ax_) and np.isfinite(ay)):
                continue
            both = np.isfinite(vals(S, d_, m, "ADAPT", seeds)) & np.isfinite(vals(S, d_, m, "RET", seeds))
            counts.append(int(both.sum()))
            xs_all.append(100 * ax_)
            ax.errorbar(100 * ax_, 100 * ay, xerr=100 * sx, yerr=100 * sy, fmt=mk, color=c,
                        mfc=c if size == 1000 else "white", ms=13 if mk == "*" else 8, capsize=2, lw=1,
                        zorder=zorder[role])
            ax.annotate(str(size), (100 * ax_, 100 * ay), textcoords="offset points",
                        xytext=offset["ours" if d_ == "insects_gradual_start" else role], fontsize=7, color=c)
        span = f"\n(x-axis span of means: {max(xs_all) - min(xs_all):.1f} pp)" if xs_all else ""
        ax.set_title(TITLE[ds] + span, fontsize=10)
        ax.set_xlabel("ADAPT: balanced prequential accuracy (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RET: balanced accuracy on holdout (%)")
    handles = [plt.Line2D([], [], marker="o", color=C_SW, ls="", label="Sliding window"),
               plt.Line2D([], [], marker="s", color=C_KDD, ls="", label="KDD dual memory"),
               plt.Line2D([], [], marker="*", color=C_OURS, ls="", ms=11,
                          label="Ours (regime-balanced; arm (a) on gradual slice)"),
               plt.Line2D([], [], marker="D", color=C_START, ls="", label="Ours, boundary at transition start (b)"),
               plt.Line2D([], [], marker="P", color=C_ADWIN, ls="", label="Ours, ADWIN boundary (c)"),
               plt.Line2D([], [], marker="o", color="0.4", mfc="white", ls="", label="open = 200 rows, filled = 1000 rows")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Adaptation vs retention at equal context size (mean ± SD over seeds; {n_seeds_text(counts)}; "
                 "x-axes differ between panels)", fontsize=11, y=1.02)
    save(fig, out, "fig_adapt_ret")


def fig_regime_ret(S, runs, seeds, out):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8), squeeze=False)
    for c, ds in enumerate(HOLM_DATASETS):
        K = next((len(d["boundaries"]) for (d_, m, s), d in sorted(runs.items()) if d_ == ds), 0)
        for r, size in enumerate(SIZES):
            ax = axes[r, c]
            ms = [(f"sw{size}", C_SW, "Sliding window"), (f"dual{size}", C_KDD, "KDD dual memory"),
                  (f"dual{size}_rb", C_OURS, "Ours (regime-balanced)")]
            w = 0.27
            for k, (m, col, lab) in enumerate(ms):
                if K == 0:
                    continue
                v = np.array([vals(S, ds, m, f"RET_R{j}", seeds) for j in range(K)]) * 100     # K × seeds
                mu = np.array([np.nanmean(x) if np.isfinite(x).any() else np.nan for x in v])
                sd = np.array([np.nanstd(x, ddof=1) if np.isfinite(x).sum() > 1 else 0.0 for x in v])
                ax.bar(np.arange(K) + (k - 1) * w, mu, w, yerr=sd, color=col, capsize=1.5, label=lab)
            ax.set_xticks(range(K))
            ax.set_xticklabels([f"R{j}" for j in range(K)], fontsize=8)
            ax.set_xlim(-0.6, max(K, 3) - 0.4)                  # 只有一个阶段的子图里柱宽与其他子图接近
            ax.set_ylim(0, 103)                                  # 留出 +SD 误差棒不被截断的空间
            ax.set_yticks(range(0, 101, 20))
            ax.set_title(f"{TITLE[ds]}, {size} rows" + (" (arm (a) only)" if ds == "insects_gradual" else ""), fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            if c == 0:
                ax.set_ylabel("RET: balanced accuracy on holdout (%)")
            if r == len(SIZES) - 1:
                ax.set_xlabel("Past evaluation regime")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Retention per old regime (mean ± SD over seeds; gradual slice: arm (a) only)", fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "fig_regime_ret")


def fig_pool(runs, ds, seed, out):
    if ds == "insects_gradual":                           # 与其他图对同一个臂的叫法一致
        cols = [("kdd", "KDD dual memory"), ("ours", "Ours (a): documented change"),
                ("ours_adwin", "Ours (c): ADWIN boundary")]
    else:
        cols = [("kdd", "KDD dual memory"), ("ours", "Ours (regime-balanced)")]
    d_any = next((d for (d_, m, s), d in sorted(runs.items()) if d_ == ds and s == seed), None)
    if d_any is None:
        return
    fig, axes = plt.subplots(len(SIZES), len(cols), figsize=(5.6 * len(cols), 3.2 * len(SIZES)), squeeze=False)
    n_reg = int(d_any["regime"].max()) + 1
    has_zone = "zone" in d_any and np.asarray(d_any["zone"]).size > 0
    made = False
    for r, size in enumerate(SIZES):
        for c, (role, lab) in enumerate(cols):
            ax = axes[r, c]
            m = {"kdd": f"dual{size}", "ours": f"dual{size}_rb", "ours_adwin": f"dual{size}_rb_adwin"}[role]
            d = runs.get((ds, m, seed))
            ax.set_title(f"{lab}, {size} rows", fontsize=9)
            if d is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                continue
            try:
                rep = replay_run(d, m, verify=True)
            except AssertionError as e:
                ax.text(0.5, 0.5, f"replay mismatch\n{e}", transform=ax.transAxes, ha="center", fontsize=7)
                continue
            made = True
            polys = ax.stackplot(rep["ts"], rep["comp"], colors=[STAGE_STYLES[j % len(STAGE_STYLES)][0] for j in range(n_reg)],
                                 labels=[f"R{j}" for j in range(n_reg)])
            for j, poly in enumerate(polys):
                poly.set_hatch(STAGE_STYLES[j % len(STAGE_STYLES)][1])
                poly.set_edgecolor("0.25")
                poly.set_linewidth(0)
            for b in d["boundaries"]:
                ax.axvline(int(b), color="k", ls="--", lw=0.6)
            for z in np.asarray(d["zone"]).reshape(-1, 2) if "zone" in d else []:
                ax.axvspan(int(z[0]), int(z[1]), color="0.8", alpha=0.5, lw=0)
            if role == "ours_adwin":                          # ADWIN 报警位置（新阶段第一行）
                for b in np.asarray(d["closures"]):
                    ax.axvline(int(b), color=C_ADWIN, ls=":", lw=1.5)
            ax.set_ylim(0, size // 4)
            ax.set_xlim(0, rep["ts"][-1])
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if c == 0:
                ax.set_ylabel(f"Rows in long-term pool (of {size // 4})")
            if r == len(SIZES) - 1:
                ax.set_xlabel("Stream position (rows)")
    if not made:
        plt.close(fig)
        return
    leg_title = ("evaluation regime of stored row\n(R0 = 35 °C + transition)" if ds == "insects_gradual"
                 else "regime of stored row")
    axes[0, -1].legend(title=leg_title, fontsize=6.5, title_fontsize=7, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), ncol=1 if n_reg <= 10 else 2)
    note = "dashed = evaluation-regime ends"
    if has_zone:
        note += ", grey = gradual transition"
    if ds == "insects_gradual":
        note += ", pink dotted = ADWIN alarms"
    fig.suptitle(f"Long-term pool composition over time, {TITLE[ds]} (example: seed {seed}; {note})", fontsize=10, y=1.0)
    fig.tight_layout()
    save(fig, out, f"fig_pool_{ds}")


def fig_transition(runs, seeds, out, win=500, step=50):
    arms = lambda size: [("insects_gradual", f"sw{size}", C_SW, "Sliding window"),            # noqa: E731
                         ("insects_gradual", f"dual{size}_rb", C_OURS, "Ours (a): boundary at documented change"),
                         ("insects_gradual_start", f"dual{size}_rb", C_START, "Ours (b): boundary at transition start"),
                         ("insects_gradual", f"dual{size}_rb_adwin", C_ADWIN, "Ours (c): ADWIN boundary"),
                         ("insects_gradual", f"dual{size}", C_KDD, "KDD dual memory")]       # KDD 最后画（虚线），不被 (a) 盖住
    d0 = next((d for (ds, m, s), d in sorted(runs.items()) if ds == "insects_gradual"), None)
    if d0 is None:
        return
    z = np.asarray(d0["zone"]).reshape(-1, 2)[0]
    fig, axes = plt.subplots(1, len(SIZES), figsize=(7.2 * len(SIZES), 4.0), sharey=True, squeeze=False)
    counts = []
    for ax, size in zip(axes[0], SIZES):
        for ds, m, col, lab in arms(size):
            curves, grid = [], None
            for s in seeds:
                d = runs.get((ds, m, s))
                if d is None:
                    continue
                t, a = rolling_ba(d["y"], d["pred"], win, step)
                if not len(t):
                    continue
                grid = t if grid is None else grid
                curves.append(np.interp(grid, t, a) * 100)
            if not curves:
                continue
            counts.append(len(curves))
            cv = np.vstack(curves)
            is_kdd = m == f"dual{size}"
            ax.plot(grid, cv.mean(0), color=col, lw=1.8 if is_kdd else 1.3, ls="--" if is_kdd else "-",
                    zorder=6 if is_kdd else 3, label=lab)
            if len(cv) > 1:
                sd = cv.std(0, ddof=1)
                ax.fill_between(grid, cv.mean(0) - sd, cv.mean(0) + sd, color=col, alpha=0.12, lw=0)
        ax.axvspan(int(z[0]), int(z[1]), color="0.85", alpha=0.6, lw=0, label="gradual transition")
        ax.axvline(int(d0["boundaries"][0]), color="k", ls="--", lw=0.8, label="documented change point")
        ax.set_xlim(max(0, int(z[0]) - 2500), min(len(d0["y"]), int(z[1]) + 3500))
        ax.set_title(f"{size}-row context", fontsize=10)
        ax.set_xlabel("Stream position (rows)")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(f"Balanced accuracy, trailing {win} rows (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"Accuracy around the gradual transition (mean ± SD over seeds, {n_seeds_text(counts)})", fontsize=11)
    fig.text(0.5, 0.0, "KDD (dashed) and Ours (a) coincide before the documented change point (identical memories). "
             f"Curves are trailing {win}-row windows, so a change appears up to {win} rows after its true position.",
             ha="center", va="bottom", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    save(fig, out, "fig_transition")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/round3", help="第三轮 npz 目录")
    ap.add_argument("--round2_dir", default="results/round2", help="第二轮 npz 目录（提供 insects 的四个方法）")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--pool_seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures_round3")
    args = ap.parse_args()

    runs, probs, _ = load_all(args)
    if not runs:
        raise SystemExit(f"{args.dir} 里没有 r2_*_n{args.n_estimators}_S50_seed*{args.tag}.npz")
    for p in probs + check_consistency(runs):
        print(f"注意：{p}")
    seeds = parse_seeds(args.seeds)
    S = {k: run_metrics(d) for k, d in runs.items()}
    os.makedirs(args.out, exist_ok=True)
    fig_adapt_ret(S, seeds, args.out)
    fig_regime_ret(S, runs, seeds, args.out)
    for ds in HOLM_DATASETS:
        fig_pool(runs, ds, args.pool_seed, args.out)
    fig_transition(runs, seeds, args.out)
    print(f"wrote {sorted(WRITTEN)} to {args.out}")


if __name__ == "__main__":
    main()
