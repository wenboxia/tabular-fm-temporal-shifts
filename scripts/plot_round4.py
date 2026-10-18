"""第四轮结果图（短期占比 0.65 / 0.75 / 0.85 的探索性分析；英文标注，PNG + PDF）。

    python scripts/plot_round4.py --out results/figures_round4
    # 截断自测数据（0.75 列没有文件时该点留空）
    python scripts/plot_round4.py --dir <dir> --tag _smoke --n_estimators 1 --out <dir_out>

数据读取与 analyze_round4.py 相同（第四轮目录 + 第三轮的 0.75 列与 sw200 + 第二轮 insects 的 sw200）；缺的点自动跳过。
1. fig_ratio       每个数据集一列：上 RET、下 ADAPT 随短期占比（KDD 蓝方块、ours 红星、误差棒 = 95% t 区间，灰色虚线 = sw200 均值；
                   标记和第三轮图 1 一致；某个占比没有数据时线在那里断开，不跨点连接）
2. fig_ratio_diff  ours − KDD 的 ΔRET（上）与 ΔADAPT（下）随短期占比（95% t 区间 + 逐 seed 的点；0 线；
                   ΔADAPT 画 −0.5 pp 参考线，仅供参照）
探索性、只描述：不画显著性标记。
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

from analyze_round4 import parse_seeds  # noqa: E402
from src.eval.round3_analysis import NONINF_MARGIN, run_metrics  # noqa: E402
from src.eval.round4_analysis import (DATASETS4, SHARES, check_consistency4, diff_points, level_points,  # noqa: E402
                                      load_all4, sw_reference)

# 颜色与第二、三轮一致：滑窗灰、KDD 蓝、ours 红
C_SW, C_KDD, C_OURS = "#555555", "#1f77b4", "#d62728"
TITLE = {"insects": "Insects (abrupt)", "gas": "Gas Sensor Drift", "electricity": "Electricity (NSW)",
         "insects_gradual": "Insects (gradual slice, official change point)"}
XTICK = [f"{int(round(s * 100))}%" + ("\n(round 3)" if abs(s - 0.75) < 1e-9 else "") for s in SHARES]
WRITTEN = []


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        WRITTEN.append(f"{name}.{ext}")
    plt.close(fig)


def n_text(ns):
    ns = [n for n in ns if n > 0]
    if not ns:
        return "no data"
    if min(ns) == max(ns):
        return f"n = {ns[0]} seed" + ("" if ns[0] == 1 else "s")
    return f"n = {min(ns)}-{max(ns)} seeds"


def _no_data(ax):
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color="#888888")
    ax.set_yticks([])


def _style_x(ax):
    ax.set_xticks(range(len(SHARES)))
    ax.set_xticklabels(XTICK)
    ax.set_xlim(-0.5, len(SHARES) - 0.5)
    ax.grid(axis="y", alpha=0.25)


def fig_ratio(S, seeds, out):
    fig, axes = plt.subplots(2, len(DATASETS4), figsize=(4.2 * len(DATASETS4), 6.4), squeeze=False)
    ns = []
    for r, (key, ylab) in enumerate((("RET", "RET (%)"), ("ADAPT", "ADAPT (%)"))):
        pts = level_points(S, seeds, DATASETS4, key)
        ref = sw_reference(S, seeds, DATASETS4, key)
        for c, ds in enumerate(DATASETS4):
            ax = axes[r, c]
            drew = False
            for role, col, mk, ms, off, lab in (("kdd", C_KDD, "s", 6, -0.06, "KDD dual memory"),
                                                ("ours", C_OURS, "*", 10, 0.06, "Ours (regime-balanced)")):
                p = [q for q in pts if q["dataset"] == ds and q["role"] == role and q["n"] > 0]
                if not p:
                    continue
                ns += [q["n"] for q in p]
                # 三个占比各占一格，缺的占比放 nan，线在那里断开（不把 65% 和 85% 直接连起来）
                x = np.arange(len(SHARES)) + off
                m, lo, hi = (np.full(len(SHARES), np.nan) for _ in range(3))
                for q in p:
                    i = SHARES.index(q["share"])
                    m[i], lo[i], hi[i] = q["mean"], q["lo"], q["hi"]
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.plot(x, m, color=col, lw=1.0, alpha=0.6, zorder=2)
                if ok.any():
                    ax.errorbar(x[ok], m[ok], yerr=[(m - lo)[ok], (hi - m)[ok]], fmt="none", ecolor=col, elinewidth=1.4,
                                capsize=3, zorder=3)
                ax.plot(x, m, ls="none", marker=mk, color=col, ms=ms, label=lab, zorder=4)
                drew = True
            if np.isfinite(ref[ds]):
                ax.axhline(ref[ds], color=C_SW, ls="--", lw=1.2, label="Sliding window (200 rows)", zorder=1)
                drew = True
            if not drew:
                _no_data(ax)
            else:
                lo_, hi_ = ax.get_ylim()
                ax.set_ylim(lo_, min(hi_, 100.5))         # 准确率不会超过 100%，t 区间的上端可以，轴截到 100（同第三轮图 1）
            _style_x(ax)
            if r == 0:
                ax.set_title(TITLE[ds], fontsize=10)
            if r == 1:
                ax.set_xlabel("Short-term share of the 200-row memory")
            if c == 0:
                ax.set_ylabel(ylab)
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    if h:
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Short/long ratio at 200 rows (exploratory). Mean over seeds, bars = 95% t interval, {n_text(ns)}. "
                 "75% = round-3 runs; sliding window = round 3 (Insects abrupt: round 2). Each panel has its own y-axis.",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.975))
    save(fig, out, "fig_ratio")


def fig_ratio_diff(S, seeds, out):
    fig, axes = plt.subplots(2, len(DATASETS4), figsize=(4.2 * len(DATASETS4), 6.4), squeeze=False)
    ns = []
    for r, (key, ylab) in enumerate((("RET", "Δ RET, ours − KDD (pp)"), ("ADAPT", "Δ ADAPT, ours − KDD (pp)"))):
        pts = diff_points(S, seeds, DATASETS4, key)
        for c, ds in enumerate(DATASETS4):
            ax = axes[r, c]
            ax.axhline(0, color="black", lw=0.8, zorder=1)
            if key == "ADAPT":
                ax.axhline(100 * -NONINF_MARGIN, color="0.15", ls=(0, (1, 2)), lw=1.2, zorder=1,
                           label="−0.5 pp: round-3 non-inferiority margin (judged there by a one-sided bound); reference only")
            p = [q for q in pts if q["dataset"] == ds and q["n"] > 0]
            if p:
                ns += [q["n"] for q in p]
                x = np.arange(len(SHARES), dtype=float)
                m, lo, hi = (np.full(len(SHARES), np.nan) for _ in range(3))
                for q in p:                                  # 缺的占比放 nan：线断开，不跨点连
                    i = SHARES.index(q["share"])
                    m[i], lo[i], hi[i] = q["mean"], q["lo"], q["hi"]
                for q in p:
                    xi = SHARES.index(q["share"])
                    jit = np.linspace(-0.12, 0.12, len(q["per_seed"])) if len(q["per_seed"]) > 1 else np.zeros(1)
                    ax.plot(xi + jit, q["per_seed"], ls="none", marker=".", color=C_OURS, alpha=0.3, ms=5, zorder=2)
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.plot(x, m, color=C_OURS, lw=1.0, alpha=0.6, zorder=3)
                if ok.any():
                    ax.errorbar(x[ok], m[ok], yerr=[(m - lo)[ok], (hi - m)[ok]], fmt="none", ecolor=C_OURS, elinewidth=1.6,
                                capsize=3, zorder=4)
                ax.plot(x, m, ls="none", marker="o", color=C_OURS, ms=6, zorder=5,
                        label="Mean difference, 95% t interval (small dots: single seeds, spread in seed order)")
            else:
                _no_data(ax)
            _style_x(ax)
            if r == 0:
                ax.set_title(TITLE[ds], fontsize=10)
            if r == 1:
                ax.set_xlabel("Short-term share of the 200-row memory")
            if c == 0:
                ax.set_ylabel(ylab)
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    if h:
        fig.legend(h, l, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=9)
    fig.suptitle(f"Ours − KDD dual memory by short-term share, 200 rows (exploratory, no multiplicity correction), {n_text(ns)}. "
                 "Each panel has its own y-axis.",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.975))
    save(fig, out, "fig_ratio_diff")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/round4", help="第四轮 npz 目录")
    ap.add_argument("--round3_dir", default="results/round3")
    ap.add_argument("--round2_dir", default="results/round2")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--round3_tag", default="")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", default="results/figures_round4")
    args = ap.parse_args()

    runs, origin, probs = load_all4(args.dir, args.round3_dir, args.round2_dir, args.n_estimators, args.tag, args.round3_tag)
    if not any(v == "r4" for v in origin.values()):
        raise SystemExit("\n".join(probs) or f"{args.dir} 里没有第四轮的 npz")
    for p in probs + check_consistency4(runs, origin):
        print(p if p.startswith("!!") else f"注意：{p}")
    seeds = parse_seeds(args.seeds)
    S = {k: run_metrics(d) for k, d in runs.items()}
    os.makedirs(args.out, exist_ok=True)
    fig_ratio(S, seeds, args.out)
    fig_ratio_diff(S, seeds, args.out)
    print(f"wrote {sorted(WRITTEN)} to {args.out}")


if __name__ == "__main__":
    main()
