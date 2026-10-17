"""第三轮结果图（英文标注，PNG + PDF）。

    python scripts/plot_round3.py --out results/figures_round3

数据读取与 analyze_round3.py 相同（第二轮 insects 的 1000 行与滑窗 + 第三轮目录）；缺的方法或数据集自动跳过。

1. fig_adapt_ret         每个数据集的 ADAPT–RET 散点（滑窗 / KDD / ours × 200、1000 行；误差棒 = seed SD；四个子图刻度相同）
   fig_diff_forest       配对差森林图（ours − KDD、ours − 同规模滑窗；左 ΔRET、右 ΔADAPT；95% t 区间）
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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402

from analyze_round3 import load_all, parse_seeds  # noqa: E402
from src.eval.round3_analysis import (DATASET_METHODS, HOLM_DATASETS, NONINF_MARGIN, SIZES,  # noqa: E402
                                      centered_limits, check_consistency, common_span, forest_points,
                                      parse_method, replay_run, rolling_ba, run_metrics, tick_step)

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
ACC_CAP = 100.5                                                   # 准确率坐标轴上限（略高于 100，留出标记的边）
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


def place_labels(fig, ax, labels):
    """给每个点放规模标签（'200' / '1000'）：在点的八个方向、三种距离里挑第一个不压住任何点、误差棒、
    已放标签、也不超出坐标轴的位置；都不行就取重叠最少的。labels 的元素 = (文字, x, y, xerr, yerr, 点大小, 颜色, 偏好)。"""
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72.0                                            # 点 → 像素
    tr = ax.transData
    boxes = []                                                    # 障碍物（像素矩形 x0, y0, x1, y1）
    for _, x, y, sx, sy, ms, _, _ in labels:
        px, py = tr.transform((x, y))
        h = (ms / 2 + 1) * k
        boxes.append((px - h, py - h, px + h, py + h))
        (x0, _), (x1, _) = tr.transform((x - sx, y)), tr.transform((x + sx, y))
        (_, y0), (_, y1) = tr.transform((x, y - sy)), tr.transform((x, y + sy))
        boxes.append((x0 - 2, py - 2, x1 + 2, py + 2))
        boxes.append((px - 2, y0 - 2, px + 2, y1 + 2))
    axbox = ax.get_window_extent(r)

    def inter(a, b):
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0
    dirs = {"ur": (1, 1), "dr": (1, -1), "ul": (-1, 1), "dl": (-1, -1), "r": (1, 0), "l": (-1, 0), "u": (0, 1), "d": (0, -1)}
    for text, x, y, sx, sy, ms, col, pref in labels:
        order = ["dl", "dr", "ul", "ur", "r", "l", "d", "u"] if pref == "sw" else ["ur", "dr", "ul", "dl", "r", "l", "u", "d"]
        px, py = tr.transform((x, y))
        best = None
        for mult in (1.0, 1.9, 2.8, 3.8):
            for dname in order:
                ux, uy = dirs[dname]
                off = (ms / 2 + 2) * mult
                t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(ux * off, uy * off), fontsize=7,
                                color=col, ha="left" if ux > 0 else ("right" if ux < 0 else "center"),
                                va="bottom" if uy > 0 else ("top" if uy < 0 else "center"))
                bb = t.get_window_extent(r)
                bx = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
                out = (max(0, axbox.x0 - bx[0]) + max(0, bx[2] - axbox.x1) + max(0, axbox.y0 - bx[1])
                       + max(0, bx[3] - axbox.y1))
                cost = sum(inter(bx, b) for b in boxes) + 1e4 * out
                if best is None or cost < best[0]:
                    if best is not None:
                        best[1].remove()
                    best = (cost, t, bx)
                else:
                    t.remove()
                if cost == 0:
                    break
            if best[0] == 0:
                break
        boxes.append(best[2])


def fig_adapt_ret(S, seeds, out):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    counts = []
    zorder = {"sw": 4, "kdd": 3, "ours": 5}                       # 滑窗画在 KDD 上面（两者在 Gas / Electricity 上几乎重合）
    panel_labels = []
    panels = []                                                   # 先收集数据，再按四个子图里最大的范围统一刻度
    for ds in HOLM_DATASETS:
        arms = [(ds, m) for m in DATASET_METHODS[ds]]
        if ds == "insects_gradual":
            arms += [("insects_gradual_start", m) for m in DATASET_METHODS["insects_gradual_start"]]
        pts = []
        for d_, m in arms:
            ax_, sx = mean_sd(vals(S, d_, m, "ADAPT", seeds))
            ay, sy = mean_sd(vals(S, d_, m, "RET", seeds))
            if not (np.isfinite(ax_) and np.isfinite(ay)):
                continue
            both = np.isfinite(vals(S, d_, m, "ADAPT", seeds)) & np.isfinite(vals(S, d_, m, "RET", seeds))
            counts.append(int(both.sum()))
            pts.append((d_, m, 100 * ax_, 100 * ay, 100 * sx, 100 * sy))
        panels.append(pts)
    ranges = []                                                   # 每个子图的 (x 范围, y 范围)，含 ±SD
    for pts in panels:
        if not pts:
            ranges.append(None)
            continue
        ranges.append(((min(p[2] - p[4] for p in pts), max(p[2] + p[4] for p in pts)),
                       (min(p[3] - p[5] for p in pts), max(p[3] + p[5] for p in pts))))
    ok = [r for r in ranges if r is not None]
    wx = common_span([r[0] for r in ok]) if ok else 1.0
    wy = common_span([r[1] for r in ok]) if ok else 1.0
    for ax, ds, pts, rg in zip(axes, HOLM_DATASETS, panels, ranges):
        labels = []
        for d_, m, x, y, sx, sy in pts:
            role, size, adw = parse_method(m)
            c, mk, _ = role_style(m)
            if d_ == "insects_gradual_start":
                c, mk = C_START, "D"
            ax.errorbar(x, y, xerr=sx, yerr=sy, fmt=mk, color=c,
                        mfc=c if size == 1000 else "white", ms=13 if mk == "*" else 8, capsize=2, lw=1,
                        zorder=zorder[role], alpha=0.75 if role == "sw" else 1.0)   # 滑窗半透明，盖住 KDD 时仍能看到蓝边
            labels.append((str(size), x, y, sx, sy, 13 if mk == "*" else 8, c,
                           "sw" if role == "sw" else "other"))
        if rg is not None:
            ax.set_xlim(*centered_limits(*rg[0], wx, cap=ACC_CAP))        # 准确率不会超过 100：超出时整体下移，宽度不变
            ax.set_ylim(*centered_limits(*rg[1], wy, cap=ACC_CAP))
        ax.xaxis.set_major_locator(MultipleLocator(tick_step(wx)))
        ax.yaxis.set_major_locator(MultipleLocator(tick_step(wy)))
        ax.set_title(TITLE[ds], fontsize=10)
        ax.set_xlabel("ADAPT: balanced prequential accuracy (%)")
        ax.grid(alpha=0.3)
        panel_labels.append((ax, labels))
    fig.canvas.draw()                                             # 刻度定下来之后再放标签，才能量出像素位置
    for ax, labels in panel_labels:
        place_labels(fig, ax, labels)
    axes[0].set_ylabel("RET: balanced accuracy on holdout (%)")
    handles = [plt.Line2D([], [], marker="o", color=C_SW, ls="", label="Sliding window"),
               plt.Line2D([], [], marker="s", color=C_KDD, ls="", label="KDD dual memory"),
               plt.Line2D([], [], marker="*", color=C_OURS, ls="", ms=11,
                          label="Ours (regime-balanced; arm (a) on gradual slice)"),
               plt.Line2D([], [], marker="D", color=C_START, ls="", label="Ours, boundary at transition start (b)"),
               plt.Line2D([], [], marker="P", color=C_ADWIN, ls="", label="Ours, ADWIN boundary (c)"),
               plt.Line2D([], [], marker="o", color="0.4", mfc="white", ls="", label="open = 200 rows, filled = 1000 rows")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(f"Adaptation vs retention at equal context size (mean ± SD over seeds; {n_seeds_text(counts)}); "
                 f"all panels share the same scale ({wx:g} pp × {wy:g} pp)", fontsize=11, y=1.02)
    save(fig, out, "fig_adapt_ret")


def noninf_agreement(pts):
    """ADAPT 面板里：双侧 95% 下界与单侧 95% 下界（= 非劣性判定用的）对"下界 > −0.5 pp"的判定是否一致。
    返回 (一致的点数, 点数)。"""
    same = tot = 0
    for (metric, _, _, _), p in pts.items():
        if metric != "ADAPT" or not np.isfinite(p["lo"]) or p["n"] < 2:
            continue
        se = (p["hi"] - p["mean"]) / stats.t.ppf(0.975, p["n"] - 1)
        one = p["mean"] - stats.t.ppf(0.95, p["n"] - 1) * se
        tot += 1
        same += int((p["lo"] > -100 * NONINF_MARGIN) == (one > -100 * NONINF_MARGIN))
    return same, tot


def fig_diff_forest(S, seeds, out):
    """配对差森林图：左 ΔRET，右 ΔADAPT；每行（数据集 × 规模）两个点，颜色表示"相对谁"。"""
    sizes = SIZES
    rows = [(ds, size) for ds in HOLM_DATASETS for size in sizes]
    pts = {}
    for metric in ("RET", "ADAPT"):
        for p in forest_points(S, seeds, HOLM_DATASETS, sizes, metric):
            pts[(metric, p["dataset"], p["size"], p["vs"])] = p
    if not any(np.isfinite(p["mean"]) for p in pts.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True)
    ypos = {r: i for i, r in enumerate(rows)}
    dy = 0.17
    style = {"kdd": (C_KDD, "o", -dy, "Ours − KDD dual memory"),
             "sw": (C_SW, "D", dy, "Ours − sliding window (same size)")}
    for ax, metric, title in zip(axes, ("RET", "ADAPT"), ("Δ RET (retention)", "Δ ADAPT (adaptation)")):
        for (ds, size) in rows:
            for vs, (col, mk, off, _) in style.items():
                p = pts[(metric, ds, size, vs)]
                if not np.isfinite(p["mean"]):
                    continue
                y = ypos[(ds, size)] + off
                xerr = [[p["mean"] - p["lo"]], [p["hi"] - p["mean"]]] if np.isfinite(p["lo"]) else None
                ax.errorbar(p["mean"], y, xerr=xerr, fmt=mk, color=col, mfc=col,        # 两个比较都实心：形状 + 颜色区分，空心/实心在这里不表示规模
                            ms=6, capsize=2.5, lw=1.3, zorder=3)
        ax.axvline(0, color="k", lw=1, zorder=1)
        if metric == "ADAPT":
            ax.axvline(-100 * NONINF_MARGIN, color="0.35", ls="--", lw=1, zorder=1)
        for k in range(1, len(HOLM_DATASETS)):                     # 数据集之间的分隔线
            ax.axhline(2 * k - 0.5, color="0.8", lw=0.8, zorder=0)
        ax.set_title(title + " in percentage points", fontsize=10)
        ax.set_xlabel("difference in pp (paired over seeds; bars = two-sided 95% t interval)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{TITLE[ds]}, {size} rows" for ds, size in rows], fontsize=9)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)                       # 第一行在最上面
    handles = [plt.Line2D([], [], marker=mk, color=col, mfc=col, ls="-", ms=6, label=lab)
               for vs, (col, mk, _, lab) in style.items()]
    handles += [plt.Line2D([], [], color="k", lw=1, label="no difference (0)"),
                plt.Line2D([], [], color="0.35", ls="--", lw=1,
                           label="pre-registered non-inferiority margin (−0.5 pp, right panel)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
    n = [p["n"] for p in pts.values() if p["n"] > 0]
    fig.suptitle(f"Paired differences of Ours (regime-balanced) at equal context size ({n_seeds_text(n)}; "
                 "colour = compared against; gradual slice: arm (a))\n"
                 "note: the two panels have different x scales", fontsize=10.5, y=1.0)
    k_same, k_all = noninf_agreement(pts)
    agree = ("both give the same verdict at all %d points" % k_all if k_same == k_all
             else "the two bounds give different verdicts at %d of %d points" % (k_all - k_same, k_all))
    fig.text(0.5, -0.085, "Bars are two-sided 95% t intervals. The non-inferiority verdict uses the one-sided 95% lower bound "
             "(report Sec. 2.3), which is slightly narrower;\n" + agree + " in the right panel. "
             "Filled markers here only mark the comparison (colour/shape), not the context size as in the ADAPT-RET figure.",
             ha="center", va="top", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, out, "fig_diff_forest")


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
            ax.set_title(f"{lab}, {size} rows", fontsize=9, pad=12 if has_zone else 6)
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
                # 过渡区：画在坐标轴上方的灰条 + 区内两条细竖线，不盖在阶段颜色上（否则 R0 的绿色会变成灰绿）
                ax.add_patch(Rectangle((int(z[0]), 1.0), int(z[1]) - int(z[0]), 0.045, transform=ax.get_xaxis_transform(),
                                       facecolor="0.7", edgecolor="0.35", lw=0.6, clip_on=False))
                for zz in (z[0], z[1]):
                    ax.axvline(int(zz), color="0.35", lw=0.8, zorder=2)
            if role == "ours_adwin":                          # ADWIN 报警位置（新阶段第一行）：粉线加深色描边，在绿/橙底上也看得见
                for b in np.asarray(d["closures"]):
                    ax.axvline(int(b), color=C_ADWIN, ls=(0, (1.2, 1.2)), lw=2.2, zorder=4,
                               path_effects=[pe.Stroke(linewidth=3.8, foreground="0.15"), pe.Normal()])
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
        note += ", grey bar on top = gradual transition"
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
    fig_diff_forest(S, seeds, args.out)
    fig_regime_ret(S, runs, seeds, args.out)
    for ds in HOLM_DATASETS:
        fig_pool(runs, ds, args.pool_seed, args.out)
    fig_transition(runs, seeds, args.out)
    print(f"wrote {sorted(WRITTEN)} to {args.out}")


if __name__ == "__main__":
    main()
