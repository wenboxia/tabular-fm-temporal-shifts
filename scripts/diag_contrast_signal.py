"""
Phase 5.5 Step 4 — 对比信号诊断：对比信号 vs 错误指示器

对比信号：滑窗 0（不适应）与滑窗 300（适应）两路预测的差值作为检测器的输入，用它刻画 drift。

即：不再把"适应之后的误差"喂给检测器，而是把"适应 vs 不适应两路预测的差值"喂进去。

**成本**：两路预测都用 TabPFN 现算。stale 路的 context 全程固定，所以整段可以**一次批量**算完
（实测逐样本独立，误差 3e-7；批量比逐条快 7.4×）；结果缓存在 results/contrast_signal_cache.npz。

评价口径（不是只看一个均值差）：
  - 检测**延迟**：官方变点之后多久报警
  - **误报**：变点容差窗之外的报警数（d0_control 段全部报警都是误报）
  - 三种信号并排：contrast（本诊断新增）/ indicator（现状）/ P(y_pred=1)

用法：
    python scripts/diag_contrast_signal.py --segments d3_33240,d4_double,d0_control
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.real_world import _INSECTS_ALIGNED_V2_SEGMENTS, load_real_world
from src.drift.error_detector import make_detector
from src.models.slow_prior import SlowPrior
from src.utils.seeding import set_global_seed


def compute_signals(X, y, context_size, stale_size, n_estimators, batch=256):
    """逐步跑 sliding 路，一次性批量跑 stale 路，返回三条信号。

    stale 路 = 固定用**该段最早 stale_size 个样本**做 context，永不更新
              → 代表"不做 in-context 适应"。
    sliding 路 = 当前滑动 context（现有行为）→ 代表"做 in-context 适应"。
    """
    sp = SlowPrior(n_estimators=n_estimators)
    T = len(X)
    idx = np.arange(context_size, T)

    # stale 路：context 固定 ⇒ 全部查询点可以一次批量算
    t0 = time.time()
    X_stale, y_stale = X[:stale_size], y[:stale_size]
    p_stale = np.empty(len(idx), dtype=np.float64)
    for s in range(0, len(idx), batch):
        chunk = idx[s: s + batch]
        p_stale[s: s + len(chunk)] = sp.predict_proba(X_stale, y_stale, X[chunk])[:, 1]
    t_stale = time.time() - t0

    # sliding 路：每步 context 都变 ⇒ 只能逐步
    t0 = time.time()
    p_slide = np.empty(len(idx), dtype=np.float64)
    for j, t in enumerate(idx):
        p_slide[j] = sp.predict_proba(X[t - context_size: t], y[t - context_size: t],
                                      X[t: t + 1])[0, 1]
    t_slide = time.time() - t0

    y_true = y[idx]
    return {
        "t": idx,
        "contrast_prob": np.abs(p_stale - p_slide),
        "contrast_hard": ((p_stale >= 0.5) != (p_slide >= 0.5)).astype(float),
        "indicator": ((p_slide >= 0.5) != y_true).astype(float),
        "pred1": (p_slide >= 0.5).astype(float),
        "_timing": (t_stale, t_slide),
    }


def score(alarms, drifts, tolerance, pre_tolerance=0):
    """命中判定窗口 = [d - pre_tolerance, d + tolerance]。

    允许**早于**标注点的报警算命中：Souza 的变点标的是温度**设定**的切换时刻，
    传感器读数在切换前后若干步内就开始变，实测 d3_33240 上三种信号都在
    标注点前 22–45 步报警。把它们记成误报会系统性低估检测器。
    """
    hits, delays, used = 0, [], set()
    for d in drifts:
        cand = [a for a in alarms if -pre_tolerance <= a - d <= tolerance and a not in used]
        if cand:
            hits += 1
            used.add(cand[0])
            delays.append(cand[0] - d)
    return {"n_alarms": len(alarms), "hits": hits, "n_drifts": len(drifts),
            "false_alarms": len(alarms) - hits, "delays": delays}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=str,
                    default="d3_33240,d4_double,d0_control",
                    help="默认只用**有效**的漂移段 + 无漂移对照段；"
                         "d1_14352 与 pair_A_vs_B 下的 d2_19500 漂移前是单类，已排除")
    ap.add_argument("--label_scheme", type=str, default="pair_A_vs_B",
                    choices=["pair_parity", "pair_A_vs_B"])
    ap.add_argument("--context_size", type=int, default=200)
    ap.add_argument("--stale_size", type=int, default=200,
                    help="stale 路的固定 context 大小（取该段最早这么多样本）")
    ap.add_argument("--n_estimators", type=int, default=1,
                    help="诊断用 1 即可（只比信号形状，不比绝对准确率）")
    ap.add_argument("--delta", type=float, default=0.002)
    ap.add_argument("--cooldown", type=int, default=80)
    ap.add_argument("--tolerance", type=int, default=600,
                    help="报警落在变点后多少步内算命中")
    ap.add_argument("--pre_tolerance", type=int, default=100,
                    help="报警早于变点多少步内仍算命中（标注点是温度设定切换时刻，"
                         "传感器读数会提前变）")
    ap.add_argument("--cache", type=str, default="results/contrast_signal_cache.npz",
                    help="信号缓存；命中则跳过 TabPFN，改判据可零成本重算")
    ap.add_argument("--recompute", action="store_true",
                    help="忽略缓存，强制重算信号")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out", type=str, default="results/contrast_signal_diag.md")
    ap.add_argument("--results_dir", type=str, default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    for s in segments:
        if s not in _INSECTS_ALIGNED_V2_SEGMENTS:
            raise SystemExit(f"[error] 未知 segment {s!r}；可选 {_INSECTS_ALIGNED_V2_SEGMENTS}")

    SIGNALS = ["contrast_prob", "contrast_hard", "indicator", "pred1"]
    rows, per_seg = [], []

    for seg in segments:
        ds = load_real_world("insects", segment_id=seg, aligned_v2=True,
                             label_scheme=args.label_scheme)   # 退化段会在此报错
        X, y = ds.X, ds.y
        if args.max_steps is not None:
            X, y = X[: args.context_size + args.max_steps], y[: args.context_size + args.max_steps]
        drifts = [d for d in ds.drift_points if d >= args.context_size]
        print(f"[{seg}] n={len(X)} drifts(local)={drifts} label_scheme={args.label_scheme}")

        ckey = f"{seg}|{args.label_scheme}|{args.context_size}|{args.stale_size}|" \
               f"{args.n_estimators}|{args.max_steps}"
        sig = None
        if not args.recompute and os.path.exists(args.cache):
            z = np.load(args.cache, allow_pickle=True)
            if ckey in z.files:
                sig = {k: v for k, v in z[ckey].item().items()}
                print("  [cache] 命中，跳过 TabPFN")
        if sig is None:
            sig = compute_signals(X, y, args.context_size, args.stale_size,
                                  args.n_estimators)
            ts, tl = sig["_timing"]
            print(f"  stale(batched) {ts:.0f}s vs sliding(per-step) {tl:.0f}s")
            store = {}
            if os.path.exists(args.cache):
                z = np.load(args.cache, allow_pickle=True)
                store = {k: z[k] for k in z.files}
            store[ckey] = np.array(sig, dtype=object)
            np.savez(args.cache, **store)

        for name in SIGNALS:
            det = make_detector("river", delta=args.delta, cooldown=args.cooldown)
            alarms = [int(sig["t"][i]) for i, v in enumerate(sig[name])
                      if det.update(float(v))]
            sc = score(alarms, drifts, args.tolerance, args.pre_tolerance)
            # 变点前后 ±200 步的均值差（与 γ 诊断同口径，便于直接对比）
            shifts = []
            for d in drifts:
                j = int(np.searchsorted(sig["t"], d))
                pre, post = sig[name][max(0, j - 200): j], sig[name][j: j + 200]
                if len(pre) and len(post):
                    shifts.append(abs(float(post.mean() - pre.mean())))
            rows.append({
                "segment": seg, "signal": name, **sc,
                "max_shift": max(shifts) if shifts else None,
                "alarms": alarms,
            })
        per_seg.append((seg, sig, drifts))

    # ── 图：每段一张，四条信号的滑动均值 + 变点 + 报警 ──────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    for seg, sig, drifts in per_seg:
        fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(11, 2.2 * len(SIGNALS)),
                                 sharex=True)
        for ax, name in zip(axes, SIGNALS):
            v = sig[name]
            w = min(100, max(5, len(v) // 20))
            sm = np.convolve(v, np.ones(w) / w, mode="valid")
            ax.plot(sig["t"][: len(sm)], sm, linewidth=1.2, label=f"{name} (w={w})")
            for d in drifts:
                ax.axvline(d, color="red", linestyle="--", alpha=0.6)
            for a in next(r for r in rows if r["segment"] == seg and r["signal"] == name)["alarms"]:
                ax.axvline(a, color="purple", linestyle=":", alpha=0.7)
            ax.set_ylabel(name, fontsize=8)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("local t   (red = official drift, purple = river ADWIN alarm)")
        fig.suptitle(f"Contrast-signal diagnostic — insects {seg} ({args.label_scheme})",
                     fontsize=10)
        fig.tight_layout()
        png = os.path.join(args.results_dir, f"contrast_signal_{seg}_{args.label_scheme}.png")
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  图: {png}")

    lines = [
        "# Contrast-signal diagnostic (Phase 5.5 Step 4)",
        "",
        "把 **stale（不适应，context 固定在段首）vs sliding（适应）两路预测的差值** 喂给检测器，"
        "对比现有的 0/1 错误指示器。",
        "",
        f"- label_scheme = `{args.label_scheme}`，context = {args.context_size}，"
        f"stale context = {args.stale_size}，n_estimators = {args.n_estimators}",
        f"- 检测器 = river ADWIN，δ = {args.delta}，cooldown = {args.cooldown}，"
        f"命中窗口 = 变点前 {args.pre_tolerance} 步 ~ 变点后 {args.tolerance} 步"
        "（允许早报：标注点是温度**设定**的切换时刻，读数会提前变）",
        "- Insects 变点用 **Souza 2020 官方坐标**；`d0_control` 段无变点，那里的报警全是误报",
        "",
        "| segment | signal | alarms | recall | false alarms | median delay | max shift |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rec = f"{r['hits']}/{r['n_drifts']}" if r["n_drifts"] else "—"
        dly = f"{np.median(r['delays']):.0f}" if r["delays"] else "—"
        ms = f"{r['max_shift']:.3f}" if r["max_shift"] is not None else "—"
        lines.append(f"| {r['segment']} | `{r['signal']}` | {r['n_alarms']} | {rec} | "
                     f"{r['false_alarms']} | {dly} | {ms} |")
    lines += ["", "## 每段报警时刻", "",
              "| segment | signal | drifts (local) | alarms (local) | delays |",
              "|---|---|---|---|---|"]
    for r in rows:
        drifts = [d for d in load_real_world(
            "insects", segment_id=r["segment"], aligned_v2=True,
            label_scheme=args.label_scheme).drift_points
            if d >= args.context_size]
        lines.append(f"| {r['segment']} | `{r['signal']}` | {drifts} | "
                     f"{r['alarms']} | {r['delays']} |")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
