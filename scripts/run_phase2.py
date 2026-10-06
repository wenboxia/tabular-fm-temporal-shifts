"""
Phase 2 评估脚本：Level 1 + Level 3（TabPFN + 快速校正器）

对比三种配置：
  1. TabPFN only (baseline, Level 1)
  2. TabPFN + KNN corrector (Level 1 + Level 3, method=knn)
  3. TabPFN + EMA corrector (Level 1 + Level 3, method=ema)

用法：
    # 在仓库根目录下运行
    python scripts/run_phase2.py --dataset regime_switching \
        --n_samples 3000 --regime_length 500 --context_size 200
"""

import argparse
import os
import sys
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.synthetic import make_dataset
from src.data.temporal_loader import TemporalWindowLoader
from src.models.slow_prior import SlowPrior
from src.models.fast_corrector import FastCorrector
from src.utils.metrics import summarize_results, window_accuracy
from src.utils.seeding import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Level1 + Level3 Evaluation")
    parser.add_argument("--dataset", type=str, default="regime_switching",
                        choices=["rotating_boundary", "regime_switching", "combined_drift"])
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--regime_length", type=int, default=500)
    parser.add_argument("--n_regimes", type=int, default=3)
    parser.add_argument("--drift_speed", type=float, default=0.003)
    parser.add_argument("--context_size", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=100, help="工作记忆容量")
    parser.add_argument("--knn_k", type=int, default=5, help="KNN 近邻数")
    parser.add_argument("--ema_alpha", type=float, default=0.15, help="EMA 平滑系数")
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="输出文件名 stem（默认 phase2_{dataset}）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_single(
    loader,
    slow_prior,
    fast_corrector,       # None 表示 baseline（不使用快速校正）
    total_steps,
    label: str,
    bias_window: int = 50,
    min_bias: float = 0.04,
    correction_scale: float = 0.5,
):
    """运行单个配置的 prequential 评估，返回预测数组和标签数组。"""
    all_preds = []
    all_labels = []

    if fast_corrector is not None:
        fast_corrector.reset()

    # 体制级偏差估计用的滑动窗口
    label_window: deque = deque(maxlen=bias_window)
    pred_window: deque  = deque(maxlen=bias_window)

    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        x_query = batch.X_query[0]     # (n_features,)
        y_true_label = int(batch.y_query[0])

        # Level 1: TabPFN 预测
        proba, _ = slow_prior.predict(batch.X_ctx, batch.y_ctx, batch.X_query)
        y_slow_prob = proba[0, 1]      # 正类概率（标量）

        if fast_corrector is not None:
            # Level 3: 快速校正
            correction = fast_corrector.correct(x_query)
            y_final_prob = float(np.clip(y_slow_prob + correction, 0.0, 1.0))

            # 观测真实值后：更新滑动窗口
            label_window.append(float(y_true_label))
            pred_window.append(float(y_slow_prob))

            # 体制级偏差：平滑信号，消除单步标签噪声
            if len(label_window) >= bias_window:
                regime_bias = float(np.mean(label_window) - np.mean(pred_window))
            else:
                regime_bias = 0.0

            # 仅在偏差足够显著时才更新缓冲区（避免噪声污染）
            if abs(regime_bias) > min_bias:
                calibrated_error = correction_scale * regime_bias
                fast_corrector.update(x_query, calibrated_error)
        else:
            y_final_prob = y_slow_prob

        y_pred_label = int(y_final_prob >= 0.5)
        all_preds.append(y_pred_label)
        all_labels.append(y_true_label)

        if (i + 1) % 500 == 0:
            acc = np.mean(np.array(all_preds) == np.array(all_labels))
            print(f"  [{label}] 步 {i+1:5d}/{total_steps} | 准确率: {acc:.3f}")

    return np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    set_global_seed(args.seed)   # Phase 5.5: 绑定 torch/numpy 全局 RNG
    os.makedirs(args.results_dir, exist_ok=True)

    if args.dataset == "rotating_boundary" and args.n_features == 10:
        args.n_features = 2

    print(f"\n{'='*60}")
    print(f"Phase 2: TabPFN + 快速校正器")
    print(f"数据集: {args.dataset} | 样本数: {args.n_samples}")
    print(f"缓冲区大小: {args.buffer_size} | KNN k={args.knn_k} | EMA α={args.ema_alpha}")
    print(f"{'='*60}\n")

    # ---- 生成数据 ----
    kwargs = {"n_samples": args.n_samples, "random_seed": args.seed, "n_features": args.n_features}
    if args.dataset == "rotating_boundary":
        kwargs["drift_speed"] = args.drift_speed
    elif args.dataset == "regime_switching":
        kwargs["regime_length"] = args.regime_length
        kwargs["n_regimes"] = args.n_regimes
    dataset = make_dataset(args.dataset, **kwargs)
    print(f"漂移点 ({len(dataset.drift_points)} 个): {dataset.drift_points}")

    loader = TemporalWindowLoader(
        dataset.X, dataset.y,
        context_size=args.context_size,
        step_size=1,
    )
    total_steps = len(loader)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)

    # ---- 初始化模型（只建一个 SlowPrior，三次评估共用）----
    slow_prior = SlowPrior(device="cpu", n_estimators=args.n_estimators)

    configs = [
        ("TabPFN only",       None),
        ("TabPFN + KNN",      FastCorrector(args.buffer_size, "knn", k=args.knn_k)),
        ("TabPFN + EMA",      FastCorrector(args.buffer_size, "ema", alpha=args.ema_alpha)),
    ]

    all_results = {}
    all_win_accs = {}

    for label, corrector in configs:
        print(f"\n--- 运行: {label} ---")
        t0 = time.time()
        preds, labels = run_single(loader, slow_prior, corrector, total_steps, label)
        elapsed = time.time() - t0

        results = summarize_results(
            preds, labels,
            drift_points=dataset.drift_points,
            window_size=args.window_size,
            offset=args.context_size,
        )
        all_results[label] = results
        all_win_accs[label] = results["window_accs"]

        print(f"  总体准确率: {results['overall_acc']:.4f} | "
              f"漂移前: {results['pre_drift_acc'] or 0:.4f} | "
              f"漂移后: {results['post_drift_acc'] or 0:.4f} | "
              f"适应速度: {results['avg_adaptation_speed'] or 'N/A'} | "
              f"耗时: {elapsed:.0f}s")

    # ---- 打印汇总表 ----
    print(f"\n{'='*60}")
    print("Phase 2 汇总对比:")
    print(f"{'模型':<20} {'总体准确率':>10} {'漂移后准确率':>12} {'适应速度(步)':>12}")
    print("-" * 60)
    for label, res in all_results.items():
        speed = f"{res['avg_adaptation_speed']:.1f}" if res['avg_adaptation_speed'] else "N/A"
        print(f"{label:<20} {res['overall_acc']:>10.4f} "
              f"{(res['post_drift_acc'] or 0):>12.4f} {speed:>12}")
    print(f"{'='*60}\n")

    # ---- 绘图 ----
    colors = {"TabPFN only": "steelblue", "TabPFN + KNN": "darkorange", "TabPFN + EMA": "seagreen"}
    offset = args.context_size

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # 上图：窗口准确率对比
    for label, win_accs in all_win_accs.items():
        t_axis = np.arange(len(win_accs)) + offset + args.window_size // 2
        lw = 1.8 if "KNN" in label or "EMA" in label else 1.2
        ls = "-" if "KNN" in label else ("--" if "EMA" in label else ":")
        axes[0].plot(t_axis, win_accs, color=colors[label], linewidth=lw,
                     linestyle=ls, label=label, alpha=0.9)

    for dp in dataset.drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)
    axes[0].axvline(-1, color="red", linestyle="--", alpha=0.4, linewidth=1, label="Drift Point")

    axes[0].set_ylabel("Window Accuracy", fontsize=12)
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title(
        f"Phase 2: Level1+Level3 on '{args.dataset}'\n"
        f"(context={args.context_size}, buffer={args.buffer_size}, "
        f"knn_k={args.knn_k}, ema_α={args.ema_alpha})",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 下图：各模型的窗口准确率差值（vs baseline）
    baseline_acc = all_win_accs["TabPFN only"]
    min_len = min(len(v) for v in all_win_accs.values())

    for label in ["TabPFN + KNN", "TabPFN + EMA"]:
        win_accs = all_win_accs[label]
        diff = win_accs[:min_len] - baseline_acc[:min_len]
        t_axis = np.arange(min_len) + offset + args.window_size // 2
        axes[1].plot(t_axis, diff, color=colors[label], linewidth=1.5,
                     linestyle="-" if "KNN" in label else "--",
                     label=f"{label} − Baseline", alpha=0.9)

    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="-")
    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)

    axes[1].set_ylabel("Accuracy Gain vs Baseline", fontsize=12)
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    stem = args.out_tag if args.out_tag is not None else f"phase2_{args.dataset}"
    out_path = os.path.join(args.results_dir, f"{stem}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {out_path}")

    # ---- 保存数值结果 ----
    np.savez(
        os.path.join(args.results_dir, f"{stem}.npz"),
        drift_points=np.array(dataset.drift_points),
        **{f"win_accs_{k.replace(' ', '_').replace('+', 'p')}": v
           for k, v in all_win_accs.items()},
        **{f"overall_acc_{k.replace(' ', '_').replace('+', 'p')}":
           np.array([v["overall_acc"]]) for k, v in all_results.items()},
    )
    print(f"数值结果已保存至: results/{stem}.npz")


if __name__ == "__main__":
    main()
