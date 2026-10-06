"""
Phase 1 基线评估脚本

在合成数据集上运行 vanilla TabPFN（滑动窗口），
记录逐步预测误差，生成动机图：
  "TabPFN 在漂移点处误差骤增"

用法：
    # 在仓库根目录下运行
    python scripts/run_baselines.py --dataset rotating_boundary
    python scripts/run_baselines.py --dataset regime_switching
    python scripts/run_baselines.py --dataset combined_drift --n_samples 5000
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 使环境不依赖 display（服务器 / 无图形界面时需要）
matplotlib.use("Agg")

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.real_world import load_real_world
from src.data.synthetic import make_dataset
from src.data.temporal_loader import CompositeWindowLoader, DualMemoryLoader, TemporalWindowLoader
from src.models.slow_prior import SlowPrior
from src.utils.metrics import summarize_results, window_accuracy
from src.utils.seeding import set_global_seed
from src.utils.atomic_io import atomic_savez
from src.utils.device import describe, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="regime_switching",
        choices=["rotating_boundary", "regime_switching", "combined_drift",
                 "electricity", "insects"],
        help="数据集名称（合成 3 个 + 真实 2 个）",
    )
    parser.add_argument(
        "--dataset_source", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="数据来源：synthetic（默认，向后兼容）或 real（Phase 5）",
    )
    parser.add_argument(
        "--segment_id", type=str, default="start",
        choices=["start", "middle", "end",
                 "early", "mid", "late_pre", "late_post",
                 "d1_14352", "d2_19500", "d3_33240", "d4_double", "d0_control"],
        help="real 数据集 segment 选择（A+: start/middle/end；B1+ aligned: early/mid/late_pre/late_post）",
    )
    parser.add_argument(
        "--segment_size", type=int, default=5000,
        help="real 数据集的 segment 大小（默认 5000，A+ 协议；aligned 模式下被忽略）",
    )
    parser.add_argument(
        "--insects_aligned", action="store_true",
        help="Insects 用 4 个 drift-aligned segments（Phase 5 Stage B1+）；其他 dataset 忽略",
    )
    parser.add_argument(
        "--aligned_v2", action="store_true",
        help="Insects 用 Phase 5.5 的官方变点居中 5 段（d1_14352/d2_19500/d3_33240/"
             "d4_double/d0_control）；与 --insects_aligned 互斥",
    )
    parser.add_argument(
        "--label_scheme", type=str, default="pair_parity",
        choices=["pair_parity", "pair_A_vs_B"],
        help="Insects 标签方案：pair_parity（默认，6 类按奇偶折叠，保留全部行）/ "
             "pair_A_vs_B（只留 {2,3} vs {4,5}，任务更难但丢约 1/3 行）",
    )
    parser.add_argument("--n_samples", type=int, default=5000, help="样本总数")
    parser.add_argument("--n_features", type=int, default=10, help="特征维度（rotating_boundary 建议用 2）")
    parser.add_argument("--regime_length", type=int, default=500, help="每个体制持续的样本数（regime_switching 专用）")
    parser.add_argument("--n_regimes", type=int, default=3, help="体制数量（regime_switching 专用）")
    parser.add_argument("--drift_speed", type=float, default=0.003, help="决策边界旋转速度（rotating_boundary 专用）")
    parser.add_argument("--context_size", type=int, default=300, help="TabPFN 上下文窗口大小")
    parser.add_argument("--window_size", type=int, default=100, help="评估窗口大小")
    parser.add_argument("--n_estimators", type=int, default=4, help="TabPFN 集成数（越小越快，CPU 建议 4）")
    parser.add_argument("--max_eval_steps", type=int, default=None, help="最多评估多少步（调试用，None = 全量）")
    parser.add_argument("--context_loader", type=str, default="sliding",
                        choices=["sliding", "composite", "dual"],
                        help="sliding（默认）/ composite（需 --fixed_ratio>0）/ dual（KDD 2026 长短双记忆基线）")
    parser.add_argument("--short_ratio", type=float, default=0.5,
                        help="dual 模式下短期库占 context 的比例")
    parser.add_argument("--long_max_age", type=int, default=2000,
                        help="dual 模式下长期库样本年龄上限（步）")
    parser.add_argument("--fixed_ratio", type=float, default=0.0,
                        help="固定池占 context 比例（0=纯滑动窗口，0.67=200固定+100滑动）")
    parser.add_argument("--oracle_context_reset", action="store_true",
                        help="在已知 drift_points 处强制截断 context，仅保留 post-drift 样本")
    parser.add_argument("--reset_size", type=int, default=50,
                        help="oracle reset 时保留的最近样本数（context 从该值增长回 context_size）")
    parser.add_argument("--results_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="输出文件名后缀（默认按 dataset/fixed_ratio 自动生成）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"],
                        help="auto=有 NVIDIA 显卡用 cuda 否则 cpu（mps 实测更慢，不支持）")
    return parser.parse_args()


def run_tabpfn_baseline(args):
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {args.n_samples} | 上下文大小: {args.context_size} | fixed_ratio: {args.fixed_ratio}")
    print(f"TabPFN n_estimators: {args.n_estimators}")
    print(f"{'='*60}\n")

    # 1. 生成 / 加载数据
    if args.dataset_source == "real":
        aligned_tag = " [aligned]" if args.insects_aligned else ""
        print(f"加载真实数据集 {args.dataset} segment={args.segment_id}{aligned_tag}...")
        dataset = load_real_world(
            args.dataset, segment_id=args.segment_id, size=args.segment_size,
            insects_aligned=args.insects_aligned,
            aligned_v2=args.aligned_v2,
            label_scheme=args.label_scheme,
        )
        args.n_features = dataset.X.shape[1]
    else:
        print("正在生成合成数据...")
        kwargs = {"n_samples": args.n_samples, "random_seed": args.seed, "n_features": args.n_features}
        if args.dataset == "rotating_boundary":
            kwargs["drift_speed"] = args.drift_speed
        elif args.dataset == "regime_switching":
            kwargs["regime_length"] = args.regime_length
            kwargs["n_regimes"] = args.n_regimes
        dataset = make_dataset(args.dataset, **kwargs)

    print(f"  总样本数: {len(dataset.X)}")
    print(f"  特征维度: {dataset.X.shape[1]}")
    print(f"  类别分布: {np.bincount(dataset.y)}")
    print(f"  漂移点 ({len(dataset.drift_points)} 个): {dataset.drift_points[:10]}")

    # 2. 初始化加载器和模型
    # --context_loader 与 --fixed_ratio 的对应关系必须一致，不做静默猜测
    if args.context_loader == "sliding" and args.fixed_ratio > 0:
        args.context_loader = "composite"          # 向后兼容：历史命令只给 --fixed_ratio
    if args.context_loader == "composite" and not (0 < args.fixed_ratio < 1):
        raise SystemExit("[error] --context_loader composite 需要 0 < --fixed_ratio < 1")
    if args.context_loader == "dual" and args.fixed_ratio > 0:
        raise SystemExit("[error] --context_loader dual 与 --fixed_ratio 互斥")
    if args.context_loader == "dual":
        loader = DualMemoryLoader(
            dataset.X, dataset.y,
            context_size=args.context_size,
            short_ratio=args.short_ratio,
            max_age=args.long_max_age,
            step_size=1,
        )
        print(f"  双记忆: 短期库 {loader.short_capacity} + 长期库 {loader.long_capacity} (max_age={args.long_max_age})")
    elif args.context_loader == "composite":
        loader = CompositeWindowLoader(
            dataset.X,
            dataset.y,
            context_size=args.context_size,
            fixed_ratio=args.fixed_ratio,
            step_size=1,
            random_seed=args.seed,
        )
        fixed_size = int(args.context_size * args.fixed_ratio)
        sliding_size = args.context_size - fixed_size
        print(f"  组合窗口: 固定池 {fixed_size} + 滑动窗口 {sliding_size}")
    else:
        loader = TemporalWindowLoader(
            dataset.X,
            dataset.y,
            context_size=args.context_size,
            step_size=1,
        )
    model = SlowPrior(device=args.device, n_estimators=args.n_estimators)

    # 3. Prequential 评估循环
    all_preds = []
    all_labels = []
    step_times = []

    total_steps = len(loader)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)

    print(f"\n开始 Prequential 评估（共 {total_steps} 步）...")
    print("注意：首步需要下载/加载 TabPFN 权重，可能需要几秒钟。\n")

    drift_points_set = set(int(d) for d in dataset.drift_points)
    last_drift_t = None
    n_truncations = 0

    start_total = time.time()
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        X_ctx, y_ctx = batch.X_ctx, batch.y_ctx

        if args.oracle_context_reset:
            t_abs = batch.t
            if t_abs in drift_points_set:
                last_drift_t = t_abs
                print(f"  [oracle] drift @ t={t_abs} → reset context to last {args.reset_size}")
            if last_drift_t is not None:
                max_ctx = min(args.reset_size + (t_abs - last_drift_t), args.context_size)
                if max_ctx < len(X_ctx):
                    X_ctx = X_ctx[-max_ctx:]
                    y_ctx = y_ctx[-max_ctx:]
                    n_truncations += 1

        t0 = time.time()
        _, pred_label = model.predict(X_ctx, y_ctx, batch.X_query)
        step_times.append(time.time() - t0)

        all_preds.append(pred_label[0])
        all_labels.append(batch.y_query[0])

        # 进度提示
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - start_total
            avg_step = np.mean(step_times[-100:])
            remaining = avg_step * (total_steps - i - 1)
            print(
                f"  步 {i+1:5d}/{total_steps} | "
                f"当前准确率: {np.mean(np.array(all_preds) == np.array(all_labels)):.3f} | "
                f"均步耗时: {avg_step:.3f}s | "
                f"预计剩余: {remaining/60:.1f}min"
            )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total_time = time.time() - start_total

    # 4. 计算指标
    offset = args.context_size
    results = summarize_results(
        all_preds,
        all_labels,
        drift_points=dataset.drift_points,
        window_size=args.window_size,
        offset=offset,
    )

    print(f"\n{'='*60}")
    print("评估结果 (TabPFN Baseline):")
    print(f"  总体 Prequential 准确率: {results['overall_acc']:.4f}")
    if results["pre_drift_acc"] is not None:
        print(f"  漂移前平均准确率:         {results['pre_drift_acc']:.4f}")
    if results["post_drift_acc"] is not None:
        print(f"  漂移后平均准确率:         {results['post_drift_acc']:.4f}")
    if results["avg_adaptation_speed"] is not None:
        print(f"  平均适应速度 (步):        {results['avg_adaptation_speed']:.1f}")
    print(f"  总耗时:                   {total_time:.1f}s")
    print(f"  均步耗时:                 {np.mean(step_times):.3f}s")
    print(f"{'='*60}\n")

    # 5. 绘图：误差随时间变化（动机图）
    win_accs = results["window_accs"]
    t_axis = np.arange(len(win_accs)) + offset + args.window_size // 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 上图：滑动窗口准确率
    axes[0].plot(t_axis, win_accs, color="steelblue", linewidth=1.2, label="TabPFN (baseline)")
    axes[0].set_ylabel("Window Accuracy", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    ratio_info = f", fixed_ratio={args.fixed_ratio}" if args.fixed_ratio > 0 else ""
    axes[0].set_title(
        f"TabPFN Baseline on '{args.dataset}' — Drift Detection\n"
        f"(context_size={args.context_size}, window_size={args.window_size}{ratio_info})",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 标注漂移点
    for dp in dataset.drift_points:
        if offset <= dp < offset + len(win_accs) + args.window_size:
            axes[0].axvline(dp, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].axvline(dataset.drift_points[0] if dataset.drift_points else -1,
                    color="red", linestyle="--", alpha=0.5, linewidth=1, label="Drift Point")
    axes[0].legend(fontsize=10)

    # 下图：逐步误差（1 = 错误，0 = 正确），用平滑曲线展示
    error_series = (all_preds != all_labels).astype(float)
    # 用卷积平滑
    smooth_kernel = np.ones(args.window_size) / args.window_size
    smoothed_error = np.convolve(error_series, smooth_kernel, mode="valid")
    t_error = np.arange(len(smoothed_error)) + offset + args.window_size // 2

    axes[1].fill_between(t_error, smoothed_error, alpha=0.4, color="tomato", label="Error Rate (smoothed)")
    axes[1].plot(t_error, smoothed_error, color="tomato", linewidth=1)
    axes[1].set_ylabel("Error Rate", fontsize=12)
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    if args.out_tag is not None:
        stem = args.out_tag
    else:
        ratio_suffix = f"_fr{args.fixed_ratio:.2f}" if args.fixed_ratio > 0 else ""
        stem = f"baseline_{args.dataset}{ratio_suffix}"
    out_path = os.path.join(args.results_dir, f"{stem}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {out_path}")

    # 6. 保存数值结果
    atomic_savez(
        os.path.join(args.results_dir, f"{stem}.npz"),
        context_loader=np.array([args.context_loader]),   # 自证：这次到底用了哪种 context
        device=np.array([args.device]),
        n_estimators=np.array([args.n_estimators]),
        context_size=np.array([args.context_size]),                   # 自证：这次到底在哪个设备上跑
        predictions=all_preds,
        labels=all_labels,
        drift_points=np.array(dataset.drift_points),
        window_accs=win_accs,
        overall_acc=np.array([results["overall_acc"]]),
        oracle_context_reset=np.array([int(args.oracle_context_reset)]),
        reset_size=np.array([args.reset_size]),
        n_truncations=np.array([n_truncations]),
        seed=np.array([args.seed]),
    )
    print(f"数值结果已保存至: results/{stem}.npz")
    if args.oracle_context_reset:
        print(f"  oracle 截断步数累计: {n_truncations}")

    return results


if __name__ == "__main__":
    args = parse_args()
    set_global_seed(args.seed)   # Phase 5.5: 绑定 torch/numpy 全局 RNG
    args.device = resolve_device(args.device)   # NEURAL1_REQUIRE_CUDA=1 时无 GPU 直接报错
    print(f"[device] {describe(args.device)}", flush=True)

    # rotating_boundary 默认用 2D 特征（方便可视化决策边界旋转）
    if args.dataset_source == "synthetic" and args.dataset == "rotating_boundary" and args.n_features == 10:
        print("提示: rotating_boundary 自动使用 n_features=2 以便可视化，如需其他维度请显式指定 --n_features")
        args.n_features = 2

    run_tabpfn_baseline(args)
