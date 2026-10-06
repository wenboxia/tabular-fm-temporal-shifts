"""
Phase 4 A 评估脚本：MultiTimescaleModel + ADWIN + AdapterLibrary

在 Phase 3C 脚本基础上启用 use_adapter_library=True，
保留窗口准确率 / 门控权重轨迹绘图，新增 detector_events / route_events / 各
adapter 使用次数等诊断字段落盘。

用法：
    # 在仓库根目录下运行
    python scripts/run_phase4_a.py --dataset regime_switching \
        --n_samples 3000 --context_size 200

    # 快速 smoke (~1-2 分钟)
    python scripts/run_phase4_a.py --dataset regime_switching \
        --max_eval_steps 100 --n_samples 3000
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.real_world import load_real_world
from src.data.synthetic import make_dataset
from src.data.temporal_loader import (
    CompositeWindowLoader,
    DualMemoryLoader,
    TemporalWindowLoader,
)
from src.models.multi_timescale import (
    ACTIONS_ON_ALARM,
    DETECTOR_INPUTS,
    TRIGGER_SOURCES,
    MultiTimescaleModel,
)
from src.utils.metrics import summarize_results
from src.utils.seeding import set_global_seed
from src.utils.atomic_io import atomic_savez
from src.utils.device import describe, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 A: MultiTimescaleModel + ADWIN + AdapterLibrary"
    )

    # ── 基础参数（对齐 run_phase3.py）────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="regime_switching",
        choices=["rotating_boundary", "regime_switching", "combined_drift",
                 "electricity", "insects"],
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
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_features", type=int, default=10,
                        help="rotating_boundary 自动改为 2")
    parser.add_argument("--regime_length", type=int, default=500)
    parser.add_argument("--n_regimes", type=int, default=3)
    parser.add_argument("--drift_speed", type=float, default=0.003)
    parser.add_argument("--context_size", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="输出文件 stem（默认 phase4_a_{dataset}）")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"],
                        help="auto=有 NVIDIA 显卡用 cuda 否则 cpu（mps 实测更慢，不支持）")
    parser.add_argument("--seed", type=int, default=42)

    # ── Phase 3 既有参数 ────────────────────────────────────────────────
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--fast_method", type=str, default="knn",
                        choices=["knn", "ema"])
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--ema_alpha", type=float, default=0.15)
    parser.add_argument("--consolidation_threshold", type=float, default=0.05)
    parser.add_argument("--consolidation_window", type=int, default=50)
    parser.add_argument("--consolidation_epochs", type=int, default=10)
    parser.add_argument("--consolidation_cooldown", type=int, default=100)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ── Phase 4 A 新增参数 ──────────────────────────────────────────────
    parser.add_argument("--max_adapters", type=int, default=8,
                        help="AdapterLibrary 容量上限")
    parser.add_argument("--library_fit_threshold", type=float, default=0.5,
                        help="route 时复用现有 adapter 的 MSE 上限（0.5 是 warmstart 默认；"
                             "v1 indicator run 用 0.05 导致 25/25 全 create 0 reuse）")
    parser.add_argument("--library_init_strategy", type=str, default="warm",
                        choices=["warm", "random"],
                        help="新 adapter 初始化策略：warm=自 active 复制（Day 1.5 默认），"
                             "random=永远随机初始化（Day 2 confound-busting）")
    parser.add_argument("--detector_delta", type=float, default=0.002,
                        help="ADWIN 置信参数（小=保守）")
    parser.add_argument("--detector_min_subwindow", type=int, default=30,
                        help="ADWIN 切点两侧最小子窗")
    parser.add_argument("--detector_cooldown", type=int, default=80,
                        help="ADWIN 漂移声明后冷却步数")
    parser.add_argument("--detector_impl", type=str, default="own",
                        choices=["own", "river"],
                        help="own=自写 Hoeffding 版（Phase 4/5 既有行为）；"
                             "river=标准 ADWIN（经验方差界，Phase 5.5）")
    parser.add_argument("--detector_clock", type=int, default=1,
                        help="river ADWIN 每隔多少步检查一次（仅 --detector_impl river 生效）")
    parser.add_argument("--detector_input", type=str, default="indicator",
                        choices=list(DETECTOR_INPUTS),
                        help="喂给检测器的信号：indicator（默认=既有行为）/ pred1（类先验，"
                             "Step 4 诊断召回 2/2 且无需第二路预测）/ contrast_prob / contrast_hard"
                             "（对比信号，需 stale 路，整段一次批量算完）")
    parser.add_argument("--stale_size", type=int, default=200,
                        help="contrast_* 的 stale 路固定 context 大小（取段首这么多样本）")

    # ── Phase 5.5：报警动作策略（判别性对照的自变量）────────────────────
    parser.add_argument("--action_on_alarm", type=str, default="route_adapter",
                        choices=list(ACTIONS_ON_ALARM),
                        help="报警后做什么：route_adapter（默认=Phase 4 A）/ "
                             "context_reset（截断 TabPFN context）/ buffer_clear / none")
    parser.add_argument("--trigger_source", type=str, default="detector",
                        choices=list(TRIGGER_SOURCES),
                        help="detector（默认）或 oracle（用数据集已知漂移点即时触发；"
                             "detector 转入影子模式仍记录延迟）")
    parser.add_argument("--oracle_lag", type=int, default=0,
                        help="oracle 触发相对真实漂移点的滞后步数（0=即时，用于扫检测延迟的影响）")
    parser.add_argument("--reset_size", type=int, default=50,
                        help="context_reset 动作截断后的起始 context 长度")
    parser.add_argument("--min_context_after_reset", type=int, default=20,
                        help="截断后的最小 context 长度（防单类 context 触发常量 fallback）")
    parser.add_argument("--consolidate_on_post_alarm_data", action="store_true",
                        help="把巩固推迟到 alarm_t + consolidation_window，"
                             "确保训练样本全部来自报警之后")

    # ── Phase 5.5：context 管理策略（路径 B + KDD 2026 双记忆基线）────────
    parser.add_argument("--context_loader", type=str, default="sliding",
                        choices=["sliding", "composite", "dual"],
                        help="sliding（默认，纯滑窗）/ composite（固定池+滑窗，需 --fixed_ratio）"
                             " / dual（KDD 2026 长短双记忆基线）")
    parser.add_argument("--fixed_ratio", type=float, default=0.0,
                        help="composite 模式下固定池占 context 的比例 ∈ [0,1)")
    parser.add_argument("--short_ratio", type=float, default=0.5,
                        help="dual 模式下短期库占 context 的比例 ∈ (0,1)")
    parser.add_argument("--long_max_age", type=int, default=2000,
                        help="dual 模式下长期库样本的年龄上限（步）；"
                             "不设会把过时的 P(y|x) 永久钉在 context 里")

    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    args.device = resolve_device(args.device)   # NEURAL1_REQUIRE_CUDA=1 时无 GPU 直接报错
    print(f"[device] {describe(args.device)}", flush=True)   # Phase 5.5: 绑定 torch/numpy 全局 RNG
    os.makedirs(args.results_dir, exist_ok=True)

    # rotating_boundary 必须用 n_features=2
    if args.dataset_source == "synthetic" and args.dataset == "rotating_boundary" and args.n_features == 10:
        args.n_features = 2

    print(f"\n{'='*60}")
    print(f"Phase 4 A: MultiTimescaleModel + ADWIN + AdapterLibrary")
    print(f"数据集: {args.dataset} | 样本数: {args.n_samples} | seed: {args.seed}")
    print(f"context_size: {args.context_size} | window_size: {args.window_size}")
    print(f"buffer_size: {args.buffer_size} | fast_method: {args.fast_method}")
    print(f"max_adapters: {args.max_adapters} | "
          f"library_fit_threshold: {args.library_fit_threshold} | "
          f"init_strategy: {args.library_init_strategy}")
    print(f"detector_impl: {args.detector_impl} | clock: {args.detector_clock} | "
          f"detector_delta: {args.detector_delta} | "
          f"min_subwindow: {args.detector_min_subwindow} | "
          f"cooldown: {args.detector_cooldown}")
    print(f"{'='*60}\n")

    # ── 生成 / 加载数据 ──────────────────────────────────────────────────
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
        kwargs = {
            "n_samples": args.n_samples,
            "random_seed": args.seed,
            "n_features": args.n_features,
        }
        if args.dataset == "rotating_boundary":
            kwargs["drift_speed"] = args.drift_speed
        elif args.dataset == "regime_switching":
            kwargs["regime_length"] = args.regime_length
            kwargs["n_regimes"] = args.n_regimes
        dataset = make_dataset(args.dataset, **kwargs)
    print(f"漂移点 ({len(dataset.drift_points)} 个): {dataset.drift_points}")


    # ── context 管理策略 ─────────────────────────────────────────────────
    # 两个开关重叠时以 --context_loader 为准，并显式报错而不是静默二选一。
    if args.fixed_ratio > 0 and args.context_loader != "composite":
        raise SystemExit(
            f"[error] --fixed_ratio={args.fixed_ratio} 需要 --context_loader composite，"
            f"当前是 {args.context_loader!r}。两个开关重叠时不做静默猜测。"
        )
    if args.context_loader == "composite":
        if not (0.0 < args.fixed_ratio < 1.0):
            raise SystemExit("[error] --context_loader composite 需要 0 < --fixed_ratio < 1")
        loader = CompositeWindowLoader(
            dataset.X, dataset.y,
            context_size=args.context_size,
            fixed_ratio=args.fixed_ratio,
            step_size=1,
            random_seed=args.seed,
        )
        n_fixed = int(args.context_size * args.fixed_ratio)
        print(f"  组合窗口: 固定池 {n_fixed} + 滑动窗 {args.context_size - n_fixed}")
    elif args.context_loader == "dual":
        loader = DualMemoryLoader(
            dataset.X, dataset.y,
            context_size=args.context_size,
            short_ratio=args.short_ratio,
            max_age=args.long_max_age,
            step_size=1,
        )
        print(f"  双记忆: 短期库 {loader.short_capacity} + 长期库 {loader.long_capacity}"
              f" (max_age={args.long_max_age})")
    else:
        loader = TemporalWindowLoader(
            dataset.X, dataset.y,
            context_size=args.context_size,
            step_size=1,
        )
    total_steps = len(loader)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)

    # ── Phase 5.5：oracle 触发时刻（漂移点 + lag），带守卫 ────────────────
    oracle_times = None
    if args.trigger_source == "oracle":
        if not dataset.drift_points:
            raise SystemExit(
                f"[error] --trigger_source oracle 但数据集 {args.dataset} 没有 documented 漂移点"
                "（Electricity 是渐进漂移，无 crisp 切点）。空 oracle 会静默退化成"
                "'永不适应'并写出看似正常的 npz，故直接报错。"
            )
        oracle_times = [int(d) + args.oracle_lag for d in dataset.drift_points]
        # 落在评估范围之外的触发时刻等于没有触发 —— 必须显式暴露
        last_t = args.context_size + total_steps - 1
        in_range = [t for t in oracle_times if args.context_size <= t <= last_t]
        if not in_range:
            raise SystemExit(
                f"[error] oracle 触发时刻 {oracle_times} 全部落在评估范围 "
                f"[{args.context_size}, {last_t}] 之外（--max_eval_steps 太小？）"
            )
        if len(in_range) < len(oracle_times):
            print(f"  [warn] {len(oracle_times) - len(in_range)} 个 oracle 触发点超出评估范围，被忽略")
        oracle_times = in_range
        print(f"  [oracle] 触发时刻 (lag={args.oracle_lag}): {oracle_times}")
    print(f"action_on_alarm: {args.action_on_alarm} | trigger_source: {args.trigger_source}")

    # ── 初始化 MultiTimescaleModel（开 use_adapter_library）──────────────
    model = MultiTimescaleModel(
        input_dim=args.n_features,
        buffer_size=args.buffer_size,
        fast_method=args.fast_method,
        knn_k=args.knn_k,
        ema_alpha=args.ema_alpha,
        consolidation_threshold=args.consolidation_threshold,
        consolidation_window=args.consolidation_window,
        consolidation_epochs=args.consolidation_epochs,
        consolidation_cooldown=args.consolidation_cooldown,
        gate_hidden_dim=args.gate_hidden_dim,
        lr=args.lr,
        device=args.device,
        n_estimators=args.n_estimators,
        # ── Phase 4 A ──
        use_adapter_library=True,
        max_adapters=args.max_adapters,
        library_fit_threshold=args.library_fit_threshold,
        library_init_strategy=args.library_init_strategy,
        detector_delta=args.detector_delta,
        detector_min_subwindow=args.detector_min_subwindow,
        detector_cooldown=args.detector_cooldown,
        detector_impl=args.detector_impl,
        detector_clock=args.detector_clock,
        detector_input=args.detector_input,
        # ── Phase 5.5 ──
        action_on_alarm=args.action_on_alarm,
        trigger_source=args.trigger_source,
        oracle_trigger_times=oracle_times,
        reset_size=args.reset_size,
        min_context_after_reset=args.min_context_after_reset,
        consolidate_on_post_alarm_data=args.consolidate_on_post_alarm_data,
    )

    # ── contrast_* 检测输入：整段 stale 路一次批量算完 ────────────────────
    # stale context 固定为段首 stale_size 个样本，**在 t=0 就全部可得**，
    # 所以批量预计算不构成未来信息泄漏。实测批量 2–5 s，逐步要 304–801 s。
    if args.detector_input.startswith("contrast"):
        import time as _time
        _t0 = _time.time()
        X_stale = dataset.X[: args.stale_size]
        y_stale = dataset.y[: args.stale_size]
        if len(np.unique(y_stale)) < 2:
            raise SystemExit(
                f"[error] stale context（段首 {args.stale_size} 个样本）只有单一类别，"
                "TabPFN 会退化成常量预测，contrast 信号无意义。请增大 --stale_size。"
            )
        q_lo, q_hi = args.context_size, args.context_size + total_steps
        p_stale = np.empty(q_hi - q_lo, dtype=np.float64)
        for _s in range(q_lo, q_hi, 256):
            _e = min(_s + 256, q_hi)
            p_stale[_s - q_lo: _e - q_lo] = model.slow_prior.predict_proba(
                X_stale, y_stale, dataset.X[_s:_e]
            )[:, 1]
        model.set_stale_proba(p_stale, offset=q_lo)
        print(f"  stale 路预计算完成: {len(p_stale)} 步, {_time.time() - _t0:.0f}s")

    # ── Prequential 主循环 ──────────────────────────────────────────────
    predictions: list = []
    labels: list = []
    gate_weights_trajectory: list = []
    active_id_trajectory: list = []

    print(f"开始评估（共 {total_steps} 步）...")
    t0 = time.time()

    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        x_t = batch.X_query[0]
        y_t = int(batch.y_query[0])

        pred, weights = model.step(
            batch.X_ctx, batch.y_ctx, x_t, float(y_t), t=batch.t
        )

        predictions.append(int(pred >= 0.5))
        labels.append(y_t)
        gate_weights_trajectory.append(weights)
        active_id_trajectory.append(model.adapter_library.active_id)

        if (i + 1) % 500 == 0:
            acc = float(np.mean(np.array(predictions) == np.array(labels)))
            print(f"  步 {i+1:5d}/{total_steps} | acc: {acc:.3f} | "
                  f"detect: {len(model.detector_events)} | "
                  f"routes: {len(model.route_events)} | "
                  f"adapters: {model.adapter_library.n_adapters()}")

    elapsed = time.time() - t0

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)
    gate_weights_arr = np.stack(gate_weights_trajectory, axis=0)
    active_id_arr = np.array(active_id_trajectory, dtype=np.int32)
    consolidation_events = model.consolidation_events
    detector_events = model.detector_events
    # route_events: list[(t, action_str, active_id)] — 拆 3 列存 npz
    if model.route_events:
        route_t = np.array([r[0] for r in model.route_events], dtype=np.int64)
        route_action = np.array([r[1] for r in model.route_events])
        route_active_id = np.array([r[2] for r in model.route_events], dtype=np.int32)
    else:
        route_t = np.array([], dtype=np.int64)
        route_action = np.array([], dtype="<U10")
        route_active_id = np.array([], dtype=np.int32)

    # ── 指标 ─────────────────────────────────────────────────────────────
    results = summarize_results(
        preds_arr, labels_arr,
        drift_points=dataset.drift_points,
        window_size=args.window_size,
        offset=args.context_size,
    )
    win_accs = results["window_accs"]

    speed_str = (f"{results['avg_adaptation_speed']:.1f}"
                 if results["avg_adaptation_speed"] is not None else "N/A")

    print(f"\n--- Phase 4 A 结果 ---")
    print(f"  总体准确率: {results['overall_acc']:.4f} | "
          f"漂移前: {results['pre_drift_acc'] or 0:.4f} | "
          f"漂移后: {results['post_drift_acc'] or 0:.4f} | "
          f"适应速度: {speed_str} | 耗时: {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("Phase 4 A 汇总：")
    print(f"{'指标':<20} {'值':>12}")
    print("-" * 35)
    print(f"{'总体准确率':<20} {results['overall_acc']:>12.4f}")
    print(f"{'Balanced Accuracy':<20} {results['balanced_acc']:>12.4f}")
    auc_str = f"{results['auc_roc']:.4f}" if results["auc_roc"] is not None else "N/A"
    print(f"{'AUC-ROC':<20} {auc_str:>12}")
    print(f"{'漂移前准确率':<20} {(results['pre_drift_acc'] or 0):>12.4f}")
    print(f"{'漂移后准确率':<20} {(results['post_drift_acc'] or 0):>12.4f}")
    print(f"{'适应速度 (步)':<20} {speed_str:>12}")
    print(f"{'='*60}")
    print(f"detector 触发次数: {len(detector_events)}")
    print(f"route 事件数:      {len(model.route_events)}")
    print(f"consolidation 数:  {len(consolidation_events)}")
    print(f"最终 adapter 数:   {model.adapter_library.n_adapters()}")
    print(f"adapter usage:     {dict(model.adapter_library.usage)}")
    if model.route_events:
        actions = [r[1] for r in model.route_events]
        from collections import Counter
        print(f"route action 分布: {dict(Counter(actions))}")

    # ── 绘图（与 phase3 同结构 + 漂移检测/路由叠加）───────────────────────
    offset = args.context_size
    t_win = np.arange(len(win_accs)) + offset + args.window_size // 2
    t_gate = np.arange(len(gate_weights_arr)) + offset

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # 上图：窗口准确率 + 漂移点 + detector / route 标记
    axes[0].plot(t_win, win_accs, color="steelblue", linewidth=1.8,
                 label="Phase 4 A", alpha=0.9)
    for dp in dataset.drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)
    axes[0].axvline(-1, color="red", linestyle="--", alpha=0.4, linewidth=1,
                    label="True Drift")
    for de in detector_events:
        axes[0].axvline(de, color="purple", linestyle=":", alpha=0.5, linewidth=1)
    if detector_events:
        axes[0].axvline(-1, color="purple", linestyle=":", alpha=0.5, linewidth=1,
                        label="Detector")
    for rt, action, _ in model.route_events:
        color = {"create": "darkgreen", "switch": "orange", "reuse": "gray"}.get(
            action, "black"
        )
        axes[0].axvline(rt, color=color, linestyle="-", alpha=0.6, linewidth=1.2)
    axes[0].set_ylabel("Window Accuracy")
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title(
        f"Phase 4 A on '{args.dataset}' (seed={args.seed})\n"
        f"context={args.context_size}, max_adapters={args.max_adapters}, "
        f"detector_δ={args.detector_delta}",
        fontsize=12,
    )
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # 中图：门控权重 α/β/γ
    axes[1].plot(t_gate, gate_weights_arr[:, 0], color="steelblue",
                 linewidth=1.4, label="α (slow)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 1], color="darkorange",
                 linewidth=1.4, label="β (inter)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 2], color="seagreen",
                 linewidth=1.4, label="γ (fast)", alpha=0.9)
    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.3, linewidth=1)
    axes[1].set_ylabel("Gate Weight")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 下图：active adapter id 轨迹
    axes[2].step(t_gate, active_id_arr, where="post",
                 color="darkgreen", linewidth=1.6, label="active adapter id")
    for dp in dataset.drift_points:
        axes[2].axvline(dp, color="red", linestyle="--", alpha=0.3, linewidth=1)
    for de in detector_events:
        axes[2].axvline(de, color="purple", linestyle=":", alpha=0.4, linewidth=1)
    axes[2].set_ylabel("Active Adapter ID")
    axes[2].set_xlabel("Time Step")
    max_id = int(active_id_arr.max()) if active_id_arr.size else 0
    axes[2].set_yticks(list(range(max_id + 1)))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    # 默认 stem 带上动作 / 触发源 / detector 实现，避免不同分支互相覆盖 npz。
    # 显式 --out_tag 优先（multiseed 驱动依赖它）。
    if args.out_tag is not None:
        stem = args.out_tag
    else:
        stem = f"phase4_a_{args.dataset}"
        variant = []
        if args.action_on_alarm != "route_adapter":
            variant.append(args.action_on_alarm)
        if args.trigger_source != "detector":
            variant.append(f"{args.trigger_source}lag{args.oracle_lag}")
        if args.detector_impl != "own":
            variant.append(args.detector_impl)
        if args.detector_input != "indicator":
            variant.append(args.detector_input)
        if args.context_loader != "sliding":
            variant.append(
                f"fr{args.fixed_ratio}" if args.context_loader == "composite" else "dual"
            )
        if variant:
            stem += "_" + "_".join(variant)
    png_path = os.path.join(args.results_dir, f"{stem}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {png_path}")

    # ── 保存数值结果 ─────────────────────────────────────────────────────
    npz_path = os.path.join(args.results_dir, f"{stem}.npz")
    atomic_savez(
        npz_path,
        predictions=preds_arr,
        labels=labels_arr,
        drift_points=np.array(dataset.drift_points),
        window_accs=win_accs,
        overall_acc=np.array([results["overall_acc"]]),
        balanced_acc=np.array([results["balanced_acc"]]),
        auc_roc=np.array([results["auc_roc"] if results["auc_roc"] is not None else np.nan]),
        pre_drift_acc=np.array([results["pre_drift_acc"] or np.nan]),
        post_drift_acc=np.array([results["post_drift_acc"] or np.nan]),
        avg_adaptation_speed=np.array(
            [results["avg_adaptation_speed"] if results["avg_adaptation_speed"] is not None else np.nan]
        ),
        gate_weights_trajectory=gate_weights_arr,
        active_id_trajectory=active_id_arr,
        consolidation_events=np.array(consolidation_events),
        detector_events=np.array(detector_events),
        route_t=route_t,
        route_action=route_action,
        route_active_id=route_active_id,
        n_adapters_final=np.array([model.adapter_library.n_adapters()]),
        seed=np.array([args.seed]),
        abs_error_history=np.array(model.abs_error_history, dtype=np.float32),
        indicator_history=np.array(model.indicator_history, dtype=np.int8),
        n_warmstart_inits=np.array([model.adapter_library.n_warmstart_inits]),
        n_random_inits=np.array([model.adapter_library.n_random_inits]),
        library_fit_threshold=np.array([args.library_fit_threshold]),
        library_init_strategy=np.array([args.library_init_strategy]),
        detector_impl=np.array([args.detector_impl]),
        detector_clock=np.array([args.detector_clock]),
        # ── Phase 5.5 诊断字段 ──
        action_on_alarm=np.array([args.action_on_alarm]),
        trigger_source=np.array([args.trigger_source]),
        oracle_lag=np.array([args.oracle_lag]),
        oracle_trigger_times=np.array(oracle_times if oracle_times else [], dtype=np.int64),
        alarm_events=np.array(model.alarm_events, dtype=np.int64),
        action_t=np.array([a[0] for a in model.action_events], dtype=np.int64),
        n_context_truncations=np.array([model.n_context_truncations]),
        consolidate_on_post_alarm_data=np.array([int(args.consolidate_on_post_alarm_data)]),
        detector_input=np.array([args.detector_input]),
        device=np.array([args.device]),
        n_estimators=np.array([args.n_estimators]),
        context_size=np.array([args.context_size]),
        detector_signal_history=np.array(model.detector_signal_history, dtype=np.float32),
        context_loader=np.array([args.context_loader]),
        fixed_ratio=np.array([args.fixed_ratio]),
        short_ratio=np.array([args.short_ratio]),
        long_max_age=np.array([args.long_max_age]),
    )
    print(f"数值结果已保存至: {npz_path}")


if __name__ == "__main__":
    main()
