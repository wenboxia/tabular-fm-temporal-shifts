"""
Phase 4 Day 0.5 - 实验 0b: Multi-seed 重跑 Phase 1 / Phase 2 / Phase 3

3 配置 × 3 数据集 × 5 seeds = 45 runs

用法：
    python scripts/run_multiseed.py --dry_run         # 仅打印命令
    python scripts/run_multiseed.py                   # 串行执行
    python scripts/run_multiseed.py --n_parallel 3    # 同数据集内并行 3 个 seed

约束：
- 三数据集**串行**（每完成一个 dataset 再开下一个），避免跨 dataset OOM 互扰
- 同一数据集内可并行 N 个 seed（默认 1）
- 结果落盘：results/multiseed_{config}_{dataset}_seed{S}.npz
- 每个 run 完成后写 results/multiseed_summary.partial.md（增量 dump）
- 已存在的 npz 自动 skip（resumable）
"""

import argparse
import json
import os
import subprocess
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# 项目根（脚本在 scripts/ 下）
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["regime_switching", "rotating_boundary", "combined_drift"]
REAL_DATASETS = ["electricity", "insects"]
SEGMENTS = ["start", "middle", "end"]
SEGMENTS_ALIGNED = ["early", "mid", "late_pre", "late_post"]
# Phase 5.5：官方变点居中的 5 段（d0_control 无漂移，用于测误报率）
SEGMENTS_ALIGNED_V2 = ["d1_14352", "d2_19500", "d3_33240", "d4_double", "d0_control"]
CONFIGS = ["phase1", "phase2", "phase3", "phase4a"]

def _script_accepts(script: str, flag: str) -> bool:
    """底层脚本是否声明了该 flag（直接扫源码里的字面量）。

    没有这道过滤，--extra_args 里的 --detector_impl 会被原样塞给 run_baselines.py，
    整批 phase1 run 会以 argparse error 全部失败。
    """
    try:
        src = (ROOT / script).read_text()
    except OSError:
        return False
    return f'"{flag}"' in src or f"'{flag}'" in src


def _filter_extra_args(script: str, extra: "list[str] | None") -> list[str]:
    """丢掉目标脚本不认识的 flag 及其值，返回它认识的那部分。

    ⚠️ 被丢弃的 flag 一律**大声打印**。静默丢弃会让"双记忆"运行实际跑成普通滑窗基线、
    却贴着双记忆的标签 —— 这种错误只会产出看似正常的错误结果。
    是否"整批都没有脚本认识某个 flag"由 `_assert_extra_args_used` 在启动前硬检查。
    """
    if not extra:
        return []
    out, keep, dropped = [], False, []
    for tok in extra:
        if tok.startswith("--"):
            keep = _script_accepts(script, tok)
            if keep:
                out.append(tok)
            else:
                dropped.append(tok)
        elif keep:
            out.append(tok)
    if dropped:
        print(f"  [warn] {script} 不认识 {dropped}，这些参数不会传给它", flush=True)
    return out


def _assert_extra_args_used(configs: "list[str]", extra: "list[str] | None") -> None:
    """启动任何运行之前：extra_args 里的每个 flag 必须至少被一个目标脚本认识，否则直接报错。"""
    if not extra:
        return
    scripts = {c: build_base_cmd(c, "insects")[1] for c in configs}
    orphans = [
        tok for tok in extra
        if tok.startswith("--") and not any(_script_accepts(s, tok) for s in scripts.values())
    ]
    if orphans:
        raise SystemExit(
            f"[error] --extra_args 里的 {orphans} 不被本批任何脚本 {sorted(set(scripts.values()))} 认识。"
            " 继续跑会产出贴错标签的结果，已停止。"
        )


# 各 (config, dataset) 的 base 命令（不含 --seed / --out_tag / segment）
def build_base_cmd(
    config: str, dataset: str, dataset_source: str = "synthetic",
    segment_id: str = "start", segment_size: int = 5000,
    insects_aligned: bool = False, aligned_v2: bool = False,
    label_scheme: str = "pair_parity", variant_tag: str = "",
) -> list[str]:
    """返回该 (config, dataset) 的命令模板，调用方再加 --seed / --out_tag / 真实数据 flags。"""
    if config == "phase1":
        script = "scripts/run_baselines.py"
    elif config == "phase2":
        script = "scripts/run_phase2.py"
    elif config == "phase3":
        script = "scripts/run_phase3.py"
    elif config == "phase4a":
        script = "scripts/run_phase4_a.py"
    else:
        raise ValueError(config)

    cmd = [sys.executable, script, "--dataset", dataset]

    if dataset_source == "real":
        if config == "phase2":
            raise NotImplementedError("phase2 not plumbed for real data (Phase 5 skips it)")
        cmd += [
            "--dataset_source", "real",
            "--segment_id", segment_id,
            "--segment_size", str(segment_size),
        ]
        if dataset == "insects":
            if insects_aligned:
                cmd += ["--insects_aligned"]
            if aligned_v2:
                cmd += ["--aligned_v2"]
            if label_scheme != "pair_parity":
                cmd += ["--label_scheme", label_scheme]
    else:
        if dataset == "rotating_boundary":
            if config != "phase1":
                cmd += ["--n_features", "2"]
        elif dataset == "combined_drift":
            cmd += ["--n_samples", "5000"]
        elif dataset == "regime_switching":
            if config == "phase1":
                cmd += ["--n_samples", "3000"]

    # context_size 统一为 200（合成实验既定，real 数据沿用同值）
    cmd += ["--context_size", "200"]
    return cmd


def out_tag(
    config: str, dataset: str, seed: int,
    dataset_source: str = "synthetic", segment_id: str = "start",
    variant_tag: str = "",
) -> str:
    """variant_tag 把 label_scheme / detector / action 等变体写进文件名。

    ⚠️ 没有它，skip-existing 会把不同变体当成同一个 run 直接跳过
    （例如 fr=0.67 的 run 因为 fr=0 的 npz 已存在而被静默跳过）。
    """
    vt = f"_{variant_tag}" if variant_tag else ""
    if dataset_source == "real":
        # Phase 5: real data tag 含 segment_id；aligned 段名 (early/mid/late_pre/late_post)
        # 与 A+ 段名 (start/middle/end) 不冲突，无需额外标记
        return f"multiseed_{config}_real_{dataset}{vt}_{segment_id}_seed{seed}"
    # 合成 phase4a 第五轮 (Day 2 fit05random)：indicator + fit=0.5 + random init
    if config == "phase4a" and not variant_tag:
        return f"multiseed_phase4a_fit05random_{dataset}_seed{seed}"
    return f"multiseed_{config}{vt}_{dataset}_seed{seed}"


def npz_path(
    config: str, dataset: str, seed: int,
    dataset_source: str = "synthetic", segment_id: str = "start",
    variant_tag: str = "",
) -> Path:
    return RESULTS_DIR / (
        out_tag(config, dataset, seed, dataset_source, segment_id, variant_tag) + ".npz"
    )


def build_full_cmd(
    config: str, dataset: str, seed: int,
    dataset_source: str = "synthetic", segment_id: str = "start",
    segment_size: int = 5000, insects_aligned: bool = False,
    aligned_v2: bool = False, label_scheme: str = "pair_parity",
    variant_tag: str = "", extra_args: "list[str] | None" = None,
) -> list[str]:
    cmd = build_base_cmd(
        config, dataset, dataset_source, segment_id, segment_size, insects_aligned,
        aligned_v2, label_scheme, variant_tag,
    )
    cmd += [
        "--seed", str(seed),
        "--out_tag", out_tag(config, dataset, seed, dataset_source, segment_id, variant_tag),
    ]
    # extra_args 只透传给认识它的脚本（phase1 = run_baselines.py 没有 detector/action 相关 flag）
    script = cmd[1]
    cmd += _filter_extra_args(script, extra_args)
    # phase4a：合成 Day 2 / 真实 Phase 5 都用 indicator + random init（Phase 4 final）
    if config == "phase4a":
        cmd += ["--library_init_strategy", "random"]
    return cmd


def read_overall_acc(
    config: str, dataset: str, seed: int,
    dataset_source: str = "synthetic", segment_id: str = "start",
    variant_tag: str = "",
) -> float | None:
    """从 npz 中读取 overall_acc。Phase 2 取 KNN 列。"""
    path = npz_path(config, dataset, seed, dataset_source, segment_id, variant_tag)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if config == "phase2":
        # 多 corrector npz：取 KNN 那列
        for key in ["overall_acc_TabPFN_p_KNN", "overall_acc_TabPFN_p_KNN_"]:
            if key in data.files:
                return float(data[key][0])
        return None
    if "overall_acc" in data.files:
        return float(data["overall_acc"][0])
    return None


def read_phase4a_metrics(dataset: str, seed: int) -> dict | None:
    """读取 phase4a npz 中的关键诊断字段（用于 partial md）。"""
    path = npz_path("phase4a", dataset, seed)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    out = {
        "overall_acc": float(data["overall_acc"][0]) if "overall_acc" in data.files else None,
        "post_drift_acc": (
            float(data["post_drift_acc"][0])
            if "post_drift_acc" in data.files and not np.isnan(data["post_drift_acc"][0])
            else None
        ),
        "n_routes": int(data["route_t"].shape[0]) if "route_t" in data.files else 0,
        "n_adapters": int(data["n_adapters_final"][0]) if "n_adapters_final" in data.files else 1,
        "n_detector_events": int(data["detector_events"].shape[0]) if "detector_events" in data.files else 0,
    }
    return out


def append_phase4a_partial_row(rec: dict) -> None:
    """每个 phase4a 任务跑完，把这一行 append 到 partial md。"""
    out_path = RESULTS_DIR / "multiseed_phase4a_fit05random.partial.md"
    header = "| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    init = not out_path.exists()
    metrics = read_phase4a_metrics(rec["dataset"], rec["seed"]) or {}
    with open(out_path, "a") as fh:
        if init:
            fh.write("# Phase 4 A multi-seed partial summary (live)\n\n")
            fh.write("Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).\n\n")
            fh.write(header)
            fh.write(sep)
        oa = metrics.get("overall_acc")
        pd = metrics.get("post_drift_acc")
        oa_s = f"{oa:.4f}" if oa is not None else "—"
        pd_s = f"{pd:.4f}" if pd is not None else "—"
        fh.write(
            f"| {rec['seed']} | {rec['dataset']} | {oa_s} | {pd_s} | "
            f"{metrics.get('n_routes', 0)} | {metrics.get('n_adapters', 1)} | "
            f"{rec['elapsed_sec']:.0f}s |\n"
        )


def run_one(task: tuple, log_dir: Path, insects_aligned: bool = False,
            aligned_v2: bool = False, label_scheme: str = "pair_parity",
            variant_tag: str = "", extra_args: "list[str] | None" = None) -> dict:
    """跑一个 task；synthetic = (config, dataset, seed)，real = (config, dataset, seed, segment_id)。"""
    if len(task) == 4:
        config, dataset, seed, segment_id = task
        dataset_source = "real"
    else:
        config, dataset, seed = task
        dataset_source = "synthetic"
        segment_id = "start"
    log_path = log_dir / (
        out_tag(config, dataset, seed, dataset_source, segment_id, variant_tag) + ".log"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_full_cmd(
        config, dataset, seed, dataset_source, segment_id,
        insects_aligned=insects_aligned, aligned_v2=aligned_v2,
        label_scheme=label_scheme, variant_tag=variant_tag, extra_args=extra_args,
    )
    t0 = time.time()
    try:
        with open(log_path, "w") as fh:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"exc:{type(e).__name__}"

    acc = read_overall_acc(config, dataset, seed, dataset_source, segment_id, variant_tag)
    return {
        "config": config, "dataset": dataset, "seed": seed,
        "dataset_source": dataset_source, "segment_id": segment_id,
        "variant_tag": variant_tag, "label_scheme": label_scheme,
        "status": status, "elapsed_sec": elapsed, "overall_acc": acc,
        "cmd": " ".join(cmd),
        "log": str(log_path.relative_to(ROOT)),
    }


def write_partial_summary(dataset_source: str = "synthetic", partial_tag: str = "",
                          variant_tag: str = "", seg_pool_override: "list[str] | None" = None):
    """根据 results/ 下现存 multiseed_*.npz 写 partial summary 表格。

    partial_tag: 文件名后缀（例如 "_electricity"），便于 Stage A/B 分开追踪。
    """
    rows = []
    if dataset_source == "real":
        ds_pool = REAL_DATASETS
        seg_pool = seg_pool_override or SEGMENTS
        out_name = f"multiseed_phase5{partial_tag}.partial.md"
    else:
        ds_pool, seg_pool = DATASETS, ["start"]  # synthetic 占位
        out_name = "multiseed_summary.partial.md"

    for config in CONFIGS:
        for dataset in ds_pool:
            for seg in seg_pool:
                accs = []
                for s in SEEDS:
                    a = read_overall_acc(config, dataset, s, dataset_source, seg, variant_tag)
                    if a is not None:
                        accs.append(a)
                if accs:
                    arr = np.array(accs)
                    rows.append({
                        "config": config, "dataset": dataset, "segment": seg,
                        "n": len(accs), "mean": arr.mean(),
                        "std": arr.std(ddof=1) if len(accs) > 1 else 0.0,
                    })

    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as fh:
        fh.write(f"# Multi-seed partial summary ({dataset_source}) (live)\n\n")
        fh.write(f"_Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        if dataset_source == "real":
            fh.write("| Config | Dataset | Segment | n_seeds | overall_acc mean ± std |\n")
            fh.write("|---|---|---|---|---|\n")
            for r in rows:
                fh.write(f"| {r['config']} | {r['dataset']} | {r['segment']} | {r['n']}/5 | "
                         f"{r['mean']:.4f} ± {r['std']:.4f} |\n")
        else:
            fh.write("| Config | Dataset | n_seeds | overall_acc mean ± std |\n")
            fh.write("|---|---|---|---|\n")
            for r in rows:
                fh.write(f"| {r['config']} | {r['dataset']} | {r['n']}/5 | "
                         f"{r['mean']:.4f} ± {r['std']:.4f} |\n")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed driver (Phase 4 0b + Phase 5 real)")
    p.add_argument("--dry_run", action="store_true",
                   help="仅打印命令，不执行")
    p.add_argument("--n_parallel", type=int, default=1,
                   help="同一数据集内并行任务数（默认 1=串行）")
    p.add_argument("--configs", type=str, default=",".join(CONFIGS),
                   help=f"逗号分隔 config 子集，可选 {CONFIGS}")
    p.add_argument("--datasets", type=str, default=",".join(DATASETS),
                   help=f"逗号分隔 dataset 子集，合成 {DATASETS}，真实 {REAL_DATASETS}")
    p.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                   help="逗号分隔 seed 列表")
    p.add_argument("--dataset_source", type=str, default="synthetic",
                   choices=["synthetic", "real"],
                   help="合成（默认）或真实（Phase 5）")
    p.add_argument("--segments", type=str, default=",".join(SEGMENTS),
                   help=f"real 时使用，逗号分隔 segment 子集，可选 {SEGMENTS}")
    p.add_argument("--segment_size", type=int, default=5000,
                   help="real 时 segment 大小（A+ 协议默认 5000）")
    p.add_argument("--partial_tag", type=str, default="",
                   help="partial.md 文件名后缀，例如 '_electricity'（Stage A/B 分开追踪）")
    p.add_argument("--aligned_v2", action="store_true",
                   help="Insects 用 Phase 5.5 官方变点居中的 5 段；"
                        f"会把 --segments 默认改为 {SEGMENTS_ALIGNED_V2}")
    p.add_argument("--label_scheme", type=str, default="pair_parity",
                   choices=["pair_parity", "pair_A_vs_B"],
                   help="Insects 标签方案（非默认值会自动进 variant_tag，避免 skip-existing 误跳）")
    p.add_argument("--variant_tag", type=str, default="",
                   help="变体标签，写进 out_tag/npz/log 文件名。留空时按 label_scheme 等自动推断")
    p.add_argument("--extra_args", type=str, default="",
                   help="透传给底层脚本的额外参数，空格分隔，"
                        "例如 '--detector_impl river --action_on_alarm context_reset'")
    p.add_argument("--insects_aligned", action="store_true",
                   help="Insects 用 4 个 drift-aligned segments (Phase 5 B1+)；"
                        "会自动把 --segments 默认改为 early,mid,late_pre,late_post")
    return p.parse_args()


def main():
    args = parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.insects_aligned and args.aligned_v2:
        raise SystemExit("[error] --insects_aligned 与 --aligned_v2 互斥")

    # --aligned_v2 时若用户没显式给 --segments，就用 v2 的 5 段
    default_segments = ",".join(SEGMENTS)
    if args.aligned_v2 and args.segments == default_segments:
        args.segments = ",".join(SEGMENTS_ALIGNED_V2)
    elif args.insects_aligned and args.segments == default_segments:
        args.segments = ",".join(SEGMENTS_ALIGNED)
    segments = (
        [s.strip() for s in args.segments.split(",") if s.strip()]
        if args.dataset_source == "real" else ["start"]  # synthetic 占位
    )

    # variant_tag：没显式给就从会改变结果的开关里推断。
    # 没有它，skip-existing 会把不同变体当成已跑过的同一个 run 直接跳过。
    variant_tag = args.variant_tag
    if not variant_tag:
        if args.extra_args:
            raise SystemExit(
                "[error] 用了 --extra_args 就必须显式给 --variant_tag。\n"
                "  extra_args 会改变结果，而 out_tag 不含它 → skip-existing 会把这批 run\n"
                "  当成已跑过的旧变体直接跳过，或反过来覆盖旧结果。\n"
                f"  例如：--variant_tag v2AvsB_river_ctxreset"
            )
        parts = []
        if args.aligned_v2:
            parts.append("v2")
        if args.label_scheme != "pair_parity":
            parts.append(args.label_scheme)
        variant_tag = "_".join(parts)
    extra_args = args.extra_args.split() if args.extra_args else None
    _assert_extra_args_used(configs, extra_args)
    if variant_tag:
        print(f"variant_tag: {variant_tag}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_DIR / "multiseed"

    # 计划：外层 dataset 串行，每个 dataset 内 (config × segments × seeds) 池
    n_per_dataset = len(configs) * len(seeds) * (len(segments) if args.dataset_source == "real" else 1)
    total = n_per_dataset * len(datasets)
    print(f"=== Multi-seed driver ({args.dataset_source}) ===")
    print(f"configs={configs}  datasets={datasets}  seeds={seeds}")
    if args.dataset_source == "real":
        print(f"segments={segments}  segment_size={args.segment_size}")
    print(f"total runs: {total}  parallelism within dataset: {args.n_parallel}")
    print(f"dry_run: {args.dry_run}")

    def _enumerate_tasks(dataset: str) -> list:
        tasks = []
        for config in configs:
            for seed in seeds:
                if args.dataset_source == "real":
                    for seg in segments:
                        if npz_path(config, dataset, seed, "real", seg, variant_tag).exists():
                            print(f"  [skip] {npz_path(config, dataset, seed, 'real', seg, variant_tag).name}")
                            continue
                        tasks.append((config, dataset, seed, seg))
                else:
                    if npz_path(config, dataset, seed, variant_tag=variant_tag).exists():
                        print(f"  [skip] {npz_path(config, dataset, seed, variant_tag=variant_tag).name}")
                        continue
                    tasks.append((config, dataset, seed))
        return tasks

    if args.dry_run:
        print("\n--- planned commands (dry-run) ---")
        n_skip = 0
        n_plan = 0
        for dataset in datasets:
            print(f"\n## dataset = {dataset}")
            tasks = _enumerate_tasks(dataset)
            for task in tasks:
                if args.dataset_source == "real":
                    cmd = build_full_cmd(
                        task[0], task[1], task[2], "real", task[3], args.segment_size,
                        insects_aligned=args.insects_aligned, aligned_v2=args.aligned_v2,
                        label_scheme=args.label_scheme, variant_tag=variant_tag,
                        extra_args=extra_args,
                    )
                else:
                    cmd = build_full_cmd(
                        task[0], task[1], task[2], variant_tag=variant_tag,
                        extra_args=extra_args,
                    )
                print("  " + " ".join(cmd))
                n_plan += 1
        print(f"\n--- summary: {n_plan} runs to execute ---")
        return

    overall_t0 = time.time()
    all_records = []

    for dataset in datasets:
        print(f"\n========== dataset: {dataset} ==========")
        tasks = _enumerate_tasks(dataset)

        if not tasks:
            continue

        n_par = max(1, args.n_parallel)
        if n_par == 1:
            # 串行
            for task in tasks:
                rec = run_one(task, log_dir, insects_aligned=args.insects_aligned,
                              aligned_v2=args.aligned_v2, label_scheme=args.label_scheme,
                              variant_tag=variant_tag, extra_args=extra_args)
                all_records.append(rec)
                print(f"  [{rec['status']}] {rec['config']} / {rec['dataset']} / seed{rec['seed']} "
                      f"acc={rec['overall_acc']} elapsed={rec['elapsed_sec']:.0f}s log={rec['log']}")
                write_partial_summary(args.dataset_source, args.partial_tag,
                                      variant_tag, segments)
        else:
            # 同 dataset 内并行
            # 所有平台统一用 spawn：Linux 默认 fork 会复制带线程的父进程（numpy/BLAS），
            # 且与在 Mac 上自测过的行为不一致。worker 只负责起子进程，spawn 的开销可以忽略。
            with ProcessPoolExecutor(max_workers=n_par,
                                     mp_context=multiprocessing.get_context("spawn")) as ex:
                futs = {
                    ex.submit(run_one, t, log_dir, args.insects_aligned, args.aligned_v2,
                              args.label_scheme, variant_tag, extra_args): t
                    for t in tasks
                }
                for fut in as_completed(futs):
                    rec = fut.result()
                    all_records.append(rec)
                    print(f"  [{rec['status']}] {rec['config']} / {rec['dataset']} / seed{rec['seed']} "
                          f"acc={rec['overall_acc']} elapsed={rec['elapsed_sec']:.0f}s log={rec['log']}")
                    write_partial_summary(args.dataset_source, args.partial_tag,
                                          variant_tag, segments)
                    if rec["config"] == "phase4a" and rec.get("dataset_source", "synthetic") == "synthetic":
                        append_phase4a_partial_row(rec)

    total_elapsed = time.time() - overall_t0
    print(f"\n=== ALL DONE in {total_elapsed/60:.1f} min ===")
    write_partial_summary(args.dataset_source, args.partial_tag, variant_tag, segments)

    # 落盘运行记录
    # 每个变体单独一个运行记录，否则同一批里后一次调用会覆盖前一次的记录
    rec_path = RESULTS_DIR / (f"multiseed_runlog_{variant_tag}.json" if variant_tag else "multiseed_runlog.json")
    with open(rec_path, "w") as fh:
        json.dump(all_records, fh, indent=2)
    print(f"runlog: {rec_path}")
    failed = [r for r in all_records if r["status"] != "ok"]
    if failed:
        print(f"!! {len(failed)} run(s) failed, see the logs above")
        sys.exit(1)


if __name__ == "__main__":
    main()
