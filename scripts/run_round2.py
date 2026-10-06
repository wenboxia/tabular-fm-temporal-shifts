"""Round 2：一种记忆方法 × 一个 seed，在完整 Insects 流上跑批量 prequential + 留出集回测。

    python scripts/run_round2.py --method arch_routed --seed 0 --n_estimators 1 --device cpu

设计与成功标准见 results/phase56_prereg.md（先提交、后跑）。
输出：<out_dir>/r2_<method>_n<E>_S<S>_seed<seed><tag>.npz（原子写入）。
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.real_world import load_insects_stream
from src.drift.error_detector import make_detector
from src.eval.stream_runner import needs_detector, run_stream
from src.memory.context_memory import METHODS, make_memory
from src.models.slow_prior import SlowPrior
from src.utils.atomic_io import atomic_savez
from src.utils.device import describe, resolve_device
from src.utils.forgetting import carve_regime_holdouts, remap_points
from src.utils.seeding import set_global_seed

HOLDOUT_PER_CLASS = {0: 50, 1: 50, 2: 50, 3: 50, 4: 20, 5: 50}
START, NORM_ROWS = 200, 200


def commit_id() -> str:
    """记录在结果文件里的代码版本：环境变量或仓库根目录的 COMMIT 文件（没有 git 时用），否则用 git。"""
    env = os.environ.get("NEURAL1_COMMIT", "").strip()
    if env:
        return env
    root = os.path.join(os.path.dirname(__file__), "..")
    f = os.path.join(root, "COMMIT")
    if os.path.exists(f):
        return open(f).read().strip()
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def prepare(seed: int):
    """加载流、按 seed 切留出集、归一化。返回所有方法共用的数据。"""
    s = load_insects_stream()
    stream_idx, hold_idx = carve_regime_holdouts(s.y, s.regime, HOLDOUT_PER_CLASS, seed)
    X = s.X[stream_idx].astype(np.float64)
    mu = X[:NORM_ROWS].mean(0)
    sd = X[:NORM_ROWS].std(0)
    sd[sd < 1e-6] = 1.0
    norm = lambda a: ((a - mu) / sd).astype(np.float32)   # noqa: E731
    return dict(
        X=norm(X), y=s.y[stream_idx], regime=s.regime[stream_idx],
        X_hold=norm(s.X[hold_idx].astype(np.float64)), y_hold=s.y[hold_idx],
        regime_hold=s.regime[hold_idx], stream_idx=stream_idx, hold_idx=hold_idx,
        boundaries=remap_points(s.change_points, stream_idx), n_classes=len(s.class_ids),
        class_ids=s.class_ids, change_points=s.change_points,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--max_stream_rows", type=int, default=None, help="只用于自测：截断数据流")
    ap.add_argument("--out_dir", default="results/round2")
    ap.add_argument("--tag", default="", help="文件名后缀（如 _smoke、_pilot）")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[device] {describe(device)}", flush=True)
    d = prepare(args.seed)
    memory = make_memory(args.method, d["X"], d["y"], d["n_classes"], args.seed)
    prior = SlowPrior(device=device, n_estimators=args.n_estimators, random_state=args.seed)
    predict_fn = lambda Xc, yc, Xq: prior.predict_proba_global(Xc, yc, Xq, d["n_classes"])  # noqa: E731
    adwin = needs_detector(args.method)
    detector = make_detector("river", delta=0.002, cooldown=300, clock=1) if adwin else None

    print(f"[run] method={args.method} seed={args.seed} n_estimators={args.n_estimators} "
          f"stride={args.stride} stream={len(d['y'])} holdout={len(d['y_hold'])} "
          f"boundaries={d['boundaries']}", flush=True)
    t0 = time.time()
    last = [0]

    def progress(t, n, res):          # 每 2000 行打一行进度，用来确认没有卡住
        if t - last[0] >= 2000 or t == n:
            last[0] = t
            el = time.time() - t0
            eta = el / max(t - START, 1) * (n - t)
            print(f"  行 {t:6d}/{n} | {el / 60:5.1f} min | 预计还需 {eta / 60:5.1f} min | fits {res.n_fits}",
                  flush=True)

    r = run_stream(memory, predict_fn, d["X"], d["y"], d["regime"], d["X_hold"], d["regime_hold"],
                   d["boundaries"], stride=args.stride, start=START, ckpt_every=args.ckpt_every,
                   segmentation="adwin" if adwin else "official", detector=detector,
                   max_rows=args.max_stream_rows, progress=progress)
    elapsed = time.time() - t0

    done = r.pred >= 0
    acc = float((r.pred[done] == d["y"][done]).mean())
    print(f"[done] {elapsed:.0f}s fits={r.n_fits} ({r.fit_seconds / max(r.n_fits, 1):.3f}s/fit) "
          f"acc={acc:.4f} max_ctx={r.max_ctx} closures={r.closures.tolist()} ckpts={len(r.ckpt_t)}",
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"r2_{args.method}_n{args.n_estimators}_S{args.stride}_seed{args.seed}{args.tag}"
    path = atomic_savez(
        os.path.join(args.out_dir, stem + ".npz"),
        method=np.array([args.method]), seed=np.array([args.seed]),
        n_estimators=np.array([args.n_estimators]), stride=np.array([args.stride]),
        ckpt_every=np.array([args.ckpt_every]), start=np.array([START]),
        device=np.array([device]), commit=np.array([commit_id()]),
        max_stream_rows=np.array([-1 if args.max_stream_rows is None else args.max_stream_rows]),
        pred=r.pred, route=r.route, y=d["y"], regime=d["regime"],
        y_hold=d["y_hold"], regime_hold=d["regime_hold"],
        stream_idx=d["stream_idx"], hold_idx=d["hold_idx"], boundaries=np.asarray(d["boundaries"]),
        ckpt_t=r.ckpt_t, ckpt_upto=r.ckpt_upto, ckpt_pred=r.ckpt_pred, closures=r.closures,
        n_fits=np.array([r.n_fits]), fit_seconds=np.array([r.fit_seconds]),
        elapsed=np.array([elapsed]), max_ctx=np.array([r.max_ctx]),
        budget=np.array([memory.max_context()]), leak_count=np.array([r.leak_count]),
        digests=np.array(r.digests), overall_acc=np.array([acc]),
    )
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
