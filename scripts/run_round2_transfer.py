"""Round 2 阶段迁移矩阵：用阶段 i 的样本当上下文，预测阶段 j 的留出集。

    python scripts/run_round2_transfer.py --seeds 0-9 --n_estimators 4

上下文 = 从阶段 i 的流内行里按类别均衡随机抽 ctx_rows 行；测试 = 阶段 j 的留出集（与主实验同一套，按 seed）。
对角线 = 阶段内准确率（参照值 A*），非对角线 = 跨阶段迁移（遗忘有多严重、哪些阶段相似）。
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_round2 import commit_id, prepare  # noqa: E402
from src.eval.round2_metrics import balanced_accuracy  # noqa: E402
from src.models.slow_prior import SlowPrior  # noqa: E402
from src.utils.atomic_io import atomic_savez  # noqa: E402
from src.utils.device import describe, resolve_device  # noqa: E402
from src.utils.seeding import set_global_seed  # noqa: E402
from scripts.run_round2_multiseed import parse_seeds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--ctx_rows", type=int, default=198, help="每类 ctx_rows // 6 行")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out_dir", default="results/round2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = resolve_device(args.device)
    print(f"[device] {describe(device)}", flush=True)

    seeds = parse_seeds(args.seeds)
    M = None
    t0 = time.time()
    for k, seed in enumerate(seeds):
        set_global_seed(seed)
        d = prepare(seed)
        K, R = d["n_classes"], int(d["regime"].max()) + 1
        if M is None:
            M = np.full((len(seeds), R, R), np.nan)
        rng = np.random.default_rng(20_000 + seed)
        prior = SlowPrior(device=device, n_estimators=args.n_estimators, random_state=seed)
        per = args.ctx_rows // K
        for i in range(R):
            rows_i = np.flatnonzero(d["regime"] == i)
            ctx = np.concatenate([rng.choice(rows_i[d["y"][rows_i] == c], size=per, replace=False)
                                  for c in range(K)])
            proba = prior.predict_proba_global(d["X"][ctx], d["y"][ctx], d["X_hold"], K)
            p = proba.argmax(1)
            for j in range(R):
                sel = d["regime_hold"] == j
                M[k, i, j] = balanced_accuracy(d["y_hold"][sel], p[sel])
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    m = np.nanmean(M, 0)
    print("balanced acc (rows = context regime, cols = holdout regime):")
    for i in range(m.shape[0]):
        print(f"  ctx R{i} " + " ".join(f"{v * 100:5.1f}" for v in m[i]))
    os.makedirs(args.out_dir, exist_ok=True)
    path = atomic_savez(os.path.join(args.out_dir, f"r2_transfer_n{args.n_estimators}{args.tag}.npz"),
                        matrix=M, seeds=np.array(seeds), n_estimators=np.array([args.n_estimators]),
                        ctx_rows=np.array([per * 6]), device=np.array([device]),
                        commit=np.array([commit_id()]))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
