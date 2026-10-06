"""Round 2 批量驱动：方法 × seed，结果已存在就跳过（可断点续跑），上下文大的先跑。

    python scripts/run_round2_multiseed.py --methods sw200,arch_routed --seeds 0-4 --n_parallel 2

每个运行在独立子进程里执行（spawn），日志写到 logs/round2/<stem>.log。
"""
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.memory.context_memory import METHODS  # noqa: E402

COST = {"sw2000": 9, "sw1500": 8, "dual1000": 7, "sw1000": 7, "arch_union": 6, "sw600": 5,
        "dual400": 4, "cbfifo400": 4, "sw400": 4, "arch_routed": 3, "arch_routed_adwin": 3,
        "sw300": 3, "sw200": 2, "sw100": 1, "dual1000_rb": 7, "dual1000_rb_adwin": 7}


def parse_seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def stem(method, seed, args):
    return f"r2_{method}_n{args.n_estimators}_S{args.stride}_seed{seed}{args.tag}"


def run_one(method, seed, args):
    s = stem(method, seed, args)
    log_dir = os.path.join(ROOT, "logs", "round2"); os.makedirs(log_dir, exist_ok=True)
    cmd = [sys.executable, "scripts/run_round2.py", "--method", method, "--seed", str(seed),
           "--n_estimators", str(args.n_estimators), "--stride", str(args.stride),
           "--device", args.device, "--out_dir", args.out_dir]
    if args.tag:
        cmd += ["--tag", args.tag]
    if args.max_stream_rows:
        cmd += ["--max_stream_rows", str(args.max_stream_rows)]
    t0 = time.time()
    with open(os.path.join(log_dir, s + ".log"), "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    return dict(method=method, seed=seed, rc=rc, elapsed=time.time() - t0, log=f"logs/round2/{s}.log")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n_parallel", type=int, default=2)
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", default="results/round2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max_stream_rows", type=int, default=None)
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        raise SystemExit(f"未知方法 {bad}")
    seeds = parse_seeds(args.seeds)
    todo = []
    for m in methods:
        for s in seeds:
            if os.path.exists(os.path.join(ROOT, args.out_dir, stem(m, s, args) + ".npz")):
                print(f"[skip] {stem(m, s, args)}", flush=True)
            else:
                todo.append((m, s))
    todo.sort(key=lambda ms: (ms[1], -COST.get(ms[0], 5)))   # 先按 seed，同一 seed 内上下文大的先跑
    print(f"=== round2: {len(todo)} runs to do, n_parallel={args.n_parallel} ===", flush=True)

    recs, t0 = [], time.time()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.n_parallel), mp_context=ctx) as ex:
        futs = [ex.submit(run_one, m, s, args) for m, s in todo]
        for f in as_completed(futs):
            r = f.result(); recs.append(r)
            print(f"  [{'ok' if r['rc'] == 0 else 'FAIL rc=%d' % r['rc']}] {r['method']} seed{r['seed']} "
                  f"{r['elapsed'] / 60:.1f} min  {r['log']}", flush=True)
    print(f"=== ALL DONE in {(time.time() - t0) / 3600:.2f} h ===", flush=True)
    os.makedirs(os.path.join(ROOT, args.out_dir), exist_ok=True)
    tagname = f"runlog_{'-'.join(methods) if len(methods) < 4 else 'multi'}_{args.seeds}{args.tag}.json"
    with open(os.path.join(ROOT, args.out_dir, tagname), "w") as fh:
        json.dump(recs, fh, indent=1)
    if any(r["rc"] != 0 for r in recs):
        sys.exit(1)


if __name__ == "__main__":
    main()
