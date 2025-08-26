# examples/auto_sweep_transformer.py
# Find the largest seq_len that fits for baseline vs checkpointed training using the transformer probe.
# Then time each mode at its own max and report tokens/s and p95 peak MB.

import argparse, json, os, subprocess, sys, math

PROBE = os.path.join(os.path.dirname(__file__), "one_step_probe_transformer.py")

def run_probe(args_list, env=None):
    """Run the probe and return parsed JSON (or None on failure)."""
    cmd = [sys.executable, PROBE] + args_list
    env2 = dict(os.environ)
    env2["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if env:
        env2.update(env)
    try:
        out = subprocess.check_output(cmd, env=env2, stderr=subprocess.DEVNULL, timeout=1200)
        return json.loads(out.decode("utf-8").strip())
    except Exception:
        return None

def fits_once(common_args, seq_len, env):
    args = common_args + ["--seq-len", str(seq_len), "--repeat", "1", "--warmup", "0"]
    res = run_probe(args, env=env)
    return bool(res and res.get("ok", False))

def find_max_seq(common_args, start, max_cap, env):
    """Exponential grow until fail, then binary search last good..fail interval."""
    if not fits_once(common_args, start, env):
        lo, hi = 1, start
        while hi > 1 and not fits_once(common_args, hi, env):
            hi //= 2
        return max(1, hi)
    lo, hi = start, start
    while hi < max_cap and fits_once(common_args, hi, env):
        lo = hi
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits_once(common_args, mid, env):
            lo = mid
        else:
            hi = mid
    return lo

def time_at_seq(common_args, seq_len, warmup, repeat, env):
    args = common_args + ["--seq-len", str(seq_len), "--timed",
                          "--warmup", str(warmup), "--repeat", str(repeat)]
    return run_probe(args, env=env)

def tokens_per_sec(batch, seq_len, mean_ms):
    return (batch * seq_len) / (mean_ms / 1000.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--ff-dim", type=int, default=2048)
    ap.add_argument("--start-seq", type=int, default=1024)
    ap.add_argument("--max-cap", type=int, default=16384)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--allocator", type=str, default="",
                    help="e.g. backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:128")
    args = ap.parse_args()

    base_common = [
        "--transformer",
        "--batch", str(args.batch),
        "--d-model", str(args.d_model),
        "--n-heads", str(args.n_heads),
        "--n-layers", str(args.n_layers),
        "--ff-dim", str(args.ff_dim),
        "--dropout", str(args.dropout),
    ]
    ckpt_common = base_common + ["--use-ckpt"]

    env = {}
    if args.allocator:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.allocator

    # 1) Find max seq for each mode
    base_max = find_max_seq(base_common, start=args.start_seq, max_cap=args.max_cap, env=env)
    ckpt_max = find_max_seq(ckpt_common, start=max(base_max, args.start_seq), max_cap=args.max_cap, env=env)

    # 2) Time at each mode's own max
    # If the size is huge, shorten repeat to keep runtime reasonable.
    base_rep = args.repeat if base_max <= args.max_cap else max(5, args.repeat // 5)
    ckpt_rep = args.repeat if ckpt_max <= args.max_cap else max(5, args.repeat // 5)

    base_stats = time_at_seq(base_common, base_max, warmup=args.warmup, repeat=base_rep, env=env) if base_max > 0 else None
    ckpt_stats = time_at_seq(ckpt_common, ckpt_max, warmup=args.warmup, repeat=ckpt_rep, env=env) if ckpt_max > 0 else None

    print("\n=== Auto-sweep summary ===")
    print(f"Config: B={args.batch}, d={args.d_model}, L={args.n_layers}, H={args.n_heads}, FF={args.ff_dim}")
    if args.allocator:
        print(f"Allocator: PYTORCH_CUDA_ALLOC_CONF={args.allocator}")

    print(f"\nBaseline maximum seq_len: {base_max}")
    if base_stats and base_stats.get('ok'):
        t = base_stats["timing_ms"]["mean"]
        peak = base_stats["peaks_mb"]["p95"]
        tps = tokens_per_sec(args.batch, base_max, t)
        print(f"  baseline @ S={base_max}: mean={t:.2f} ms, p95_peak={peak} MB, tokens/s={tps:,.1f}")

    print(f"\nCheckpoint maximum seq_len: {ckpt_max}")
    if ckpt_stats and ckpt_stats.get('ok'):
        t = ckpt_stats["timing_ms"]["mean"]
        peak = ckpt_stats["peaks_mb"]["p95"]
        tps = tokens_per_sec(args.batch, ckpt_max, t)
        print(f"  ckpt     @ S={ckpt_max}: mean={t:.2f} ms, p95_peak={peak} MB, tokens/s={tps:,.1f}")

    if base_max and ckpt_max:
        if ckpt_max > base_max:
            lift = (ckpt_max / base_max - 1.0) * 100.0
            print(f"\nResult: Checkpointing enables +{lift:.1f}% longer sequences at same B, ({base_max} \u2192 {ckpt_max}).")
        else:
            print("\nResult: No capacity gain for seq_len under this config.")
    print()
    sys.stdout.flush()

if __name__ == "__main__":
    main()
