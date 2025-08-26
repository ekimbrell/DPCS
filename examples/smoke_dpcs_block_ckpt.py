import argparse, math, sys, traceback
import torch, torch.nn as nn
from dpcs import DPCS
import os, sys, json, subprocess


# -----------------------------
# Model: blocky MLP (activations-heavy when B grows)
# -----------------------------
def make_block(in_out=2048, hidden=4096):
    return nn.Sequential(
        nn.Linear(in_out, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, in_out)
    )

class DeepMLP(nn.Module):
    def __init__(self, depth=6, in_out=2048, hidden=4096):
        super().__init__()
        self.blocks = nn.Sequential(*[make_block(in_out, hidden) for _ in range(depth)])
    def forward(self, x): return self.blocks(x)

# -----------------------------
# One full training step (forward+backward+step)
# Measures full-step peak via reset_peak_memory_stats + max_memory_allocated
# -----------------------------
def one_full_step(B, D, use_ckpt, in_out, hidden, depth, emit_snapshot=False, snapshot_path=None):
    """
    Spawn a clean subprocess that does exactly one full step and returns (ok, peak_mb).
    Launch by file path from the repo root and inject PYTHONPATH for src-layout.
    """
    import os, sys, json, subprocess

    # For this MLP, feature dim must match in_out (safety)
    if D != in_out:
        D = in_out

    # Resolve repo root and probe script path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../DPCS
    probe_path = os.path.join(repo_root, "examples", "one_step_probe.py")

    # Build command to run the probe *by path* (not -m) so cwd can be repo root
    cmd = [
        sys.executable, probe_path,
        "--batch", str(B),
        "--in-out", str(in_out),
        "--hidden", str(hidden),
        "--depth",  str(depth),
    ]
    if use_ckpt:
        cmd.append("--use-ckpt")
    if emit_snapshot and snapshot_path:
        cmd += ["--emit-snapshot", "--snapshot", snapshot_path]

    # Environment for child: robust allocator + src-layout PYTHONPATH
    env = os.environ.copy()
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:128"
    )
    # Ensure child can import 'dpcs' from src-layout even if not pip-installed
    src_dir  = os.path.join(repo_root, "src")
    extra_pp = os.pathsep.join([src_dir, repo_root])
    env["PYTHONPATH"] = extra_pp + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env and env["PYTHONPATH"] else "")

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            cwd=repo_root,  # run from repo root so relative imports/paths work
        )
        out = proc.stdout.strip()
        # If stdout is empty, treat as failure (don’t try to parse stderr noise)
        data = json.loads(out) if out else {"ok": False, "peak_mb": -1}
        ok = bool(data.get("ok", False))
        peak_mb = int(data.get("peak_mb", -1))
        return ok, peak_mb
    except Exception:
        return False, -1





# -----------------------------
# Exponential search + binary search to find max batch that fits
# -----------------------------
def max_batch_that_fits(D, use_ckpt, in_out, hidden, depth, start=1, max_limit=1<<20):
    dev = "cuda"
    # grow until first OOM
    b = start
    ok, _ = one_full_step(b, D, use_ckpt, in_out, hidden, depth)
    if not ok:
        return 0  # even batch=1 doesn't fit
    last_good = b
    while True:
        if b >= max_limit:
            last_good = b
            break
        b *= 2
        ok, _ = one_full_step(b, D, use_ckpt, in_out, hidden, depth)
        if ok:
            last_good = b
        else:
            first_bad = b
            break
    # if we never failed within limit, return last_good
    if 'first_bad' not in locals():
        return last_good
    # binary search [last_good+1, first_bad-1]
    L, R = last_good, first_bad
    while R - L > 1:
        mid = (L + R) // 2
        ok, _ = one_full_step(mid, D, use_ckpt, in_out, hidden, depth)
        if ok: L = mid
        else:  R = mid
    return L
    
def bench_throughput_same_batch(B, in_out, hidden, depth, use_ckpt, warmup=5, steps=30):
    """
    Spawn the probe in timing mode and return (ok, ms_per_step, samples_per_sec, last_loss).
    """
    import os, sys, json, subprocess
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe_path = os.path.join(repo_root, "examples", "one_step_probe.py")

    cmd = [
        sys.executable, probe_path,
        "--batch", str(B),
        "--in-out", str(in_out),
        "--hidden", str(hidden),
        "--depth",  str(depth),
        "--bench-steps", str(steps),
        "--warmup", str(warmup),
    ]
    if use_ckpt:
        cmd.append("--use-ckpt")

    env = os.environ.copy()
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:128"
    )
    src_dir  = os.path.join(repo_root, "src")
    env["PYTHONPATH"] = src_dir + (os.pathsep + env.get("PYTHONPATH","") if env.get("PYTHONPATH") else "")

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False, cwd=repo_root)
    out = proc.stdout.strip()
    if not out:
        return False, -1.0, -1.0, float("nan")
    data = json.loads(out)
    return bool(data.get("ok", False)), float(data.get("ms_per_step", -1.0)), float(data.get("samples_per_sec", -1.0)), float(data.get("last_loss", float("nan")))

def main():
    p = argparse.ArgumentParser(description="DPCS ckpt smoke: auto-tune batch near allocator cap; measure full-step peak.")
    p.add_argument("--in-out", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--depth",  type=int, default=6)
    p.add_argument("--dim",    type=int, default=None, help="input feature dim D (defaults to --in-out for this MLP)")
    p.add_argument("--alloc-frac", type=float, default=0.90, help="per-process allocator cap fraction of visible VRAM")
    p.add_argument("--emit-snapshot", action="store_true", help="dump CUDA memory snapshots for baseline/ckpt")
    p.add_argument("--snapshot-prefix", type=str, default="snap", help="snapshot filename prefix (adds _base/_ckpt).pkl")
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    dev = "cuda"

    # cap allocator so we can search near a meaningful ceiling
    try:
        torch.cuda.memory.set_per_process_memory_fraction(args.alloc_frac)  # cap process allocator
    except Exception:
        pass

    # report device + cap
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"Device: {dev}, CUDA available: {torch.cuda.is_available()}")
    print(f"GPU Free/Total: {free_b//(1024*1024)}MB / {total_b//(1024*1024)}MB")
    print(f"Allocator cap fraction set to: {args.alloc_frac:.2f} (best effort)")

    D = args.in_out if args.dim is None else args.dim
    if D != args.in_out:
        print(f"[warn] MLP expects input dim == in_out ({args.in_out}); overriding D {D} -> {args.in_out}")
        D = args.in_out


    # 1) Find baseline max batch (no checkpoint)
    base_B = max_batch_that_fits(D=D, use_ckpt=False, in_out=args.in_out, hidden=args.hidden, depth=args.depth)
    if base_B == 0:
        print("Baseline: even batch=1 did not fit under current config.")
        sys.exit(1)
    print(f"Baseline max batch: {base_B}")

    # 2) Measure baseline full-step peak at that batch
    ok, base_peak = one_full_step(
        B=base_B, D=D, use_ckpt=False, in_out=args.in_out, hidden=args.hidden, depth=args.depth,
        emit_snapshot=args.emit_snapshot, snapshot_path=(f"{args.snapshot_prefix}_base.pkl" if args.emit_snapshot else None)
    )
    print(f"Baseline full-step peak: {base_peak} MB at batch={base_B}")

    # 3) Measure DPCS ckpt full-step peak at the SAME batch (apples-to-apples)
    ok, dpcs_peak_sameB = one_full_step(
        B=base_B, D=D, use_ckpt=True, in_out=args.in_out, hidden=args.hidden, depth=args.depth,
        emit_snapshot=args.emit_snapshot, snapshot_path=(f"{args.snapshot_prefix}_ckpt_sameB.pkl" if args.emit_snapshot else None)
    )
    print(f"DPCS ckpt full-step peak at same batch: {dpcs_peak_sameB} MB (batch={base_B})")

    # 4) Find DPCS max batch (should be >= baseline if activations dominate)
    dpcs_B = max_batch_that_fits(D=D, use_ckpt=True, in_out=args.in_out, hidden=args.hidden, depth=args.depth)
    print(f"DPCS ckpt max batch: {dpcs_B}")

    # 5) If DPCS allows larger batch, show its peak at that larger batch too
    if dpcs_B > base_B:
        ok, dpcs_peak_maxB = one_full_step(
            B=dpcs_B, D=D, use_ckpt=True, in_out=args.in_out, hidden=args.hidden, depth=args.depth,
            emit_snapshot=args.emit_snapshot, snapshot_path=(f"{args.snapshot_prefix}_ckpt_maxB.pkl" if args.emit_snapshot else None)
        )
        print(f"DPCS ckpt full-step peak: {dpcs_peak_maxB} MB at batch={dpcs_B}")
        ok_b, b_ms, b_sps, b_loss = bench_throughput_same_batch(base_B, args.in_out, args.hidden, args.depth, use_ckpt=False, warmup=5, steps=30)
        ok_d, d_ms, d_sps, d_loss = bench_throughput_same_batch(base_B, args.in_out, args.hidden, args.depth, use_ckpt=True,  warmup=5, steps=30)

        print("\n=== Throughput at same batch (avg over steps) ===")
        if ok_b:
            print(f"Baseline AMP:   {b_ms:8.2f} ms/step, {b_sps:8.1f} samples/s, last loss {b_loss:.4f}")
        else:
            print("Baseline AMP:   (throughput measure failed)")

        if ok_d:
            print(f"DPCS (ckpt):    {d_ms:8.2f} ms/step, {d_sps:8.1f} samples/s, last loss {d_loss:.4f}")
        else:
            print("DPCS (ckpt):    (throughput measure failed)")
    else:
        print("DPCS did not enable a larger batch under current settings (likely parameter-dominated regime).")

if __name__ == "__main__":
    main()
