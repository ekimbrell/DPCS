# examples/one_step_probe_transformer.py
# One or many full training steps on a tiny Transformer (or MLP).
# Prints JSON:
#  - single step: {"ok": bool, "peak_mb": int, "elapsed_ms": float, ...}
#  - repeated:    {
#        "ok": true,
#        "n_measured": N,
#        "timing_ms": {"mean": ..., "median": ..., "stdev": ..., "min": ..., "max": ...},
#        "peaks_mb":  {"p95": ..., "max": ..., "median": ...},
#        "last_loss": float
#     }
#
# Implements per-iteration peak measurement per PyTorch docs:
#   reset_peak_memory_stats() before each step, read max_memory_allocated() after sync.
# See: https://pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html

import argparse, json, sys, time, math, statistics
import torch, torch.nn as nn

def make_block(in_out: int, hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_out, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, in_out),
    )

class DeepMLP(nn.Module):
    def __init__(self, depth: int, in_out: int, hidden: int):
        super().__init__()
        self.blocks = nn.Sequential(*[make_block(in_out, hidden) for _ in range(depth)])
    def forward(self, x):  # x: (B, in_out)
        return self.blocks(x)

class TinyTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
    def forward(self, x):  # x: (B, S, D)
        return self.enc(x)

def _percentile_nearest_rank(values, p: float) -> float:
    """Nearest-rank percentile (p in [0,100]); works without NumPy."""
    if not values:
        return float("nan")
    vals = sorted(values)
    k = max(1, min(len(vals), int(math.ceil(p/100.0 * len(vals))))) - 1
    return float(vals[k])

def run_one_or_many_steps(
    use_ckpt: bool,
    emit_snapshot: bool,
    snapshot_path: str | None,
    timed: bool,
    verbose: bool,
    repeat: int,
    warmup: int,
    # MLP
    batch: int = 1, in_out: int = 2048, hidden: int = 4096, depth: int = 6,
    # Transformer
    use_transformer: bool = True, d_model: int = 768, n_heads: int = 6,
    n_layers: int = 12, ff_dim: int = 3072, seq_len: int = 1024,
    dropout: float = 0.0,
) -> dict:
    from dpcs import DPCS

    def vlog(msg: str):
        if verbose:
            print(f"[probe] {msg}", file=sys.stderr, flush=True)

    if not torch.cuda.is_available():
        return {"ok": False, "peak_mb": -1, "err": "cuda_unavailable"}

    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    try:
        # --- Build model & I/O ---
        vlog("build model")
        if use_transformer:
            model = TinyTransformer(d_model=d_model, n_heads=n_heads,
                                    n_layers=n_layers, ff_dim=ff_dim, dropout=dropout).to(dev)
            x = torch.randn(batch, seq_len, d_model, device=dev)
        else:
            model = DeepMLP(depth=depth, in_out=in_out, hidden=hidden).to(dev)
            x = torch.randn(batch, in_out, device=dev)

        opt = torch.optim.SGD(model.parameters(), 1e-3)
        scaler = torch.amp.GradScaler("cuda")

        vlog("wrap with DPCS")
        dpcs = DPCS(device_type=dev, signals_freq_steps=1, ckpt_low=0.0, ckpt_high=1.0)
        model = dpcs.wrap(model)
        dpcs._ckpt_on = bool(use_ckpt)

        if emit_snapshot:
            try:
                torch.cuda.memory._record_memory_history(enabled=True, trace_alloc_max_entries=256)
            except Exception:
                pass

        # --- Utility closures ---
        def one_step() -> tuple[float, int, float]:
            """Run exactly one full step. Returns (elapsed_ms, peak_mb, loss_val)."""
            dpcs.start_step()
            opt.zero_grad(set_to_none=True)

            torch.cuda.reset_peak_memory_stats()  # per-iteration peak measurement (docs)
            t0 = time.perf_counter()

            fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()  # non-reentrant path under the hood
            with fwd_ctx:
                y = model(x)
                loss = (y ** 2).mean()

            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            dpcs.end_step(opt, scaler)
            return elapsed_ms, peak_mb, float(loss.detach().item())

        # --- Warmup ---
        vlog(f"warmup {warmup} step(s)")
        for _ in range(max(0, warmup)):
            try:
                _ = one_step()
            except RuntimeError as e:
                return {"ok": False, "peak_mb": -1, "err": f"warmup_failed: {str(e)[:160]}"}

        # --- Measured iterations ---
        timings_ms: list[float] = []
        peaks_mb:  list[int]   = []
        last_loss: float       = float("nan")

        vlog(f"measure {repeat} step(s)")
        for _ in range(max(1, repeat)):
            try:
                ms, mb, lval = one_step()
                timings_ms.append(ms)
                peaks_mb.append(mb)
                last_loss = lval
            except RuntimeError as e:
                # Treat as failed fit/run; report partial metrics if any
                if timings_ms and peaks_mb:
                    break
                return {"ok": False, "peak_mb": -1, "err": f"measure_failed: {str(e)[:160]}"}

        # --- Summarize ---
        if len(timings_ms) == 1 and len(peaks_mb) == 1:
            # Single-step JSON (back-compat with earlier probe usage)
            out = {"ok": True, "peak_mb": int(peaks_mb[0])}
            if timed:
                out["elapsed_ms"] = round(timings_ms[0], 2)
            out["last_loss"] = round(last_loss, 8)
            return out

        # Repeated stats
        mean_ms   = statistics.fmean(timings_ms)
        median_ms = statistics.median(timings_ms)
        stdev_ms  = statistics.pstdev(timings_ms) if len(timings_ms) >= 2 else 0.0
        min_ms    = min(timings_ms)
        max_ms    = max(timings_ms)

        p95_mb    = _percentile_nearest_rank(peaks_mb, 95.0)
        med_mb    = statistics.median(peaks_mb)
        max_mb    = max(peaks_mb)

        return {
            "ok": True,
            "n_measured": len(timings_ms),
            "timing_ms": {
                "mean": round(mean_ms, 3),
                "median": round(median_ms, 3),
                "stdev": round(stdev_ms, 3),
                "min": round(min_ms, 3),
                "max": round(max_ms, 3),
            },
            "peaks_mb": {
                "p95": int(p95_mb) if not math.isnan(p95_mb) else None,
                "median": int(med_mb),
                "max": int(max_mb),
            },
            "last_loss": round(last_loss, 8),
        }

    finally:
        # Optional final snapshot (from the last step)
        if emit_snapshot and snapshot_path:
            try:
                torch.cuda.memory._dump_snapshot(snapshot_path)
            except Exception:
                pass
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-ckpt", action="store_true")
    ap.add_argument("--emit-snapshot", action="store_true")
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # repetitions
    ap.add_argument("--repeat", type=int, default=1, help="measured iterations")
    ap.add_argument("--warmup", type=int, default=0, help="unmeasured warmup steps")

    # mode
    ap.add_argument("--transformer", action="store_true", help="use Transformer (default for this file)")
    ap.add_argument("--mlp", action="store_true", help="use MLP instead")

    # MLP args
    ap.add_argument("--batch",  type=int, default=2)
    ap.add_argument("--in-out", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--depth",  type=int, default=6)

    # Transformer args
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--ff-dim",  type=int, default=3072)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)

    args = ap.parse_args()

    use_transformer = True
    if args.mlp: use_transformer = False
    if args.transformer: use_transformer = True

    res = run_one_or_many_steps(
        use_ckpt=args.use_ckpt,
        emit_snapshot=args.emit_snapshot,
        snapshot_path=args.snapshot,
        timed=args.timed,
        verbose=args.verbose,
        repeat=max(1, args.repeat),
        warmup=max(0, args.warmup),
        batch=args.batch, in_out=args.in_out, hidden=args.hidden, depth=args.depth,
        use_transformer=use_transformer,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ff_dim=args.ff_dim, seq_len=args.seq_len, dropout=args.dropout
    )
    sys.stdout.write(json.dumps(res))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
