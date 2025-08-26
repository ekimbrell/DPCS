import argparse, json, os, statistics, time
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Optional DPCS
from dpcs import DPCS

def build_encoder(d_model: int, n_heads: int, n_layers: int, ff_dim: int) -> nn.Module:
    enc_layer = TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim, batch_first=True
    )
    return TransformerEncoder(enc_layer, num_layers=n_layers)

@torch.no_grad()
def _tokens_per_step(batch: int, seq_len: int) -> int:
    return batch * seq_len

def measure_case(
    model: nn.Module,
    device: str,
    batch: int,
    seq_len: int,
    warmup: int,
    repeat: int,
    use_dpcs: bool,
    dpcs: DPCS | None = None,
):
    model.train()
    tok_per_step = _tokens_per_step(batch, seq_len)

    # Simple optimizer + scaler
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device=device)

    def one_step(i: int):
        x = torch.randn(batch, seq_len, model.layers[0].linear1.in_features, device=device)  # (B, S, D)
        torch.cuda.reset_peak_memory_stats(device)
        if use_dpcs and dpcs is not None:
            dpcs.start_step()
            opt.zero_grad(set_to_none=True)
            fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
            with fwd_ctx, torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                h = model(x)
                loss = (h**2).mean()
            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
            dpcs.end_step(opt, scaler)
        else:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                h = model(x)
                loss = (h**2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()

        torch.cuda.synchronize()
        peak_mb = int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        return float(loss.detach().item()), peak_mb

    # Warmup
    for i in range(warmup):
        one_step(i)

    # Timed
    times = []
    peaks = []
    last_loss = None
    for i in range(repeat):
        t0 = time.perf_counter()
        last_loss, peak = one_step(i)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        peaks.append(peak)

    stats = {
        "n_measured": repeat,
        "timing_ms": {
            "mean": round(statistics.mean(times), 3),
            "median": round(statistics.median(times), 3),
            "stdev": round(statistics.pstdev(times), 3),
            "min": round(min(times), 3),
            "max": round(max(times), 3),
        },
        "peaks_mb": {
            "p95": int(statistics.quantiles(peaks, n=20)[18]),  # ~p95
            "median": int(statistics.median(peaks)),
            "max": int(max(peaks)),
        },
        "tokens_per_s": round(tok_per_step / (statistics.mean(times) / 1000.0), 1),
        "last_loss": float(last_loss),
    }
    if use_dpcs and dpcs is not None:
        stats["dpcs_modes"] = dpcs.precision_mix()
        stats["ckpt_on"] = dpcs._ckpt_on
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--ff-dim", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--log", type=str, default="")
    p.add_argument("--ckpt-low", type=float, default=0.15)
    p.add_argument("--ckpt-high", type=float, default=0.35)
    p.add_argument("--ckpt-need", type=int, default=2)
    p.add_argument("--epsilon-g", type=float, default=1e-3)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--cooldown-steps", type=int, default=3)
    p.add_argument("--ema-beta", type=float, default=0.9)

    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"

    # Print allocator backend so you can verify env is picked up
    try:
        backend = torch.cuda.memory.get_allocator_backend()
    except Exception:
        backend = "unknown"
    print(f"Allocator backend: {backend}")  # e.g., cudaMallocAsync or native
    # (Some knobs like max_split_size_mb are ignored with cudaMallocAsync.)  # docs confirm this
    # https://docs.pytorch.org/docs/stable/notes/cuda.html

    # Build model
    D = args.d_model
    model_base = build_encoder(args.d_model, args.n_heads, args.n_layers, args.ff_dim).to(device)

    # Baseline
    base = measure_case(
        model=model_base, device=device, batch=args.batch, seq_len=args.seq_len,
        warmup=args.warmup, repeat=args.repeat, use_dpcs=False, dpcs=None
    )
    print("\n=== Baseline (AMP) ===")
    print(json.dumps(base, indent=2))

    # DPCS (adaptive)
    dpcs = DPCS(
        device_type=device, enable_precision=True,
        ckpt_low=args.ckpt_low, ckpt_high=args.ckpt_high, ckpt_need=args.ckpt_need,
        epsilon_g=args.epsilon_g, kappa=args.kappa,
        cooldown_steps=args.cooldown_steps, ema_beta=args.ema_beta,
    )
    model_dpcs = dpcs.wrap(build_encoder(args.d_model, args.n_heads, args.n_layers, args.ff_dim).to(device))
    dpcs_stats = measure_case(
        model=model_dpcs, device=device, batch=args.batch, seq_len=args.seq_len,
        warmup=args.warmup, repeat=args.repeat, use_dpcs=True, dpcs=dpcs
    )
    print("\n=== DPCS adaptive ===")
    print(json.dumps(dpcs_stats, indent=2))

    # Optional JSONL log
    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as f:
            row = {
                "config": vars(args),
                "allocator": backend,
                "baseline": base,
                "dpcs": dpcs_stats,
            }
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
