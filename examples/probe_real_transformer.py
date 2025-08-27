import argparse, json, os, time
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from dpcs import DPCS


class TinyEncoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=6, ff_dim=2048):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.enc(x)
        return self.proj(F.gelu(h))


def summarize_times(ms_list):
    import statistics as st
    return {
        "mean": round(st.mean(ms_list), 3),
        "median": round(st.median(ms_list), 3),
        "stdev": round(st.pstdev(ms_list), 3),
        "min": round(min(ms_list), 3),
        "max": round(max(ms_list), 3),
    }


def run_case(
    *,
    device: str,
    batch: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    ff_dim: int,
    warmup: int,
    repeat: int,
    use_dpcs: bool,
):
    torch.cuda.empty_cache()
    model = TinyEncoder(d_model, n_heads, n_layers, ff_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device=device)

    # Optionally wrap with DPCS
    dpcs = None
    if use_dpcs:
        dpcs = DPCS(
            device_type=device,
            enable_precision=True,
            epsilon_g=1e3,      # tuned to allow fp16 for this synthetic
            kappa=1e-1,
            cooldown_steps=2,
            ema_beta=0.9,
            ckpt_low=0.05, ckpt_high=0.15, ckpt_need=2,
        )
        model = dpcs.wrap(model)
    if args.log:  # path like runs/dpcs_probe.jsonl
        dpcs.set_log_jsonl(args.log)
        # Optional: write a run header/meta record once
        meta = {
            "event": "meta",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "config": {
                "batch": args.batch,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "ff_dim": args.ff_dim,
                "seq_len": args.seq_len,
            },
            "dpcs_cfg": dpcs.cfg.__dict__,
        }
        dpcs._log_cb(meta)  # one-off header

    # Input
    x = torch.randn(batch, seq_len, d_model, device=device)

    # Warmup
    for _ in range(max(1, warmup)):
        if dpcs: dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            y = model(x)
            loss = (y ** 2).mean()
        scaler.scale(loss).backward()
        if dpcs: dpcs.collect_signals(loss, model)
        scaler.step(opt); scaler.update()
        if dpcs: dpcs.end_step(opt, scaler)
        torch.cuda.synchronize()

    # Timed loop
    times = []
    peaks = []
    last_loss = None
    for _ in range(repeat):
        if dpcs: dpcs.start_step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            y = model(x)
            loss = (y ** 2).mean()
        scaler.scale(loss).backward()
        if dpcs: dpcs.collect_signals(loss, model)
        scaler.step(opt); scaler.update()
        if dpcs: dpcs.end_step(opt, scaler)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        times.append(dt_ms)
        peaks.append(int(torch.cuda.max_memory_allocated() / (1024 * 1024)))
        last_loss = float(loss.detach().item())

    # Summaries
    timing_summary = summarize_times(times)
    n_measured = len(times)
    p95_peak = sorted(peaks)[int(0.95 * (n_measured - 1))]
    tokens_per_s = (batch * seq_len) / (timing_summary["mean"] / 1000.0)

    result = {
        "n_measured": n_measured,
        "timing_ms": timing_summary,
        "peaks_mb": {"p95": p95_peak, "median": int(sorted(peaks)[n_measured // 2]), "max": max(peaks)},
        "tokens_per_s": round(tokens_per_s, 1),
        "last_loss": last_loss,
    }

    if dpcs:
        # Optional diagnostics
        try:
            result["dpcs_modes"] = dpcs.precision_mix()
            result["ckpt_on"] = bool(getattr(dpcs, "_ckpt_on", False))
        except Exception:
            pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--ff-dim", type=int, default=2048)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    device = "cuda"
    print("Allocator backend:", torch.cuda.get_allocator_backend())

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "device": device,
        "config": {
            "batch": args.batch, "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "ff_dim": args.ff_dim, "seq_len": args.seq_len,
            "warmup": args.warmup, "repeat": args.repeat,
        },
    }

    print("\n=== Baseline (AMP) ===")
    base = run_case(
        device=device, batch=args.batch, seq_len=args.seq_len,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, ff_dim=args.ff_dim,
        warmup=args.warmup, repeat=args.repeat, use_dpcs=False,
    )
    print(json.dumps(base, indent=2))

    print("\n=== DPCS adaptive ===")
    dpcs_res = run_case(
        device=device, batch=args.batch, seq_len=args.seq_len,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, ff_dim=args.ff_dim,
        warmup=args.warmup, repeat=args.repeat, use_dpcs=True,
    )
    print(json.dumps(dpcs_res, indent=2))

    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        with open(args.log, "a", encoding="utf-8") as f:
            baseline_record = {"kind": "baseline", **meta, **base}
            dpcs_record = {"kind": "dpcs", **meta, **dpcs_res}
            f.write(json.dumps(baseline_record) + "\n")
            f.write(json.dumps(dpcs_record) + "\n")


if __name__ == "__main__":
    main()
