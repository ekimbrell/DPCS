# examples/one_step_probe.py
# Runs either:
#   (A) exactly one full training step and prints {"ok": bool, "peak_mb": int}
#   (B) a short timing loop and prints {"ok": bool, "ms_per_step": float, "samples_per_sec": float, "last_loss": float}
#
# Use --bench-steps > 0 to enable (B). Otherwise (A) is used.

import argparse, json, sys
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
    def forward(self, x): return self.blocks(x)

def build_model_optim(in_out, hidden, depth, device="cuda"):
    m = DeepMLP(depth=depth, in_out=in_out, hidden=hidden).to(device)
    opt = torch.optim.SGD(m.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda")
    return m, opt, scaler

def single_step(batch, in_out, hidden, depth, use_ckpt, emit_snapshot, snapshot_path):
    from dpcs import DPCS
    if not torch.cuda.is_available():
        return {"ok": False, "peak_mb": -1, "err": "cuda_unavailable"}

    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    y = None; loss = None
    try:
        model, opt, scaler = build_model_optim(in_out, hidden, depth, dev)
        dpcs = DPCS(device_type=dev, signals_freq_steps=1, ckpt_low=0.0, ckpt_high=1.0)
        model = dpcs.wrap(model)
        dpcs._ckpt_on = bool(use_ckpt)

        x = torch.randn(batch, in_out, device=dev)

        if emit_snapshot:
            try:
                torch.cuda.memory._record_memory_history(enabled=True, trace_alloc_max_entries=256)
            except Exception:
                pass

        dpcs.start_step()
        torch.cuda.reset_peak_memory_stats()
        fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
        with fwd_ctx:
            y = model(x)
            loss = (y**2).mean()

        scaler.scale(loss).backward()
        dpcs.collect_signals(loss, model)
        scaler.step(opt); scaler.update()

        torch.cuda.synchronize()
        peak_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        ok = True
    except RuntimeError:
        ok, peak_mb = False, -1
    except Exception:
        ok, peak_mb = False, -1
    finally:
        try:
            dpcs.end_step(opt, scaler)
        except Exception:
            pass
        if emit_snapshot and snapshot_path:
            try:
                torch.cuda.memory._dump_snapshot(snapshot_path)
            except Exception:
                pass
        # cleanup
        for name in ("model","x","y","loss"):
            if name in locals():
                try: del locals()[name]
                except Exception: pass
        torch.cuda.empty_cache()

    return {"ok": ok, "peak_mb": peak_mb}

def bench_loop(batch, in_out, hidden, depth, use_ckpt, warmup, iters):
    """
    Measure average full-step time with CUDA events:
      - warmup steps (not timed)
      - measure 'iters' steps: ms_per_step and throughput = batch / (ms/1000)
    """
    from dpcs import DPCS
    if not torch.cuda.is_available():
        return {"ok": False, "ms_per_step": -1.0, "samples_per_sec": -1.0, "last_loss": float("nan")}

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    try:
        model, opt, scaler = build_model_optim(in_out, hidden, depth, dev)
        dpcs = DPCS(device_type=dev, signals_freq_steps=1, ckpt_low=0.0, ckpt_high=1.0)
        model = dpcs.wrap(model)
        dpcs._ckpt_on = bool(use_ckpt)
        model.train()

        x = torch.randn(batch, in_out, device=dev)

        # Warmup (not timed)
        for _ in range(warmup):
            dpcs.start_step()
            fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()
            with fwd_ctx:
                y = model(x); loss = (y**2).mean()
            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
            dpcs.end_step(opt, scaler)

        # Timed iterations with CUDA events (accurate device timing)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        last_loss = 0.0

        for _ in range(iters):
            dpcs.start_step()
            fwd_ctx, _ = dpcs.checkpoint_contexts_if_needed()

            start.record()
            with fwd_ctx:
                y = model(x); loss = (y**2).mean()
            scaler.scale(loss).backward()
            dpcs.collect_signals(loss, model)
            scaler.step(opt); scaler.update()
            end.record()

            torch.cuda.synchronize()  # ensure accurate timing
            total_ms += start.elapsed_time(end)  # milliseconds
            last_loss = float(loss.detach().cpu().item())
            dpcs.end_step(opt, scaler)

        ms_per_step = total_ms / max(1, iters)
        sps = batch / (ms_per_step * 1e-3)  # samples / second
        return {"ok": True, "ms_per_step": ms_per_step, "samples_per_sec": sps, "last_loss": last_loss}
    except RuntimeError:
        return {"ok": False, "ms_per_step": -1.0, "samples_per_sec": -1.0, "last_loss": float("nan")}
    except Exception:
        return {"ok": False, "ms_per_step": -1.0, "samples_per_sec": -1.0, "last_loss": float("nan")}
    finally:
        torch.cuda.empty_cache()
        
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-out", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--depth",  type=int, required=True)
    ap.add_argument("--use-ckpt", action="store_true")
    ap.add_argument("--emit-snapshot", action="store_true")
    ap.add_argument("--snapshot", type=str, default=None)
    ap.add_argument("--bench-steps", type=int, default=0, help=">0 => run timing loop")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if args.bench_steps > 0:
        res = bench_loop(args.batch, args.in_out, args.hidden, args.depth, args.use_ckpt, args.warmup, args.bench_steps)
    else:
        res = single_step(args.batch, args.in_out, args.hidden, args.depth, args.use_ckpt, args.emit_snapshot, args.snapshot)

    sys.stdout.write(json.dumps(res))
    sys.stdout.flush()

if __name__ == "__main__":
    main()
