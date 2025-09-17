#!/usr/bin/env python3
"""
Runner for testing DPCS on both small and large models with OOM resilience.
- Imports DPCS from ./dpcs.py or /mnt/data/dpcs.py
- Backbones: TinyCNN / ResNet18 / ResNet50 / ViT-B/16 / GPT-style LM (long seq)
- Measures per-iteration throughput and step-scoped peak VRAM
- Calls DPCS.start_step() / collect_signals() / end_step() respecting curvature-probe retain_graph
- Large-model helpers: Flash/Efficient SDPA, activation checkpointing (non-reentrant, no RNG stash),
  MLP sequence-chunking, gradient accumulation, and an auto-shrink OOM backoff (reduce seq_len / increase chunks)

Examples:
  # CIFAR-10 baseline
  python runner.py --dataset cifar10 --model resnet18 --steps 1000 --use-dpcs 0 --amp bf16 --eval

  # GPT stress test (baseline vs DPCS)
  python runner.py --dataset synthetic_text --model gpt --seq-len 1024 --d-model 1536 --n-heads 12 --n-layers 24 \
    --batch-size 1 --grad-accum-steps 8 --ckpt 1 --force-sdpa 1 --mlp-chunks 2 --steps 1200 --use-dpcs 0
  python runner.py --dataset synthetic_text --model gpt --seq-len 1024 --d-model 1536 --n-heads 12 --n-layers 24 \
    --batch-size 1 --grad-accum-steps 8 --ckpt 1 --force-sdpa 1 --mlp-chunks 2 --steps 1200 --use-dpcs 1
"""
from __future__ import annotations

# ---- Environment knobs BEFORE importing torch/cublas ----
import os
# cuBLAS determinism workspace (required if torch.use_deterministic_algorithms(True) is used)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Reduce allocator fragmentation for borderline OOM cases
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import argparse
import json
import time
import types
import importlib.util
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt
from torch.utils.data import DataLoader, Dataset
from torch.nn.attention import sdpa_kernel, SDPBackend

# -------------------------- Utility: import DPCS ---------------------------

def import_dpcs_module() -> types.ModuleType:
    """Try to import DPCS from local file dpcs.py; fallback to /mnt/data/dpcs.py."""
    cand_paths = [
        os.path.join(os.path.dirname(__file__), "dpcs.py"),
        "/mnt/data/dpcs.py",
    ]
    for p in cand_paths:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("dpcs", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                return mod
    # Last try: plain import if it's on sys.path
    try:
        import dpcs as mod
        return mod
    except Exception as e:
        raise RuntimeError(
            "Could not import dpcs.py. Place dpcs.py next to runner.py or at /mnt/data/dpcs.py"
        ) from e


# ----------------------------- Small model --------------------------------

class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ------------------------------ Large Transformer (GPT-style) -------------

class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, mlp_chunks: int = 1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mlp_chunks = max(1, int(mlp_chunks))
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(mlp_ratio * d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # SDPA chooses Flash/Efficient backends via outer context
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return x + self.proj(attn_out)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln2(x)
        if self.mlp_chunks > 1:
            parts = y.chunk(self.mlp_chunks, dim=1)  # chunk along sequence
            outs = [self.mlp(p) for p in parts]
            m = torch.cat(outs, dim=1)
        else:
            m = self.mlp(y)
        return x + m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._attn(x)
        x = self._mlp(x)
        return x

class GPTLM(nn.Module):
    """Minimal GPT-like LM with optional checkpointing per block and weight tying."""
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_seq_len: int, mlp_chunks: int = 1):
        super().__init__()
        self.is_causal_lm = True
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, mlp_chunks=mlp_chunks) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (Press & Wolf 2017)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        use_ckpt = getattr(self, "_use_ckpt", False) and self.training
        for blk in self.blocks:
            if use_ckpt:
                # Non-reentrant + no RNG stash: lower recompute peak
                x = _ckpt(blk, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ------------------------------ Datasets ----------------------------------

class SyntheticText(Dataset):
    def __init__(self, n: int = 100000, seq_len: int = 2048, vocab_size: int = 50257):
        self.n = int(n)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.randint(self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.roll(x, shifts=-1)
        return x, y

class SyntheticImages(Dataset):
    def __init__(self, n: int = 50000, num_classes: int = 10, shape: Tuple[int, int, int] = (3, 32, 32)):
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.shape = shape
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.randn(*self.shape)
        y = torch.randint(0, self.num_classes, ()).long()
        return x, y


def make_dataloaders(dataset: str, batch_size: int, num_workers: int = 2, seq_len: int = 1024, vocab_size: int = 50257):
    if dataset == "synthetic_text":
        train = SyntheticText(n=100000, seq_len=seq_len, vocab_size=vocab_size)
        val = SyntheticText(n=10000, seq_len=seq_len, vocab_size=vocab_size)
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            vocab_size,
        )
    elif dataset == "cifar10":
        try:
            import torchvision
            from torchvision import transforms
            tf_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            tf_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train = torchvision.datasets.CIFAR10(root="./data", train=True, transform=tf_train, download=True)
            val = torchvision.datasets.CIFAR10(root="./data", train=False, transform=tf_val, download=True)
            train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            num_classes = 10
            return train_loader, val_loader, num_classes
        except Exception as e:
            print(f"[runner] torchvision unavailable or download failed ({e}); falling back to synthetic.")
    # synthetic fallback
    train = SyntheticImages(n=50000)
    val = SyntheticImages(n=5000)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        10,
    )

# ---------------------------- Repro & logging -----------------------------

def set_seed(seed: int, deterministic: bool = True):
    import random, numpy as np
    random.seed(seed)
    import numpy as np  # noqa: F811 (ensure numpy exists)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------------- Training ---------------------------------

def evaluate(model: nn.Module, loader: Optional[DataLoader], device: torch.device) -> float:
    if loader is None:
        return 0.0
    if getattr(model, "is_causal_lm", False):
        # No accuracy metric for synthetic LM
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return (correct / max(1, total))


def _make_optimizer(args, model):
    if getattr(args, "model", "tinycnn") == "gpt":
        try:
            import bitsandbytes as bnb  # optional
            return bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)
        except Exception:
            return torch.optim.AdamW(model.parameters(), lr=args.lr)
    return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


def _set_mlp_chunks(model: nn.Module, chunks: int):
    changed = False
    for m in model.modules():
        if isinstance(m, GPTBlock) and getattr(m, "mlp_chunks", None) != chunks:
            m.mlp_chunks = chunks
            changed = True
    return changed


def train_one_run(args, dpcs_mod: types.ModuleType, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, num_classes = make_dataloaders(
        args.dataset, args.batch_size, args.workers,
        seq_len=getattr(args, "seq_len", 1024), vocab_size=getattr(args, "vocab_size", 50257)
    )

    # Model & optimizer
    if args.model == "resnet18":
        import torchvision.models as models
        try:
            model = models.resnet18(weights=None, num_classes=num_classes)
        except Exception:
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "resnet50":
        import torchvision.models as models
        try:
            model = models.resnet50(weights=None, num_classes=num_classes)
        except Exception:
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "vit_b16":
        import torchvision.models as models
        try:
            model = models.vit_b_16(weights=None, num_classes=num_classes)
        except Exception:
            model = models.vit_b_16()
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif args.model == "gpt":
        model = GPTLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=args.seq_len,
            mlp_chunks=args.mlp_chunks,
        )
        model._use_ckpt = bool(args.ckpt)
    else:
        model = TinyCNN(num_classes=num_classes)
    model = model.to(device)

    opt = _make_optimizer(args, model)
    loss_fn = nn.CrossEntropyLoss()

    # DPCS setup
    dpcs = None
    scaler: Optional[torch.amp.GradScaler] = None
    if args.use_dpcs:
        dpcs_kwargs = {k.split('.', 1)[1]: v for k, v in vars(args).items() if k.startswith('dpcs.') and v is not None}
        dpcs_kwargs.setdefault("device_type", "cuda" if torch.cuda.is_available() else "cpu")
        dpcs = dpcs_mod.DPCS(**dpcs_kwargs)
        model = dpcs.wrap(model)
        scaler = torch.amp.GradScaler(enabled=dpcs.amp_uses_grad_scaler())
        dpcs.set_log_jsonl(os.path.join(out_dir, "dpcs.jsonl"))
    else:
        use_fp16 = (args.amp == "fp16")
        scaler = torch.amp.GradScaler(enabled=use_fp16)

    # OOM autoshrink state (GPT only)
    curr_seq_len = int(args.seq_len)
    curr_chunks = int(args.mlp_chunks)

    # Metrics log
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    with open(metrics_path, "w") as mf:
        step_idx = 0
        t0 = time.perf_counter()
        while step_idx < args.steps:
            for xb, yb in train_loader:
                if step_idx >= args.steps:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Apply autoshrink sequence length for GPT
                if args.model == "gpt":
                    if curr_seq_len < xb.size(1):
                        xb = xb[:, :curr_seq_len]
                        yb = yb[:, :curr_seq_len]

                # Step-scoped peak VRAM
                if torch.cuda.is_available():
                    try:
                        torch.cuda.memory.reset_peak_memory_stats()
                    except Exception:
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except Exception:
                            pass

                if dpcs is not None:
                    dpcs.start_step()

                # curvature probe steps may need retain_graph
                retain_graph = False
                if dpcs is not None and dpcs.cfg.curv_period > 0 and (dpcs._step % dpcs.cfg.curv_period == 0):
                    retain_graph = True

                # AMP config
                if dpcs is not None:
                    device_type, amp_dtype, amp_enabled = dpcs.get_amp_config()
                else:
                    device_type = "cuda" if torch.cuda.is_available() else "cpu"
                    amp_dtype = (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported() and args.amp == "bf16") else torch.float16)
                    amp_enabled = (args.amp in {"bf16", "fp16"}) and (device_type == "cuda")

                # SDPA backend selection
                backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
                if not args.force_sdpa:
                    backends.append(SDPBackend.MATH)

                try:
                    with sdpa_kernel(backends), torch.amp.autocast(device_type, dtype=amp_dtype, enabled=amp_enabled):
                        logits = model(xb)                     # (B, T, V)
                        V = logits.size(-1)
                        loss = F.cross_entropy(logits.reshape(-1, V), yb.reshape(-1))

                    # Gradient accumulation
                    gas = max(1, args.grad_accum_steps)
                    loss_to_backprop = loss / gas

                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss_to_backprop).backward(retain_graph=retain_graph)
                    else:
                        loss_to_backprop.backward(retain_graph=retain_graph)

                    # Step every GAS micro-steps
                    if (step_idx + 1) % gas == 0:
                        if scaler is not None and scaler.is_enabled():
                            scaler.step(opt); scaler.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

                except RuntimeError as e:
                    if ("out of memory" in str(e).lower()) and args.autoshrink and (args.model == "gpt"):
                        # OOM backoff: empty cache, increase MLP chunks up to max, else reduce seq_len
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        opt.zero_grad(set_to_none=True)

                        changed = False
                        if curr_chunks < args.max_mlp_chunks:
                            curr_chunks = min(args.max_mlp_chunks, max(2, curr_chunks * 2))
                            _set_mlp_chunks(model, curr_chunks)
                            changed = True
                        elif curr_seq_len > args.min_seq_len:
                            curr_seq_len = max(args.min_seq_len, curr_seq_len - args.oom_backoff_t)
                            changed = True
                        if changed:
                            print(f"[autoshrink] OOM caught; new mlp_chunks={curr_chunks}, seq_len={curr_seq_len}")
                            continue  # retry next batch
                        else:
                            raise
                    else:
                        raise

                # End step (precision decisions, logging)
                if dpcs is not None:
                    try:
                        dpcs.end_step(opt, scaler)
                    except Exception as e:
                        print(f"[runner] end_step error: {e}")

                # Metrics
                elapsed = time.perf_counter() - t0
                n = xb.size(0)
                throughput = n / elapsed if elapsed > 0 else 0.0
                peak_alloc = 0
                if torch.cuda.is_available():
                    try:
                        peak_alloc = int(torch.cuda.memory.max_memory_allocated())
                    except Exception:
                        try:
                            peak_alloc = int(torch.cuda.max_memory_allocated())
                        except Exception:
                            peak_alloc = 0

                try:
                    loss_scalar = float(loss.detach().float().cpu().item())
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        loss_scalar = float("nan")
                    else:
                        raise

                rec = {
                    "step": step_idx,
                    "loss": loss_scalar,
                    "elapsed_s": elapsed,
                    "throughput_sps": throughput,
                    "peak_alloc_bytes": peak_alloc,
                    "seq_len_active": (curr_seq_len if args.model == "gpt" else None),
                    "mlp_chunks": (curr_chunks if args.model == "gpt" else None),
                }
                mf.write(json.dumps(rec) + "\n"); mf.flush()
                t0 = time.perf_counter()
                step_idx += 1

                if step_idx % max(1, args.eval_every) == 0:
                    acc = evaluate(model, val_loader if args.eval else None, device)
                    with open(os.path.join(out_dir, "summary.csv"), "w") as sf:
                        sf.write("step,acc,throughput_sps,peak_alloc_bytes\n")
                        sf.write(f"{step_idx},{acc:.6f},{throughput:.3f},{peak_alloc}\n")

    return out_dir

# ----------------------------- CLI & main ---------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="DPCS runner (OOM-safe)")
    p.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "cifar10", "synthetic_text"], help="dataset to use")
    p.add_argument("--model", type=str, default="tinycnn", choices=["tinycnn", "resnet18", "resnet50", "vit_b16", "gpt"], help="model/backbone to run")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=200, help="number of training steps to run")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval", action="store_true", help="run quick eval every eval_every steps")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"], help="AMP mode when not using DPCS")
    p.add_argument("--use-dpcs", type=int, default=1, help="1 to enable DPCS; 0 to disable")

    # Memory helpers for large models
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="Accumulate gradients over this many steps (effective batch = batch_size * grad_accum_steps).")
    p.add_argument("--ckpt", type=int, default=0,
                   help="1 to enable activation checkpointing for Transformer blocks (GPT).")
    p.add_argument("--force-sdpa", type=int, default=0,
                   help="1 to force Flash/Efficient SDPA only (disables Math backend).")
    p.add_argument("--mlp-chunks", type=int, default=1,
                   help="Chunk the MLP over the sequence (e.g., 2 or 4) to reduce peak activation memory.")

    # GPT model hyperparameters (only used when --model gpt)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--d-model", dest="d_model", type=int, default=2048)
    p.add_argument("--n-heads", dest="n_heads", type=int, default=16)
    p.add_argument("--n-layers", dest="n_layers", type=int, default=24)

    # OOM autoshrink controls
    p.add_argument("--autoshrink", type=int, default=1, help="Enable OOM auto-shrink for GPT (1/0)")
    p.add_argument("--min-seq-len", type=int, default=512, help="Minimum sequence length when shrinking")
    p.add_argument("--max-mlp-chunks", type=int, default=8, help="Maximum MLP chunks when auto-increasing")
    p.add_argument("--oom-backoff-t", type=int, default=128, help="Tokens to remove from seq_len when shrinking")

    # DPCS knobs (prefix with dpcs.) — override any DPCSConfig field
    p.add_argument("--dpcs.device_type", type=str, default=None)
    p.add_argument("--dpcs.enable_precision", type=int, default=None)
    p.add_argument("--dpcs.epsilon_g_low", type=float, default=None)
    p.add_argument("--dpcs.epsilon_g_high", type=float, default=None)
    p.add_argument("--dpcs.kappa_low", type=float, default=None)
    p.add_argument("--dpcs.kappa_high", type=float, default=None)
    p.add_argument("--dpcs.mode_patience", type=int, default=None)
    p.add_argument("--dpcs.curv_period", type=int, default=None)
    p.add_argument("--dpcs.hvp_power_iters", type=int, default=None)
    p.add_argument("--dpcs.max_modules_per_probe", type=int, default=None)
    p.add_argument("--dpcs.low_headroom_frac", type=float, default=None)
    p.add_argument("--dpcs.hi_headroom_frac", type=float, default=None)
    p.add_argument("--dpcs.ckpt_patience", type=int, default=None)
    p.add_argument("--dpcs.ckpt_topk_frac", type=float, default=None)
    p.add_argument("--dpcs.ckpt_use_benefit_score", type=int, default=None)
    p.add_argument("--dpcs.min_activation_bytes_to_ckpt", type=int, default=None)
    p.add_argument("--dpcs.te_amax_history_len", type=int, default=None)
    p.add_argument("--dpcs.te_margin_init", type=int, default=None)
    p.add_argument("--dpcs.te_margin_inc", type=int, default=None)
    p.add_argument("--dpcs.te_margin_dec", type=int, default=None)

    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed, deterministic=True)

    out_dir = os.path.join("results", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    dpcs_mod = import_dpcs_module()
    train_one_run(args, dpcs_mod, out_dir)
    print(f"[runner] Done. Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
