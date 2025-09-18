"""Hugging Face training CLI to evaluate DPCS on a real LM task (rewritten).

Key changes from your original draft:
- **Manual training loop** (no HF Trainer) so DPCS can control **precision per step**
  and toggle checkpointing without fighting Trainer internals.
- **Fair baselines**: choose static AMP via `--amp {off,fp16,bf16}` independently of
  `--dpcs-precision` (which enables DPCS' adaptive precision policy).
- **No double-ckpt**: if `--ckpt` (HF gradient checkpointing) is on, DPCS leaf-ckpt
  is disabled to avoid double recompute cost. The leaf planner is opt-in via
  `--dpcs-ckpt-topk-frac` so the baseline remains comparable when toggling `--ckpt`.
- **Throughput timing excludes eval**; eval runs after training, separately.
- **Robust dataset loader**: auto-detects text column, tokenizes without truncation,
  packs to fixed blocks, drops empty examples, and uses a safe LM collator.
- **SDPA**: uses `sdpa_kernel([backend])` (list form) and falls back cleanly.
- **JSONL logging** of wall-time, samp/s, tok/s, eval loss/ppl, CUDA peak bytes.

Example (single run):
  python examples/hf_runner_cli.py \
    --model EleutherAI/pythia-160m --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --seq 512 --train-samples 5000 --eval-samples 1000 --batch 2 --max-steps 200 \
    --sdpa math --amp bf16 --ckpt 0 --dpcs-precision 1 --jsonl runs.jsonl

Grid (2x2 over HF-ckpt x DPCS-precision):
  python examples/hf_runner_cli.py --grid --jsonl runs.jsonl --summarize \
    --model EleutherAI/pythia-160m --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --seq 512 --train-samples 5000 --eval-samples 1000 --batch 2 --max-steps 200 --sdpa math --amp bf16
"""
from __future__ import annotations

import argparse, json, math, os, sys, time, subprocess, shutil
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Make the src/ package importable when running from examples/ ---
_THIS_DIR = os.path.dirname(__file__)
_SRC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "src"))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from contextlib import nullcontext
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# DPCS (our package)
from dpcs import DPCS, DPCSConfig

# ---- SDPA helpers (portable) ----------------------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAVE_SDPA = True
except Exception:  # pragma: no cover
    _HAVE_SDPA = False
    from contextlib import contextmanager
    @contextmanager
    def sdpa_kernel(*_a, **_k):
        yield
    class SDPBackend:  # type: ignore
        MATH = "MATH"; EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"; FLASH_ATTENTION = "FLASH_ATTENTION"

def sdpa_from_str(name: str | None):
    if not name or not _HAVE_SDPA:
        return None
    s = name.lower()
    if s == "math": return SDPBackend.MATH
    if s in ("efficient", "eff"): return SDPBackend.EFFICIENT_ATTENTION
    if s in ("flash", "flash_attention"): return SDPBackend.FLASH_ATTENTION
    if s == "auto": return None
    raise ValueError(f"unknown sdpa backend: {name}")

# ---- Data / tokenization ---------------------------------------------------

def _guess_text_col(split) -> str:
    # Prefer common names, else first string column
    prefs = {"text", "content", "document"}
    for c in split.column_names:
        if c.lower() in prefs:
            return c
    for c in split.column_names:
        val = split[c][0]
        if isinstance(val, str):
            return c
    raise ValueError("No text-like column found in dataset")


def build_dataset(dataset_id: str, dataset_config: Optional[str], tokenizer, seq_len: int,
                  n_train: int, n_eval: int):
    ds = load_dataset(dataset_id, dataset_config)
    text_col = _guess_text_col(ds["train"]) if "train" in ds else _guess_text_col(ds["validation"])  # heuristic

    def tok_fn(batch):
        return tokenizer(batch[text_col], return_attention_mask=False, truncation=False)

    remove_cols = ds["train"].column_names if "train" in ds else None
    ds = ds.map(tok_fn, batched=True, remove_columns=remove_cols)

    # Drop empty tokenized samples
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)

    block = int(seq_len)
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_len = (len(concatenated) // block) * block
        if total_len == 0:
            return {"input_ids": [], "labels": []}
        chunks = [concatenated[i:i+block] for i in range(0, total_len, block)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    ds = ds.map(group_texts, batched=True)

    train = ds["train"].select(range(min(n_train, len(ds["train"]))))
    eval_split = ds["validation"] if "validation" in ds else ds["test"]
    evald = eval_split.select(range(min(n_eval, len(eval_split))))
    return train, evald


class SafeCLM:
    """Minimal collator enforcing correct integer dtypes for LM."""
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
    def __call__(self, features):
        # features is list of {input_ids, labels}
        max_len = max(len(f["input_ids"]) for f in features)
        def pad(seq):
            return seq + [self.pad_token_id] * (max_len - len(seq))
        inps = torch.tensor([pad(f["input_ids"]) for f in features], dtype=torch.long)
        labs = torch.tensor([pad(f["labels"]) for f in features], dtype=torch.long)
        return {"input_ids": inps, "labels": labs}

# ---- AMP helpers -----------------------------------------------------------

def amp_from_flag(flag: str, device: str) -> Tuple[str, torch.dtype | None, bool]:
    """Return (device_type, dtype, enabled) for static AMP choice."""
    if device != "cuda":
        return device, None, False
    s = flag.lower()
    if s == "bf16" and torch.cuda.is_bf16_supported():
        return "cuda", torch.bfloat16, True
    if s == "fp16":
        return "cuda", torch.float16, True
    return "cuda", None, False  # off / unsupported

# ---- Run config ------------------------------------------------------------

@dataclass
class RunCfg:
    model_id: str
    dataset_id: str
    dataset_config: Optional[str]
    seq_len: int
    n_train: int
    n_eval: int
    batch_size: int
    lr: float
    max_steps: int
    epochs: int
    seed: int
    sdpa: str
    amp: str  # off|fp16|bf16
    dpcs_ckpt_topk_frac: float
    dpcs_epsilon_g_low: float
    dpcs_epsilon_g_high: float

# ---- Single run ------------------------------------------------------------

def run_once(cfg: RunCfg, ckpt_on: bool, dpcs_precision_on: bool, jsonl: Optional[str] = None) -> dict:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer & data
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:  # ensure pad exists
        tok.pad_token = tok.eos_token
    tok.model_max_length = 1_000_000

    train_ds, eval_ds = build_dataset(cfg.dataset_id, cfg.dataset_config, tok, cfg.seq_len, cfg.n_train, cfg.n_eval)
    collate = SafeCLM(tok)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device=="cuda"), drop_last=True, collate_fn=collate)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=2, pin_memory=(device=="cuda"), drop_last=False, collate_fn=collate)

    # Model
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id)
    model.to(device)
    model.train()

    # HF gradient checkpointing (dimension of the grid); disable DPCS leaf-ckpt when HF ckpt is on
    if ckpt_on and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Optimizer & LR schedule (simple AdamW + linear warmup)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else (cfg.epochs * max(1, len(train_loader)))
    warmup = max(0, int(0.06 * total_steps))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    # DPCS scheduler (adaptive precision + optional leaf ckpt)
    sdpa_backend = sdpa_from_str(cfg.sdpa)
    ckpt_frac = float(cfg.dpcs_ckpt_topk_frac)
    if ckpt_on:
        ckpt_frac = 0.0

    dpcs_kwargs = dict(
        device_type=device,
        enable_precision=bool(dpcs_precision_on),
        curv_period=0,  # turn off curvature probes for clean throughput
        ckpt_topk_frac=ckpt_frac,
        min_activation_bytes_to_ckpt=1<<21,  # ignore very small activations
        epsilon_g_low=cfg.dpcs_epsilon_g_low,
        epsilon_g_high=cfg.dpcs_epsilon_g_high,
    )
    dpcs = DPCS(**DPCSConfig.from_kwargs(**dpcs_kwargs).to_kwargs())
    model = dpcs.wrap(model)

    dpcs.set_log_jsonl("out-hf-dpcs/dpcs.jsonl")


    # AMP mode (static baseline when DPCS precision is OFF)
    static_dev, static_dtype, static_enabled = amp_from_flag(cfg.amp, device)

    # Use a GradScaler only when DPCS requests fp16 *this* step
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # SDPA context
    sdpa_ctx = (lambda: sdpa_kernel([sdpa_backend])) if (sdpa_backend is not None) else \
               (lambda: sdpa_kernel([SDPBackend.MATH])) if _HAVE_SDPA else (lambda: sdpa_kernel())

    # ---- Train (measure only train time) ----
    cuda_total_bytes = None
    cuda_min_free_bytes = None
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # Track the smallest observed free VRAM so we can report a physical
        # peak instead of PyTorch's virtual allocation counter (which may count
        # Unified Memory spillover and therefore exceed the device capacity).
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        cuda_total_bytes = int(total_bytes)
        cuda_min_free_bytes = int(free_bytes)
    t0 = time.perf_counter()

    steps_done = 0
    model.train()
    for epoch in range(max(1, cfg.epochs)):
        for batch in train_loader:
            if cfg.max_steps and steps_done >= cfg.max_steps:
                break
            dpcs.start_step()
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # choose AMP per step
            if dpcs_precision_on:
                dev, dtype, enabled = dpcs.get_amp_config()
            else:
                dev, dtype, enabled = static_dev, static_dtype, static_enabled

            amp_ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=enabled) if device == "cuda" else nullcontext()
            with sdpa_ctx():
                with amp_ctx:
                    out = model(**inputs)
                    loss = out.loss

            opt.zero_grad(set_to_none=True)
            use_scaler = enabled and (dtype is torch.float16)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()

            dpcs.end_step(opt, scaler if use_scaler else None)


            if device == "cuda" and cuda_min_free_bytes is not None:
                # `mem_get_info` reflects the current free physical memory on the
                # device. Track the minimum to estimate a physical peak usage.
                free_bytes, _ = torch.cuda.mem_get_info()
                cuda_min_free_bytes = min(cuda_min_free_bytes, int(free_bytes))

            steps_done += 1
        if cfg.max_steps and steps_done >= cfg.max_steps:
            break

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # ---- Eval (not included in throughput timing) ----
    model.eval()
    eval_loss_sum, eval_tokens = 0.0, 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            eval_amp_ctx = torch.amp.autocast("cuda", dtype=static_dtype, enabled=static_enabled) if device == "cuda" else nullcontext()
            with sdpa_ctx():
                with eval_amp_ctx:
                    out = model(**inputs)
                    loss = out.loss
            eval_loss_sum += float(loss.detach().cpu()) * inputs["input_ids"].size(0)
            eval_tokens += inputs["input_ids"].numel()
    eval_loss = eval_loss_sum / max(1, len(eval_loader.dataset))
    ppl = math.exp(eval_loss) if math.isfinite(eval_loss) else float("nan")

    # ---- Metrics & logging ----
    wall_s = (t1 - t0)
    steps = max(1, steps_done)
    seen_samples = steps * cfg.batch_size
    samp_s = seen_samples / wall_s
    tok_s = (seen_samples * cfg.seq_len) / wall_s
    peak_alloc_bytes = int(torch.cuda.memory.max_memory_allocated()) if device == "cuda" else 0
    peak_bytes = peak_alloc_bytes
    oversub_bytes = 0
    if device == "cuda" and cuda_total_bytes is not None:
        peak_bytes = min(peak_alloc_bytes, cuda_total_bytes)
        if cuda_min_free_bytes is not None:
            peak_bytes = min(peak_bytes, cuda_total_bytes - cuda_min_free_bytes)
        oversub_bytes = max(0, peak_alloc_bytes - cuda_total_bytes)

    row = {
        "run_id": f"hf-c{int(ckpt_on)}-p{int(dpcs_precision_on)}",
        "device": device,
        "model_id": cfg.model_id,
        "dataset": f"{cfg.dataset_id}/{cfg.dataset_config or 'default'}",
        "sdpa": cfg.sdpa,
        "amp": cfg.amp,
        "batch": cfg.batch_size,
        "seq": cfg.seq_len,
        "max_steps": cfg.max_steps,
        "epochs": cfg.epochs,
        "avg_ms": (wall_s / steps) * 1000.0,
        "samp_s": samp_s,
        "tok_s": tok_s,
        "eval_loss": eval_loss,
        "ppl": ppl,
        "cuda_peak": peak_bytes,
    }

    if device == "cuda":
        row["cuda_peak_alloc"] = peak_alloc_bytes
        row["cuda_oversub"] = oversub_bytes

    if jsonl:
        os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _fmt_bytes(b):
        if b >= (1<<30): return f"{b/(1<<30):.2f} GiB"
        if b >= (1<<20): return f"{b/(1<<20):.2f} MiB"
        return f"{b} B"
    extra = ""
    if device == "cuda" and oversub_bytes:
        extra = f" (virt {_fmt_bytes(peak_alloc_bytes)}; oversub {_fmt_bytes(oversub_bytes)})"
    print(
        f"run_id {row['run_id']:<8} device {device:<4} ckpt {'Y' if ckpt_on else 'N'} "
        f"dpcs_prec {'Y' if dpcs_precision_on else 'N'} sdpa {row['sdpa']:<7} amp {row['amp']:<4} "
        f"avg_ms {row['avg_ms']:.2f} samp/s {row['samp_s']:.2f} tok/s {row['tok_s']:.2f} "
        f"eval_loss {row['eval_loss']:.4f} cuda_peak {_fmt_bytes(peak_bytes)}{extra}")

    return row

# ---- CLI -------------------------------------------------------------------

def _maybe_clean(jsonl_path: Optional[str], out_dir: str, clean: bool):
    if not clean:
        return
    try:
        if jsonl_path and os.path.exists(jsonl_path):
            os.remove(jsonl_path)
            print(f"[hf_runner_cli] removed {jsonl_path}")
    except Exception as e:
        print(f"[hf_runner_cli] warn: failed to remove {jsonl_path}: {e}")
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"[hf_runner_cli] removed {out_dir}/")
    except Exception as e:
        print(f"[hf_runner_cli] warn: failed to remove {out_dir}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--train-samples", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=200, help="0=use epochs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--sdpa", default="math", choices=["auto","math","efficient","flash"], help="Scaled-dot attention backend")
    ap.add_argument("--amp", default="bf16", choices=["off","fp16","bf16"], help="Static AMP baseline (used when --dpcs-precision=0)")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--grid", action="store_true", help="Run the 2x2 grid: (ckpt ∈ {0,1})×(DPCS precision ∈ {0,1})")
    ap.add_argument("--ckpt", type=int, default=0, help="HF gradient checkpointing on(1)/off(0) for single-run mode")
    ap.add_argument("--dpcs-precision", type=int, default=1, help="Enable DPCS adaptive precision policy")
    ap.add_argument("--dpcs-ckpt-topk-frac", type=float, default=0.0,
                    help="Fraction of heaviest leaves DPCS checkpoints when HF ckpt is off")
    ap.add_argument("--dpcs-epsilon-g-low", type=float, default=5e-4,
                    help="Gradient-variance threshold to drop precision (smaller => more fp32)")
    ap.add_argument("--dpcs-epsilon-g-high", type=float, default=2e-3,
                    help="Gradient-variance threshold to raise precision")
    ap.add_argument("--clean", action="store_true", help="Delete output_dir and --jsonl before running")

    # Auto-summarize helpers (optional external script)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--summary-baseline", default="hf-c0-p0")
    ap.add_argument("--summary-sort", default="avg_ms", choices=["avg_ms","samp_s","tok_s","cuda_peak","eval_loss","ppl"]) 
    ap.add_argument("--summary-desc", action="store_true")
    ap.add_argument("--summary-csv", default=None)

    args = ap.parse_args()

    cfg = RunCfg(
        model_id=args.model,
        dataset_id=args.dataset,
        dataset_config=args.dataset_config,
        seq_len=args.seq,
        n_train=args.train_samples,
        n_eval=args.eval_samples,
        batch_size=args.batch,
        lr=args.lr,
        max_steps=args.max_steps,
        epochs=args.epochs,
        seed=args.seed,
        sdpa=args.sdpa,
        amp=args.amp,
        dpcs_ckpt_topk_frac=args.dpcs_ckpt_topk_frac,
        dpcs_epsilon_g_low=args.dpcs_epsilon_g_low,
        dpcs_epsilon_g_high=args.dpcs_epsilon_g_high,
    )

    _maybe_clean(args.jsonl, "out-hf-dpcs", args.clean)

    if not args.grid:
        run_once(cfg, ckpt_on=bool(args.ckpt), dpcs_precision_on=bool(args.dpcs_precision), jsonl=args.jsonl)
    else:
        combos = [(0,0),(0,1),(1,0),(1,1)]
        for ck, pr in combos:
            run_once(cfg, ckpt_on=bool(ck), dpcs_precision_on=bool(pr), jsonl=args.jsonl)
        # optional external summarize script (kept for parity with your workflow)
        if args.summarize and args.jsonl:
            summ = os.path.join(os.path.dirname(__file__), "summarize_jsonl.py")
            cmd = [sys.executable, summ, args.jsonl,
                   "--baseline", args.summary_baseline,
                   "--sort", args.summary_sort]
            if args.summary_desc:
                cmd.append("--desc")
            if args.summary_csv:
                cmd += ["--csv", args.summary_csv]
            cmd += ["--filter-model", args.model, "--filter-dataset", args.dataset_config]
            print(f"[hf_runner_cli] summarizing with: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"[hf_runner_cli] summarize failed: {e}")

if __name__ == "__main__":
    main()
