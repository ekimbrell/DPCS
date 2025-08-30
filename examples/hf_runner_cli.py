#!/usr/bin/env python
"""
Hugging Face training CLI to evaluate DPCS on a real LM task.
Runs a 2x2 grid: (HF gradient checkpointing on/off) x (DPCS precision on/off),
logs per-run JSONL with throughput, peak CUDA memory, and eval loss/perplexity.

Fixes in this version:
  - Packs CLM data with the standard group_texts (fixed block_size) to avoid zero-length sequences
  - Filters empty tokenized examples before packing
  - Disables model.config.use_cache when gradient checkpointing is enabled
  - AMP is enabled *only* when DPCS precision is ON
  - SafeCLM collator enforces integer dtypes for ids/labels/masks
  - Version-agnostic TrainingArguments (evaluation_strategy ⇄ eval_strategy)

Example:
  python examples/hf_runner_cli.py --grid --jsonl runs.jsonl --summarize \
    --model EleutherAI/pythia-160m --dataset Salesforce/wikitext --dataset-config wikitext-2-raw-v1 \
    --max-steps 200 --batch 2 --seq 512 --sdpa math

You can later scale with Accelerate/DDP:
  accelerate launch --num_processes 2 examples/hf_runner_cli.py ...
"""
from __future__ import annotations
import argparse, json, math, os, time, sys, subprocess, inspect
from dataclasses import dataclass
from typing import Optional

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer,
    DataCollatorForLanguageModeling, TrainerCallback, TrainerControl, TrainerState
)

from dpcs import DPCS

# ---- version-agnostic TrainingArguments helper ------------------------------

def make_training_args(**kwargs):
    from transformers import TrainingArguments
    sig = inspect.signature(TrainingArguments.__init__)
    has_eval_strategy = "eval_strategy" in sig.parameters
    has_evaluation_strategy = "evaluation_strategy" in sig.parameters

    if has_eval_strategy and not has_evaluation_strategy:
        if "evaluation_strategy" in kwargs:
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    elif has_evaluation_strategy and not has_eval_strategy:
        if "eval_strategy" in kwargs:
            kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


# ---- SDPA helpers (portable) -------------------------------------------------
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _SDPA = True
except Exception:  # pragma: no cover
    _SDPA = False
    from contextlib import contextmanager
    @contextmanager
    def sdpa_kernel(*_a, **_k):
        yield
    class SDPBackend:
        MATH = "MATH"; EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"; FLASH_ATTENTION = "FLASH_ATTENTION"


def _sdpa_from_str(name: str):
    name = (name or "auto").lower()
    if not _SDPA or name == "auto":
        return None
    if name == "math":
        return SDPBackend.MATH
    if name in ("eff", "efficient"):
        return SDPBackend.EFFICIENT_ATTENTION
    if name in ("flash", "flash_attention"):
        return SDPBackend.FLASH_ATTENTION
    raise ValueError(f"Unknown sdpa backend: {name}")


# ---- Data/model builders -----------------------------------------------------

def build_dataset(dataset_id: str, dataset_config: Optional[str], tokenizer, seq_len: int,
                  n_train: int, n_eval: int):
    ds = load_dataset(dataset_id, dataset_config)

    # Tokenize without truncation; we'll pack to fixed blocks below
    text_col_names = ds["train"].column_names
    def tok_fn(batch):
        return tokenizer(batch["text"], return_attention_mask=False, truncation=False)

    ds = ds.map(tok_fn, batched=True, remove_columns=text_col_names)

    # Drop empties defensively
    ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)

    block_size = int(seq_len)
    def group_texts(examples):
        # Concatenate lists of tokens and split by block_size
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        chunks = [concatenated[i:i + block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    ds = ds.map(group_texts, batched=True)

    # Slice to requested sizes
    train = ds["train"].select(range(min(n_train, len(ds["train"]))))
    evald = (ds["validation"] if "validation" in ds else ds["test"]) \
        .select(range(min(n_eval, len(ds["validation"]) if "validation" in ds else len(ds["test"]))))

    return train, evald


def build_model(model_id: str):
    return AutoModelForCausalLM.from_pretrained(model_id)


# ---- Safe collator to enforce integer dtypes --------------------------------
class SafeCLM(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        # Enforce integral types for indices/labels
        if "input_ids" in batch and batch["input_ids"].dtype not in (torch.int64, torch.int32):
            batch["input_ids"] = batch["input_ids"].to(torch.long)
        if "labels" in batch and batch["labels"].dtype not in (torch.int64, torch.int32):
            batch["labels"] = batch["labels"].to(torch.long)
        if "attention_mask" in batch and batch["attention_mask"].dtype not in (torch.int64, torch.int32, torch.bool):
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)
        return batch


# ---- DPCS ↔ Trainer glue ----------------------------------------------------
class DPCSCallback(TrainerCallback):
    def __init__(self, sched: DPCS, model: torch.nn.Module):
        self.sched = sched
        self.model = model
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kw):
        self.sched._ckpt_on = bool(getattr(args, "gradient_checkpointing", False))
        self.sched.start_step()
    def on_backward_end(self, args, state, control, **kw):
        loss = kw.get("loss")
        self.sched.collect_signals(loss, self.model)
    def on_optimizer_step(self, args, state, control, **kw):
        self.sched.end_step(kw.get("optimizer"), scaler=None)


# ---- Runner -----------------------------------------------------------------
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
    sdpa: Optional[str]


def run_once(cfg: RunCfg, ckpt_on: bool, dpcs_precision_on: bool, jsonl: Optional[str] = None, log_modes: bool = False) -> dict:

    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Be explicit about max length to keep padding/clipping consistent
    tok.model_max_length = cfg.seq_len

    train, evald = build_dataset(cfg.dataset_id, cfg.dataset_config, tok, cfg.seq_len, cfg.n_train, cfg.n_eval)
    collate = SafeCLM(tok, mlm=False)

    # --- Device & AMP flags (gate by DPCS precision) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = bool(dpcs_precision_on and device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = bool(dpcs_precision_on and device == "cuda" and not use_bf16)

    # Model
    model = build_model(cfg.model_id)

    # HF gradient checkpointing as the ckpt dimension
    if ckpt_on:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    # Disable KV cache when using checkpointing (HF recommendation)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = not ckpt_on

    # ---- DPCS + wrapping ----------------------------------------------------
    sdpa_backend = _sdpa_from_str(cfg.sdpa)
    sched = DPCS(
        device_type=device,
        enable_precision=bool(dpcs_precision_on),
        autotune_precision=True,
        autotune_warmup_steps=50,
        wrap_types=(torch.nn.Linear, torch.nn.TransformerEncoderLayer),
        force_sdpa_in_blocks=True,
        sdpa_backends=(sdpa_backend,) if sdpa_backend is not None else (SDPBackend.MATH,) if _SDPA else ("AUTO",),
    )
    # If the scheduler exposes a runtime toggle, honor it; otherwise the ctor flag is enough.
    if hasattr(sched, "set_precision_enabled"):
        sched.set_precision_enabled(bool(dpcs_precision_on))
    model = sched.wrap(model)

    # ---- Trainer args (AMP gated by DPCS precision) -------------------------
    targs = make_training_args(
        output_dir="out-hf-dpcs",
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else None,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=ckpt_on,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=max(50, min(200, cfg.n_train // cfg.batch_size)),
        save_steps=0,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=evald,
        data_collator=collate,
        callbacks=[DPCSCallback(sched, model)],
    )
    if log_modes:
        trainer.add_callback(ModeLogger(sched, every=20))


    # measure wall + peak mem
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.memory.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sdpa_kernel(sdpa_backend) if (sdpa_backend is not None) else sdpa_kernel([SDPBackend.MATH]) if _SDPA else sdpa_kernel():
        trainer.train()
        if device == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Evaluate (not timed for throughput)
    with sdpa_kernel(sdpa_backend) if (sdpa_backend is not None) else sdpa_kernel([SDPBackend.MATH]) if _SDPA else sdpa_kernel():
        eval_metrics = trainer.evaluate()


        wall_s = t1 - t0
        peak_bytes = int(torch.cuda.memory.max_memory_allocated()) if device == "cuda" else 0

        # tokens per sample ≈ seq_len (causal LM)
        steps = max(1, trainer.state.global_step)
        seen_samples = steps * cfg.batch_size
        samp_s = seen_samples / wall_s
        tok_s = (seen_samples * cfg.seq_len) / wall_s

        eval_loss = float(eval_metrics.get("eval_loss", float("nan")))
        ppl = float(math.exp(eval_loss)) if math.isfinite(eval_loss) else float("nan")

        row = {
            "run_id": f"hf-c{int(ckpt_on)}-p{int(dpcs_precision_on)}",
            "device": device,
            "model_id": cfg.model_id,
            "dataset": f"{cfg.dataset_id}/{cfg.dataset_config or 'default'}",
            "sdpa": cfg.sdpa or "auto",
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

        if jsonl:
            os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
            with open(jsonl, "a") as f:
                f.write(json.dumps(row) + "\n")

        # pretty print
        def _fmt_bytes(b):
            if b >= (1<<30): return f"{b/(1<<30):.2f} GiB"
            if b >= (1<<20): return f"{b/(1<<20):.2f} MiB"
            return f"{b} B"

        print(
            f"run_id {row['run_id']:<8} device {device:<4} ckpt {'Y' if ckpt_on else 'N'} "
            f"prec {'Y' if dpcs_precision_on else 'N'} sdpa {row['sdpa']:<7} "
            f"avg_ms {row['avg_ms']:.2f} samp/s {row['samp_s']:.2f} tok/s {row['tok_s']:.2f} "
            f"eval_loss {row['eval_loss']:.4f} cuda_peak {_fmt_bytes(peak_bytes)}"
        )

        return row


# ---- CLI --------------------------------------------------------------------

import shutil
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
from transformers import TrainerCallback

def _mode_hist(sched) -> dict:
    # Build a {mode: count} histogram from the DPCS registry
    hist = {}
    reg = getattr(sched, "_registry", {})
    for st in reg.values():
        hist[st.mode] = hist.get(st.mode, 0) + 1
    return hist

class ModeLogger(TrainerCallback):
    def __init__(self, sched, every: int = 20):
        self.sched = sched
        self.every = max(1, int(every))
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            headroom = getattr(self.sched, "vram_headroom", lambda: None)()
            ckpts = len(getattr(self.sched, "_ckpt_selected", []))
            hist = _mode_hist(self.sched)
            print(f"[modes] step {state.global_step:>5} hist {hist} ckpt_selected {ckpts} headroom {headroom:.3f}" if headroom is not None else f"[modes] step {state.global_step:>5} hist {hist} ckpt_selected {ckpts}")


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
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--grid", action="store_true", help="Run the 2x2 grid: (ckpt ∈ {0,1})×(precision ∈ {0,1})")
    ap.add_argument("--log-modes", action="store_true", help="Print DPCS mode histogram/ckpt count every N steps")
    ap.add_argument("--clean", action="store_true", help="Delete output_dir and --jsonl before running")


    # --- Auto summarize options ----------------------------------------------
    ap.add_argument("--summarize", action="store_true", help="After --grid, auto-run summarize_jsonl.py on the JSONL")
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
    )

    _maybe_clean(args.jsonl, "out-hf-dpcs", args.clean)


    if not args.grid:
        run_once(cfg, ckpt_on=False, dpcs_precision_on=True, jsonl=args.jsonl, log_modes=args.log_modes)
    else:
        combos = [ (False, False), (False, True), (True, False), (True, True) ]
        for ck, pr in combos:
            run_once(cfg, ckpt_on=ck, dpcs_precision_on=pr, jsonl=args.jsonl, log_modes=args.log_modes)


        # auto-summarize
        if args.summarize and args.jsonl:
            summ = os.path.join(os.path.dirname(__file__), "summarize_jsonl.py")
            cmd = [sys.executable, summ, args.jsonl,
                   "--baseline", args.summary_baseline,
                   "--sort", args.summary_sort]
            if args.summary_desc:
                cmd.append("--desc")
            if args.summary_csv:
                cmd += ["--csv", args.summary_csv]
            # helpful filters to isolate current experiment rows
            cmd += ["--filter-model", args.model, "--filter-dataset", args.dataset_config]
            print(f"[hf_runner_cli] summarizing with: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print(f"[hf_runner_cli] summarize failed: {e}")


if __name__ == "__main__":
    main()
