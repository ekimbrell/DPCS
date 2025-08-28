#!/usr/bin/env python
"""
Hugging Face training CLI to evaluate DPCS on a real LM task.
Runs a 2x2 grid: (HF gradient checkpointing on/off) x (DPCS precision on/off),
logs per-run JSONL with throughput, peak CUDA memory, and eval loss/perplexity.

This revision adds:
  - Always disables model.config.use_cache during training (fair baselines)
  - Dynamically adds model-specific transformer blocks to DPCS.wrap_types
      * GPT-NeoX/Pythia → GPTNeoXLayer
      * GPT-2 → GPT2Block
      * Llama/Mistral → LlamaDecoderLayer / MistralDecoderLayer (best-effort)
  - Packs CLM data with group_texts to avoid zero-length sequences
  - SafeCLM collator enforces integer dtypes for ids/labels/masks
  - AMP enabled only when DPCS precision is ON
  - Version-agnostic TrainingArguments adapter
  - **Timing hardened with CUDA fences, and training-only avg_ms**
  - **FP32 evaluation forced (fp16_full_eval=False, bf16_full_eval=False)**
  - **Adds `cuda_peak` (alias of max_memory_allocated) plus reserved** for the summarizer
"""
from __future__ import annotations
import argparse, json, math, os, time, sys, subprocess, inspect
from dataclasses import dataclass
from typing import Optional, List, Tuple, Type

import torch
import torch.nn as nn
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
    eval_src = ds["validation"] if "validation" in ds else ds["test"]
    evald = eval_src.select(range(min(n_eval, len(eval_src))))
    return train, evald


def build_model(model_id: str):
    return AutoModelForCausalLM.from_pretrained(model_id)

# ---- Safe collator to enforce integer dtypes --------------------------------
class SafeCLM(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        if "input_ids" in batch and batch["input_ids"].dtype not in (torch.int64, torch.int32):
            batch["input_ids"] = batch["input_ids"].to(torch.long)
        if "labels" in batch and batch["labels"].dtype not in (torch.int64, torch.int32):
            batch["labels"] = batch["labels"].to(torch.long)
        if "attention_mask" in batch and batch["attention_mask"].dtype not in (torch.int64, torch.int32, torch.bool):
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)
        return batch

# ---- DPCS ↔ Trainer glue ----------------------------------------------------
class DPCSCallback(TrainerCallback):
    def __init__(self, sched: DPCS, model: nn.Module):
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

# ---- Helper: pick wrap_types based on model family --------------------------

def pick_wrap_types(model: nn.Module) -> Tuple[Type[nn.Module], ...]:
    types: List[Type[nn.Module]] = [nn.Linear]

    # always catch generic encoder blocks used in toy tests
    types.append(nn.TransformerEncoderLayer)

    mt = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(mt, str):
        mt = mt.lower()
    try:
        if mt == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
            types.append(GPTNeoXLayer)
        elif mt == "gpt2":
            from transformers.models.gpt2.modeling_gpt2 import GPT2Block
            types.append(GPT2Block)
        elif mt == "llama":
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            types.append(LlamaDecoderLayer)
        elif mt == "mistral":
            try:
                from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
                types.append(MistralDecoderLayer)
            except Exception:
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                types.append(LlamaDecoderLayer)
    except Exception:
        # Best-effort: if import fails, continue with whatever we have
        pass

    # Deduplicate while preserving order
    seen = set()
    out: List[Type[nn.Module]] = []
    for t in types:
        if t not in seen:
            seen.add(t); out.append(t)
    return tuple(out)

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

def run_once(cfg: RunCfg, ckpt_on: bool, dpcs_precision_on: bool, jsonl: Optional[str] = None) -> dict:
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.model_max_length = cfg.seq_len

    train, evald = build_dataset(cfg.dataset_id, cfg.dataset_config, tok, cfg.seq_len, cfg.n_train, cfg.n_eval)
    collate = SafeCLM(tok, mlm=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    model = build_model(cfg.model_id)

    # Always disable KV cache during training for fair comparisons / GC compat
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # HF gradient checkpointing as the ckpt dimension
    if ckpt_on:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    # DPCS
    sdpa_backend = _sdpa_from_str(cfg.sdpa)
    wrap_types = pick_wrap_types(model)
    sched = DPCS(
        device_type=device,
        enable_precision=bool(dpcs_precision_on),
        autotune_precision=True,
        autotune_warmup_steps=50,
        wrap_types=wrap_types,
        force_sdpa_in_blocks=True,
        sdpa_backends=(sdpa_backend,) if sdpa_backend is not None else (SDPBackend.MATH,) if _SDPA else ("AUTO",),
    )
    model = sched.wrap(model)

    # AMP policy: enable mixed precision ONLY when DPCS precision is ON
    use_fp16 = bool(dpcs_precision_on and device == "cuda")
    use_bf16 = bool(dpcs_precision_on and device == "cuda" and not use_fp16 and torch.cuda.is_bf16_supported())

    args = make_training_args(
        output_dir="out-hf-dpcs",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps if cfg.max_steps > 0 else None,
        fp16=use_fp16,
        bf16=use_bf16,
        # force FP32 eval for apples-to-apples
        fp16_full_eval=False,
        bf16_full_eval=False,
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
        args=args,
        train_dataset=train,
        eval_dataset=evald,
        data_collator=collate,
        callbacks=[DPCSCallback(sched, model)],
    )

    # measure TRAINING wall + peak mem only (exclude eval), with proper CUDA fences
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = sdpa_kernel(sdpa_backend) if (sdpa_backend is not None) else sdpa_kernel([SDPBackend.MATH]) if _SDPA else sdpa_kernel()
    with ctx:
        trainer.train()
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Now run evaluation OUTSIDE the timing window (still under same SDPA selection)
    eval_metrics = trainer.evaluate()

    wall_s = t1 - t0
    peak_alloc = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0

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
        # Memory fields used by summarizer and for richer analysis
        "cuda_peak": peak_alloc,                # alias for compatibility
        "cuda_peak_alloc": peak_alloc,
        "cuda_peak_reserved": peak_reserved,
    }

    if jsonl:
        os.makedirs(os.path.dirname(jsonl) or ".", exist_ok=True)
        with open(jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _fmt_bytes(b):
        if b >= (1<<30): return f"{b/(1<<30):.2f} GiB"
        if b >= (1<<20): return f"{b/(1<<20):.2f} MiB"
        return f"{b} B"

    print(
        f"run_id {row['run_id']:<8} device {device:<4} ckpt {'Y' if ckpt_on else 'N'} "
        f"prec {'Y' if dpcs_precision_on else 'N'} sdpa {row['sdpa']:<7} "
        f"avg_ms {row['avg_ms']:.2f} samp/s {row['samp_s']:.2f} tok/s {row['tok_s']:.2f} "
        f"eval_loss {row['eval_loss']:.4f} alloc {_fmt_bytes(peak_alloc)} reserved {_fmt_bytes(peak_reserved)}"
    )

    return row

# ---- CLI --------------------------------------------------------------------

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

    # summary args are passed through to summarize_jsonl.py if present
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--summary-baseline", default="hf-c0-p0")
    ap.add_argument("--summary-sort", default="avg_ms", choices=["avg_ms","samp_s","tok_s","cuda_peak_alloc","cuda_peak_reserved","eval_loss","ppl"])
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

    if not args.grid:
        run_once(cfg, ckpt_on=False, dpcs_precision_on=True, jsonl=args.jsonl)
    else:
        combos = [ (False, False), (False, True), (True, False), (True, True) ]
        for ck, pr in combos:
            run_once(cfg, ckpt_on=ck, dpcs_precision_on=pr, jsonl=args.jsonl)

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
