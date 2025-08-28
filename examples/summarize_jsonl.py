#!/usr/bin/env python
"""
Summarize JSONL results produced by examples/hf_runner_cli.py.
Prints a clean, bench-like table with optional baseline-relative deltas.

Usage:
  python examples/summarize_jsonl.py runs.jsonl \
    --baseline hf-c0-p0 \
    --filter-model EleutherAI/pythia-160m \
    --filter-dataset wikitext-2-raw-v1

Optional:
  --csv out.csv          # also dump a CSV
  --sort avg_ms          # key: avg_ms|samp_s|tok_s|cuda_peak|eval_loss|ppl
  --desc                 # sort descending
"""
from __future__ import annotations
import argparse, json, math, csv, os, sys
from typing import List, Dict, Any, Optional

COLS = [
    ("run_id", 12, str),
    ("ckpt", 4, str),
    ("prec", 4, str),
    ("sdpa", 7, str),
    ("avg_ms", 9, float),
    ("samp_s", 10, float),
    ("tok_s", 10, float),
    ("eval_loss", 10, float),
    ("ppl", 9, float),
    ("cuda_peak", 12, int),
]


def fmt_bytes(b: int) -> str:
    if b >= (1 << 30):
        return f"{b/(1<<30):.2f} GiB"
    if b >= (1 << 20):
        return f"{b/(1<<20):.2f} MiB"
    return f"{b} B"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def filter_rows(rows: List[Dict[str, Any]], model: Optional[str], dataset: Optional[str], sdpa: Optional[str]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if model and model not in str(r.get("model_id", "")):
            continue
        if dataset and dataset not in str(r.get("dataset", "")):
            continue
        if sdpa and sdpa.lower() != str(r.get("sdpa", "")).lower():
            continue
        out.append(r)
    return out


def pick_baseline(rows: List[Dict[str, Any]], baseline: Optional[str]) -> Optional[Dict[str, Any]]:
    if baseline:
        for r in rows:
            if r.get("run_id") == baseline:
                return r
        return None
    # default heuristic: first ckpt=0,prec=0 if present
    for r in rows:
        if r.get("run_id", "").startswith("hf-c0-p0"):
            return r
    return rows[0] if rows else None


def as_bool_str(run_id: str, key: str) -> str:
    # run_id like hf-c{0/1}-p{0/1}
    try:
        flag = run_id.split("-")
        c = {seg[0]: seg[1] for seg in (s.split("c")[-1] if s.startswith("hf") else s for s in run_id.split("-"))}
    except Exception:
        pass
    if key == "ckpt":
        return "Y" if "-c1-" in run_id or run_id.endswith("-c1") or run_id.startswith("hf-c1") else "N"
    if key == "prec":
        return "Y" if "-p1" in run_id else "N"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--baseline", default=None, help="run_id to use as baseline (default: hf-c0-p0 if present)")
    ap.add_argument("--filter-model", default=None)
    ap.add_argument("--filter-dataset", default=None)
    ap.add_argument("--filter-sdpa", default=None)
    ap.add_argument("--sort", default="avg_ms", choices=["avg_ms","samp_s","tok_s","cuda_peak","eval_loss","ppl"])
    ap.add_argument("--desc", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = filter_rows(load_jsonl(args.jsonl), args.filter_model, args.filter_dataset, args.filter_sdpa)
    if not rows:
        print("No rows after filtering.")
        sys.exit(0)

    base = pick_baseline(rows, args.baseline)

    # sorting
    rows.sort(key=lambda r: r.get(args.sort, float("inf")), reverse=args.desc)

    # header
    print("run_id         ckpt prec sdpa    avg_ms   samp/s     tok/s      eval_loss   ppl       cuda_peak    Δmem%   speedup")
    print("-" * 112)

    def rels(r):
        if not base:
            return ("", "")
        try:
            mem0 = float(base.get("cuda_peak", 0) or 0)
            mem1 = float(r.get("cuda_peak", 0) or 0)
            dmem = (mem0 - mem1) / mem0 if mem0 > 0 else 0.0
        except Exception:
            dmem = 0.0
        try:
            t0 = float(base.get("avg_ms", 0) or 0)
            t1 = float(r.get("avg_ms", 0) or 0)
            speed = (t0 / t1) if (t0 > 0 and t1 > 0) else 0.0
        except Exception:
            speed = 0.0
        return (f"{dmem*100:6.2f}%", f"{speed:7.3f}x")

    for r in rows:
        ck = as_bool_str(r.get("run_id",""), "ckpt")
        pr = as_bool_str(r.get("run_id",""), "prec")
        dmem, speed = rels(r)
        print(
            f"{r.get('run_id',''):<14} {ck:<4} {pr:<4} {str(r.get('sdpa','')):<7} "
            f"{float(r.get('avg_ms',float('nan'))):7.2f} "
            f"{float(r.get('samp_s',float('nan'))):8.2f} "
            f"{float(r.get('tok_s',float('nan'))):10.2f} "
            f"{float(r.get('eval_loss',float('nan'))):10.4f} "
            f"{float(r.get('ppl',float('nan'))):9.2f} "
            f"{fmt_bytes(int(r.get('cuda_peak',0))):>11}  "
            f"{dmem:>6} {speed:>9}"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([c[0] for c in COLS] + ["dmem_pct","speedup_vs_base"])
            for r in rows:
                dmem, speed = rels(r)
                w.writerow([
                    r.get("run_id",""), as_bool_str(r.get("run_id",""),"ckpt"), as_bool_str(r.get("run_id",""),"prec"),
                    r.get("sdpa",""), r.get("avg_ms",""), r.get("samp_s",""), r.get("tok_s",""),
                    r.get("eval_loss",""), r.get("ppl",""), r.get("cuda_peak",""),
                    dmem, speed
                ])

if __name__ == "__main__":
    main()
