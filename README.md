# DPCS: Dynamic Precision & Checkpointing Scheduler

## Project overview
Dynamic Precision & Checkpointing Scheduler (DPCS) is a lightweight control
plane you can drop into an existing PyTorch training loop to automatically
balance GPU memory headroom against throughput. It keeps track of the signals
that matter for mixed precision and activation checkpointing, picks the right
mode for the current step, and keeps every device in a distributed job in sync.
DPCS focuses on three complementary capabilities:

* **Dynamic AMP orchestration.** The scheduler can demote from fp32 into bf16 or
  fp16 when gradients look stable, and promote back to full precision when AMP
  overflows occur. When NVIDIA TransformerEngine is available it can also push
  eligible linear layers into fp8 and gradually tune the FP8 scaling margin.
* **Selective activation checkpointing.** DPCS wraps parameter leaves so it can
  toggle checkpointing on the heaviest activations when VRAM headroom is tight
  and release them when memory pressure subsides.
* **Fast signal collection.** Gradients, curvature probes, activation sizes, and
  allocator statistics are sampled with tiny hooks and cached tensors so the
  training step stays bandwidth-bound rather than Python-bound.

The end result is a single object that you instantiate once, wrap your model
with, and call at predictable places in your loop (`start_step → forward/backward
→ collect_signals → optimizer.step → end_step`). Each call updates an internal
state machine that enforces hysteresis so precision modes do not thrash.

## Objective
DPCS aims to keep training stable while maximizing throughput under available
memory headroom. It does that by promoting precision only when signals indicate
numerical risk, and by adding checkpointing only when allocator pressure is
meaningful enough to warrant the extra recompute.

## Signals and precision transitions (high level)
* **Gradient variance** acts as a stability proxy: low, steady variance allows
  safe demotion into bf16/fp16/fp8, while spikes push the policy to stay in or
  return to fp32.
* **Curvature estimates** (top Hessian eigenvalue probes) capture sensitivity;
  higher curvature tightens the precision budget and can trigger promotions even
  if variance is otherwise stable.
* **Memory headroom** gates aggressive modes: if headroom is tight, the policy
  avoids precision promotion and leans on checkpointing/FP8 only when there is
  room for the extra metadata and scaling overhead.

## How the scheduler works
1. **Model wrapping.** `DPCS.wrap(model)` recursively replaces every leaf module
   (linears and convolutions by default, plus any module with no children) with
   a `_CkptWrap`. The wrapper records activation sizes, exposes hooks for FP8,
   and can route the forward pass through `torch.utils.checkpoint` with the
   configured options.
2. **Step lifecycle.** `start_step()` resets VRAM peak counters, decays the
   checkpoint blacklist, and toggles FP8 autocast contexts before the forward
   pass begins. After the backward pass, `collect_signals(loss)` optionally
   triggers curvature probes and refreshes cached gradient statistics. Finally
   `end_step(optimizer, scaler)` reads memory headroom, updates the precision
   policy, plans which leaves to checkpoint, broadcasts the decisions across the
   distributed job, logs JSONL telemetry, and advances the global step counter.
3. **Policies and signals.** Precision decisions come from a finite-state
   machine that prefers bf16 when available, falls back to fp16 otherwise, and
   escalates back to fp32 on overflow. When FP8 is available it uses the same
   headroom gates as activation checkpointing. The checkpoint policy ranks leaves
   by activation-byte EMAs (bytes saved) and a benefit score (bytes saved per
   recompute cost), then enables the best candidates within global and per-step
   pressure limits. Gradient variance EMAs, optional curvature power iterations,
   and allocator introspection feed those policies without allocating new tensors
   on the hot path.
4. **Distributed coordination.** If `torch.distributed` is initialized, DPCS
   encodes the chosen AMP mode and checkpoint mask into a tiny tensor and
   broadcasts it from rank 0 so that every replica applies identical decisions.

## Module reference (`src/dpcs`)
### `__init__.py`
Exports the public surface (`DPCS`, `DPCSConfig`, `CheckpointCfg`, `PrecisionCfg`)
so users can `from dpcs import DPCS` without thinking about the internal layout.
It exists solely as a convenience shim.

### `config.py`
Defines the frozen dataclasses that configure the scheduler:

* `DPCSConfig` mirrors all runtime knobs (precision bands, patience, curvature
  probe cadence, memory thresholds, logging rate, TransformerEngine defaults).
  `from_kwargs` filters unknown keys, clips fractions into `[0, 1]`, normalizes
  patience/period counters, and keeps the configuration immutable so it can be
  shared safely across threads or processes.
* `TelemetryCfg` enables optional forward-time and activation sampling.
* `CheckpointCfg` configures the checkpoint wrapper (whether to preserve RNG
  state, reentrant behaviour, and how much of the model is ever eligible).

### `dpcs.py`
Legacy compatibility module that re-exports the same API as `__init__` plus the
`AmpOverflowMonitor` helper. Older code that imported `dpcs.dpcs` continues to
work without modifications.

### `policies.py`
Pure-Python decision logic that stays free of PyTorch dependencies:

* `PrecisionPolicy` encapsulates the AMP mode state machine. It tracks a stable
  step counter, cooldown windows after overflow, and optional FP8 support. The
  scheduler feeds it gradient variance, curvature summaries, headroom estimates,
  and the overflow flag to obtain the next precision mode.
* `CheckpointPolicy` keeps a lazily decaying blacklist for leaves that misbehave
  under checkpointing and ranks candidates by activation-byte EMA. It applies a
  configurable patience window before switching plans so modules are not toggled
  on and off every step.

### `runtime.py`
Collects the runtime helpers the scheduler needs but keeps them separated from
slow imports:

* AMP utilities (`amp_preferred_dtype`, `amp_enabled`, `amp_uses_grad_scaler`).
* Scaled dot-product attention helpers that normalize backend selection and
  expose a context manager to force specific kernels when required.
* TransformerEngine integration for FP8 (`te_prepare_fp8_modules`,
  `te_fp8_autocast`) that swaps compatible `nn.Linear` modules and prepares the
  delayed-scaling recipe.
* Memory instrumentation wrappers around `torch.cuda.memory` and
  `cudaMemGetInfo` plus `headroom_frac()` which the policies use to judge
  pressure.
* `JsonlLogger` for buffered step telemetry writes and lightweight distributed
  helpers (`dist_is_initialized`, `dist_get_rank`, `dist_world_size`,
  `dist_broadcast`).
* `checkpoint_call`, `timed_cuda`, and other utilities that the wrappers rely on
  without pulling optional dependencies into every import.

### `scheduler.py`
The heart of DPCS. It wires together configuration, runtime helpers, signal
collectors, and policies into a single training-time object. Highlights include:

* `_CkptWrap`, the leaf wrapper that tracks activation sizes, owns the FP8/AMP
  contexts, and routes the forward pass through checkpointing on demand.
* `_LeafState` and `_StateRegistry`, lightweight views that expose per-leaf
  telemetry to legacy callers that expect dictionary-like access.
* Precision override APIs (`force_precision`, `force_fp32`, `clear_precision_override`)
  and helpers to query AMP configs for the current device.
* Automatic wrapping of eligible leaves, GradSignals/CurvatureSignals creation,
  FP8 module replacement, and caching needed for distributed broadcasts.
* Step orchestration (`start_step`, `collect_signals`, `end_step`) that applies
  the policies, updates TransformerEngine margins, logs JSONL records, and keeps
  the distributed tensor of decisions in sync with rank 0.

### `signals.py`
Implements the lightweight signal collection infrastructure:

* `EMA` and `tensor_bytes` utilities that keep statistics without allocating
  tensors on each call.
* `GradSignals` attaches tiny per-parameter hooks that downsample gradients,
  accumulate absolute-mean and variance EMAs per leaf, and expose simple getters
  for policies.
* `CurvatureSignals` performs scheduled Hessian–vector power iterations using a
  cached random vector per leaf and keeps the estimated top eigenvalue in a list
  aligned with the wrappers.
* `ActivationBytesEMA`, `CudaTimer`, and `ForwardTimer` collect optional
  telemetry for operators selected by a regex, tracking activation sizes and
  forward latency while respecting sampling rates from `TelemetryCfg`.

## Usage example
```python
from dpcs import DPCS

model = build_model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scaler = torch.cuda.amp.GradScaler()

scheduler = DPCS(device_type="cuda", log_every=10)
model = scheduler.wrap(model)

for step, batch in enumerate(loader):
    scheduler.start_step()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(*scheduler.get_amp_config()):
        loss = compute_loss(model, batch)

    scaler.scale(loss).backward()
    scheduler.collect_signals(loss, model)
    scaler.step(optimizer)
    scheduler.end_step(optimizer, scaler)
    scaler.update()
```

`get_amp_config()` advertises the preferred autocast tuple `(device_type,
dtype, enabled)` so runners can reuse it for custom mixed-precision regions.
`end_step` handles both precision and checkpointing so the main loop stays clean.

## Tests
Install the development extras and run the unit test suite with `pytest`:

```bash
pip install -e .[dev]
pytest
```

Some checks (for example `tests/test_fp8_precision.py`) require a CUDA-capable
GPU and optional dependencies such as `transformer-engine`. Use pytest markers
(e.g. `-k 'not fp8'`) if you need to skip those GPU-specific paths.

### Benchmark (static AMP + checkpointing)
Run the short DPCS micro-benchmark (defaults to 10 steps, GPU optional):

```bash
python bench/bench_static_dpcs.py
```

The script prints a JSON summary with:
* `steps_per_sec` for throughput (higher is better).
* `peak_memory_bytes` for peak allocator usage (0 on CPU).
* `overflow_count` for AMP overflow events (0 indicates stable scaling).
