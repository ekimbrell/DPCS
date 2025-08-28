import torch, math
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from dpcs import DPCS

MODEL_ID = "gpt2"  # try "EleutherAI/pythia-410m" if your GPU has room
DATA_ID  = "Salesforce/wikitext"  # subset "wikitext-2-raw-v1"

# ---- data
ds = load_dataset(DATA_ID, "wikitext-2-raw-v1")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tok.pad_token = tok.eos_token
def tok_fn(ex): return tok(ex["text"], truncation=True, max_length=512)
ds = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
collate = DataCollatorForLanguageModeling(tok, mlm=False)

# ---- model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
# enable gradient checkpointing in the model (HF side)
model.gradient_checkpointing_enable()  # pairs well with your DPCS ckpt policy. :contentReference[oaicite:0]{index=0}

# ---- DPCS wrap
sched = DPCS(
    device_type="cuda",
    enable_precision=True,
    autotune_precision=True,
    autotune_warmup_steps=50,
    wrap_types=(torch.nn.Linear, torch.nn.TransformerEncoderLayer),
    force_sdpa_in_blocks=True,
    sdpa_backends=(SDPBackend.MATH,),  # stable backend for apples-to-apples. :contentReference[oaicite:1]{index=1}
)
model = sched.wrap(model)  # local autocast + (optional) checkpoint policy

# ---- simple callback to tick the scheduler each step
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
class DPCSCallback(TrainerCallback):
    def on_step_begin(self, args:TrainingArguments, state:TrainerState, control:TrainerControl, **kw):
        sched._ckpt_on = args.gradient_checkpointing
        sched.start_step()
    def on_backward_end(self, args, state, control, **kw):
        # collect signals after .backward() so grads exist
        loss = kw["loss"] if "loss" in kw else None
        sched.collect_signals(loss, model)
    def on_optimizer_step(self, args, state, control, **kw):
        # if using GradScaler, pass it here; Trainer handles AMP internally
        sched.end_step(kw.get("optimizer"), scaler=None)

# ---- training
args = TrainingArguments(
    output_dir="out-gpt2-dpcs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    fp16=True,  # pairs with your precision scheduler
    gradient_checkpointing=True,  # grid off/on to compare
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=0,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"].select(range(5000)),   # small slice to keep runs quick
    eval_dataset=ds["validation"].select(range(1000)),
    data_collator=collate,
    callbacks=[DPCSCallback()],
)

# force math SDPA globally during the run for consistency (portable API) :contentReference[oaicite:2]{index=2}
with sdpa_kernel(SDPBackend.MATH):
    trainer.train()
    m = trainer.evaluate()
    print({"eval_loss": m["eval_loss"], "ppl": math.exp(m["eval_loss"])})
