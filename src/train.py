"""
QLoRA fine-tuning of a small instruct model for Fixlane concern parsing.

Loss is computed on the assistant JSON only (prompt tokens masked with -100).
Options:
  --cat_weight W     weight the tokens of the vehicle_system / severity VALUES W times in the loss
  --oversample C:K   repeat training rows whose severity == C, K times in total (train set only)

Usage:
  python src/train.py --train data/processed/train_clean_dev.jsonl --output_dir adapters/qwen3b_clean_dev
  python src/train.py ... --cat_weight 5
  python src/train.py ... --oversample safety_critical:3
  python src/train.py ... --dry_run
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, set_seed)

from common import SYSTEM_PROMPT_FT, format_target, load_jsonl

CAT_FIELDS = ["vehicle_system", "severity"]


def prompt_messages(text):
    return [{"role": "system", "content": SYSTEM_PROMPT_FT}, {"role": "user", "content": text}]


def end_token(tok):
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def value_spans(target, output):
    """Character spans of the categorical VALUES inside the target JSON string."""
    spans = []
    for f in CAT_FIELDS:
        prefix = f'"{f}": "'
        i = target.find(prefix + output[f] + '"')
        if i >= 0:
            s = i + len(prefix)
            spans.append((s, s + len(output[f])))
    return spans


def encode(row, tok, max_len, cat_weight=1.0):
    prompt = tok.apply_chat_template(prompt_messages(row["input"]), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target = format_target(row["output"]) + end_token(tok)
    t = tok(target, add_special_tokens=False, return_offsets_mapping=True)
    spans = value_spans(target, row["output"])
    t_w = [cat_weight if any(a < e and b > s for s, e in spans) else 1.0 for a, b in t["offset_mapping"]]
    ids = p_ids + t["input_ids"]
    enc = {"input_ids": ids[:max_len], "attention_mask": [1] * min(len(ids), max_len),
           "labels": ([-100] * len(p_ids) + t["input_ids"])[:max_len], "truncated": len(ids) > max_len}
    if cat_weight != 1.0:
        enc["loss_weights"] = ([0.0] * len(p_ids) + t_w)[:max_len]
    return enc


def build_dataset(path, tok, max_len, cat_weight=1.0, oversample=None):
    rows = load_jsonl(path)
    if oversample:
        cls, k = oversample.split(":")
        extra = [r for r in rows if r["output"]["severity"] == cls] * (int(k) - 1)
        rows = rows + extra
        print(f"  oversample: +{len(extra)} rows of '{cls}' ({k}x total)")
    enc = [encode(r, tok, max_len, cat_weight) for r in rows]
    n_trunc = sum(e.pop("truncated") for e in enc)
    lens = [len(e["input_ids"]) for e in enc]
    print(f"  {path}: {len(enc)} rows | tokens min/mean/max {min(lens)}/{sum(lens)/len(lens):.0f}/{max(lens)} | truncated {n_trunc}")
    if n_trunc:
        raise ValueError(f"{n_trunc} examples exceed max_len={max_len}; raise --max_len")
    return Dataset.from_list(enc)


def make_collator(pad_id):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        L = (L + 7) // 8 * 8
        pad = lambda k, v: [b[k] + [v] * (L - len(b[k])) for b in batch]
        out = {"input_ids": torch.tensor(pad("input_ids", pad_id)),
               "attention_mask": torch.tensor(pad("attention_mask", 0)),
               "labels": torch.tensor(pad("labels", -100))}
        if "loss_weights" in batch[0]:
            out["loss_weights"] = torch.tensor(pad("loss_weights", 0.0), dtype=torch.float32)
        return out
    return collate


class WeightedLossTrainer(Trainer):
    """Token-weighted cross-entropy: mean over target tokens, weighted by loss_weights."""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        w = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits[:, :-1].float()
        tgt, ww = labels[:, 1:], w[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                             ignore_index=-100, reduction="none").view(tgt.shape)
        mask = (tgt != -100).float()
        loss = (ce * ww * mask).sum() / (ww * mask).sum().clamp(min=1e-8)
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval_seen", default="data/processed/val_seen.jsonl")
    ap.add_argument("--eval_novel", default="data/processed/val_novel.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--evals_per_epoch", type=int, default=2)
    ap.add_argument("--cat_weight", type=float, default=1.0)
    ap.add_argument("--oversample", default=None, help="e.g. safety_critical:3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Tokenizing:")
    train_ds = build_dataset(args.train, tok, args.max_len, args.cat_weight, args.oversample)
    eval_ds = {"seen": build_dataset(args.eval_seen, tok, args.max_len, args.cat_weight),
               "novel": build_dataset(args.eval_novel, tok, args.max_len, args.cat_weight)}

    ex = train_ds[0]
    trained = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    print("\n--- Example: tokens that receive loss ---\n" + tok.decode(trained))
    if "loss_weights" in ex:
        up = [t for t, w in zip(ex["input_ids"], ex["loss_weights"]) if w > 1]
        print(f"--- Up-weighted ({args.cat_weight}x) tokens: {[tok.decode([t]) for t in up]} ---")
    print(f"--- {len(trained)} of {len(ex['input_ids'])} tokens trained ---\n")
    if args.dry_run:
        return

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=args.target_modules.split(","), task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    eval_steps = max(1, steps_per_epoch // args.evals_per_epoch)
    print(f"steps/epoch={steps_per_epoch} total={total_steps} eval_every={eval_steps} save_every={steps_per_epoch}")

    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=round(args.warmup_frac * total_steps), weight_decay=args.weight_decay,
        optim="paged_adamw_8bit", fp16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=steps_per_epoch,
        logging_steps=2, report_to="none",
        seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
    )
    TrainerCls = WeightedLossTrainer if args.cat_weight != 1.0 else Trainer
    trainer = TrainerCls(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                         data_collator=make_collator(tok.pad_token_id), processing_class=tok)

    print("Eval before training:", {k: round(v, 4) for k, v in trainer.evaluate().items() if "loss" in k})
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    (out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    meta = {**vars(args), "n_train": len(train_ds), "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps, "train_minutes": round(minutes, 2),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (out / "train_config.json").write_text(json.dumps(meta, indent=1))
    print(f"\nDone in {minutes:.1f} min | peak GPU {meta['peak_gpu_gb']} GB | adapter -> {out / 'final'}")


if __name__ == "__main__":
    main()
