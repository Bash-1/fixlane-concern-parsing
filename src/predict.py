"""
Generate predictions with the base model (optionally + a LoRA adapter).

Usage:
  python src/predict.py --adapter adapters/qwen3b_clean_dev/checkpoint-28 --tag ft_e1 --splits val_seen,val_novel
  python src/predict.py --adapter none --tag base_zeroshot --splits val_seen,val_novel
Writes outputs/preds/{tag}__{split}.jsonl in the format evaluate.py expects.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from common import load_jsonl
from train import prompt_messages, end_token

SPLIT_PATHS = {"val_seen": "data/processed/val_seen.jsonl",
               "val_novel": "data/processed/val_novel.jsonl",
               "test": "data/raw/test.jsonl"}


def load(base_model, adapter):
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"                      # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                 dtype=torch.float16, device_map={"": 0})
    if adapter and adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@torch.inference_mode()
def generate(tok, model, texts, batch_size=16, max_new_tokens=128):
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                               eos_token_id=[tok.convert_tokens_to_ids(end_token(tok)), tok.eos_token_id],
                               pad_token_id=tok.pad_token_id)
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        prompts = [tok.apply_chat_template(prompt_messages(t), tokenize=False, add_generation_prompt=True)
                   for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.cuda.synchronize(); t0 = time.time()
        gen = model.generate(**enc, generation_config=gen_cfg)
        torch.cuda.synchronize(); dt = time.time() - t0
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            n_tok = int((row != tok.pad_token_id).sum())
            outs.append({"raw": tok.decode(row, skip_special_tokens=True).strip(),
                         "latency_s": round(dt / len(chunk), 4),       # amortized per row in the batch
                         "new_tokens": n_tok})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", default="val_seen,val_novel")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    a = ap.parse_args()

    tok, model = load(a.base_model, a.adapter)
    for split in a.splits.split(","):
        rows = load_jsonl(SPLIT_PATHS[split])
        t0 = time.time()
        outs = generate(tok, model, [r["input"] for r in rows], a.batch_size, a.max_new_tokens)
        out_path = Path(f"outputs/preds/{a.tag}__{split}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for i, o in enumerate(outs):
                f.write(json.dumps({"row_id": i, **o, "adapter": a.adapter}) + "\n")
        print(f"[{a.tag}] {split}: {len(outs)} rows in {time.time() - t0:.0f}s -> {out_path}")
        print(f"   sample: {outs[0]['raw'][:160]}")


if __name__ == "__main__":
    main()
