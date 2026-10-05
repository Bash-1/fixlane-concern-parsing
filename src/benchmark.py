"""
Latency / throughput benchmark for the fine-tuned model.

  --mode 4bit    4-bit base + LoRA adapter (as trained)
  --mode merged  fp16 base with the adapter merged in (deployment option)

Usage:
  python src/benchmark.py --mode merged --adapter adapters/qwen3b_final/final --tag ft_final_merged
Writes outputs/preds/{tag}__test.jsonl and outputs/bench_{tag}.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import predict as pr
from common import load_jsonl


def load_merged(base_model, adapter):
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["4bit", "merged"], required=True)
    ap.add_argument("--adapter", default="adapters/qwen3b_final/final")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/raw/test.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_single", type=int, default=20)
    a = ap.parse_args()

    texts = [r["input"] for r in load_jsonl(a.data)]
    tok, model = pr.load(a.base_model, a.adapter) if a.mode == "4bit" else load_merged(a.base_model, a.adapter)
    torch.cuda.reset_peak_memory_stats()

    pr.generate(tok, model, texts[:2], batch_size=1)                       # warm-up
    single = []
    for x in texts[:a.n_single]:
        t0 = time.time(); pr.generate(tok, model, [x], batch_size=1); single.append(time.time() - t0)
    t0 = time.time()
    outs = pr.generate(tok, model, texts, batch_size=16)
    per_row = (time.time() - t0) / len(texts)

    with open(f"outputs/preds/{a.tag}__test.jsonl", "w") as f:
        for i, o in enumerate(outs):
            f.write(json.dumps({"row_id": i, **o}) + "\n")
    res = {"setup": f"FT {a.mode}", "p50_single_s": round(float(np.median(single)), 2),
           "p90_single_s": round(float(np.percentile(single, 90)), 2),
           "batched_s_per_row": round(per_row, 3),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    Path(f"outputs/bench_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
