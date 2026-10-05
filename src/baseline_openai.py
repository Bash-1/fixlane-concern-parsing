"""
Prompted baseline: OpenAI chat model + few-shot examples + structured outputs.

Modes:
  static     k fixed examples covering every vehicle_system and severity (same for every input)
  retrieval  k most similar training concerns per input (one copy per concern, most similar last)

Usage:
  python src/baseline_openai.py --eval data/processed/val_seen.jsonl \
      --train data/processed/train_clean_dev.jsonl --mode retrieval --k 8 \
      --out outputs/preds/gpt4o_retrieval__val_seen.jsonl
"""
from __future__ import annotations

import argparse, json, os, random, threading, time
import concurrent.futures as cf
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import (SYSTEMS, SEVERITIES, SYSTEM_PROMPT_BASELINE, OUTPUT_JSON_SCHEMA,
                    load_jsonl, format_target, core_text)

DEFAULT_MODEL = "gpt-4o-2024-11-20"
SAME_CONCERN_SIM = 0.6   # examples above this similarity to each other are copies of one concern


class FewShotSelector:
    def __init__(self, train):
        self.train = train
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.X = self.vec.fit_transform([core_text(r["input"]) for r in train])
        self.S = cosine_similarity(self.X)

    def _distinct(self, i, chosen):
        return all(self.S[i, j] < SAME_CONCERN_SIM for j in chosen)

    def static(self, k=12, seed=0):
        """Cover every system, then every severity, then fill with random distinct concerns."""
        idx = list(range(len(self.train)))
        random.Random(seed).shuffle(idx)
        chosen = []
        for key, values in [("vehicle_system", SYSTEMS), ("severity", SEVERITIES)]:
            for v in values:
                if any(self.train[j]["output"][key] == v for j in chosen):
                    continue
                for i in idx:
                    if self.train[i]["output"][key] == v and self._distinct(i, chosen):
                        chosen.append(i)
                        break
        for i in idx:
            if len(chosen) >= k:
                break
            if i not in chosen and self._distinct(i, chosen):
                chosen.append(i)
        return chosen[:k]

    def retrieve(self, text, k=8):
        """Top-k most similar distinct concerns, ordered least -> most similar (closest last)."""
        sims = cosine_similarity(self.vec.transform([core_text(text)]), self.X).ravel()
        chosen = []
        for j in sims.argsort()[::-1]:
            if self._distinct(j, chosen):
                chosen.append(int(j))
            if len(chosen) == k:
                break
        return chosen[::-1]


def build_messages(train, shot_ids, query):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_BASELINE}]
    for j in shot_ids:
        msgs.append({"role": "user", "content": train[j]["input"]})
        msgs.append({"role": "assistant", "content": format_target(train[j]["output"])})
    msgs.append({"role": "user", "content": query})
    return msgs


def run(eval_path, train_path, out_path, mode="retrieval", k=8, model=DEFAULT_MODEL, workers=3, seed=0):
    from openai import OpenAI
    client = OpenAI(max_retries=10)
    eval_rows, train = load_jsonl(eval_path), load_jsonl(train_path)
    sel = FewShotSelector(train)
    static_ids = sel.static(k, seed) if mode == "static" else None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["row_id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
    todo = [i for i in range(len(eval_rows)) if i not in done]
    print(f"[{mode}, k={k}, {model}] {len(todo)} to run, {len(done)} already done -> {out_path}")
    lock = threading.Lock()

    def call(i):
        shots = static_ids if mode == "static" else sel.retrieve(eval_rows[i]["input"], k)
        msgs = build_messages(train, shots, eval_rows[i]["input"])
        for attempt in range(5):
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0, seed=seed,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "concern", "strict": True, "schema": OUTPUT_JSON_SCHEMA}})
                latency = time.time() - t0
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        msg = resp.choices[0].message
        rec = {"row_id": i, "raw": msg.content or "", "refusal": getattr(msg, "refusal", None),
               "latency_s": round(latency, 3), "prompt_tokens": resp.usage.prompt_tokens,
               "completion_tokens": resp.usage.completion_tokens, "model": model, "mode": mode, "shots": shots}
        with lock, open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(call, todo))
    recs = load_jsonl(out_path)
    print(f"  done: {len(recs)} rows | median latency {sorted(r['latency_s'] for r in recs)[len(recs)//2]:.2f}s | "
          f"avg prompt tokens {sum(r['prompt_tokens'] for r in recs)/len(recs):.0f}")
    return static_ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["static", "retrieval"], default="retrieval")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    run(a.eval, a.train, a.out, a.mode, a.k, a.model, a.workers)
