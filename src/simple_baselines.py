"""
Non-LLM reference baselines.

  majority  always predict the most common training vehicle_system / severity (+ placeholder text)
  knn       copy all four fields from the most similar training row
            (char 3-5gram TF-IDF on filler-stripped text, vectorizer fit on TRAIN inputs only)

Usage:
  python src/simple_baselines.py --train data/processed/train_clean_full.jsonl --split test
Writes {out_dir}/knn__{split}.jsonl and {out_dir}/majority__{split}.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import core_text, load_jsonl

SPLIT_PATHS = {"val_seen": "data/processed/val_seen.jsonl",
               "val_novel": "data/processed/val_novel.jsonl",
               "test": "data/raw/test.jsonl"}


def write_preds(outputs, path):
    with open(path, "w") as f:
        for i, o in enumerate(outputs):
            f.write(json.dumps({"row_id": i, "raw": json.dumps(o)}) + "\n")
    print(f"  wrote {len(outputs)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--split", required=True, choices=list(SPLIT_PATHS))
    ap.add_argument("--out_dir", default="outputs/preds")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    train, rows = load_jsonl(a.train), load_jsonl(SPLIT_PATHS[a.split])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([core_text(r["input"]) for r in train])
    sims = cosine_similarity(vec.transform([core_text(r["input"]) for r in rows]),
                             vec.transform([core_text(r["input"]) for r in train]))
    write_preds([train[j]["output"] for j in sims.argmax(1)], f"{a.out_dir}/knn__{a.split}.jsonl")

    maj = {"vehicle_system": pd.Series([r["output"]["vehicle_system"] for r in train]).mode()[0],
           "severity": pd.Series([r["output"]["severity"] for r in train]).mode()[0],
           "primary_symptom": "vehicle issue", "suggested_diagnostic": "check vehicle systems"}
    write_preds([maj] * len(rows), f"{a.out_dir}/majority__{a.split}.jsonl")


if __name__ == "__main__":
    main()
