"""
Rebuild all processed data from the raw Fixlane files (deterministic).

Steps
  1. Cluster training rows into concerns (char 3-5gram TF-IDF on filler-stripped text, cosine >= 0.60).
  2. Clean train labels:
       - vehicle_system / severity: minority labels in a cluster (>=3 rows, clear majority) -> majority label
       - free text: placeholder values ("vehicle issue", "check vehicle systems") -> cluster majority value,
         or nearest neighbour's value for rows in clusters too small for a vote
  3. Hold out 10 whole concerns as val_novel (one per vehicle_system + 2 extra; >=3 comfort and >=2
     safety_critical concerns; cosmetic concerns excluded because they are too rare to spare).
  4. Tag every val/test row as seen/novel relative to the training concerns, with a concern-level
     group id for cluster bootstrap CIs.

Note: the TF-IDF vocabulary/IDF is fit on the INPUT TEXT of all splits (no labels are used).

Usage:  python src/prepare_data.py [--raw_dir data/raw] [--out_dir data/processed]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import core_text, load_jsonl

CLUSTER_THR = 0.60          # same-concern similarity threshold
SEEN_THR = 0.50             # val/test row counts as "seen" if its nearest train row is at least this similar
MIN_CLUSTER = 3             # minimum copies for a majority vote
PLACEHOLDERS = {"primary_symptom": {"vehicle issue"}, "suggested_diagnostic": {"check vehicle systems"}}
CATEGORICAL = ["vehicle_system", "severity"]
FREETEXT = ["primary_symptom", "suggested_diagnostic"]
OUT_FIELDS = ["vehicle_system", "primary_symptom", "severity", "suggested_diagnostic"]
HOLDOUT_TARGET = {"comfort": 3, "safety_critical": 2}


def write_jsonl(frame, path):
    with open(path, "w") as f:
        for _, r in frame.iterrows():
            f.write(json.dumps({"input": r.input, "output": {k: r[k] for k in OUT_FIELDS}}) + "\n")
    print(f"  wrote {len(frame):>3} rows -> {path}")


def load_frame(raw_dir):
    parts = []
    for s in ["train", "val", "test"]:
        rows = load_jsonl(f"{raw_dir}/{s}.jsonl")
        parts.append(pd.DataFrame([{"split": s, "input": r["input"], **r["output"]} for r in rows]))
    df = pd.concat(parts, ignore_index=True)
    df["row_id"] = df.groupby("split").cumcount()
    df["core"] = df.input.apply(core_text)
    return df


def clean_labels(tr, vec):
    clean, log = tr.copy(), []

    def record(idx, field, new, reason):
        log.append({"row_id": int(clean.at[idx, "row_id"]), "cluster": int(clean.at[idx, "cluster"]),
                    "field": field, "old": clean.at[idx, field], "new": new, "reason": reason,
                    "input": clean.at[idx, "input"]})
        clean.at[idx, field] = new

    # (a) majority vote inside clusters
    for cl, sub in clean.groupby("cluster"):
        for f in CATEGORICAL + FREETEXT:
            is_ph = sub[f].isin(PLACEHOLDERS.get(f, set()))
            counts = sub.loc[~is_ph, f].value_counts()
            if counts.empty:
                continue
            top, top_n = counts.index[0], counts.iloc[0]
            clear = len(sub) >= MIN_CLUSTER and top_n > len(sub) / 2 and (len(counts) == 1 or counts.iloc[1] < top_n)
            if f in FREETEXT:
                for idx in sub.index[is_ph]:
                    if clear:
                        record(idx, f, top, "placeholder")
            else:
                for idx in sub.index[sub[f] != top]:
                    if clear:
                        record(idx, f, top, "minority_label")
                    else:
                        log.append({"row_id": int(clean.at[idx, "row_id"]), "cluster": int(cl), "field": f,
                                    "old": clean.at[idx, f], "new": None, "reason": "no_clear_majority",
                                    "input": clean.at[idx, "input"]})

    # (b) placeholders in clusters too small to vote -> nearest neighbour outside the cluster
    Xc = vec.transform(clean.core)
    sizes = clean.cluster.value_counts()
    all_ph = set().union(*PLACEHOLDERS.values())
    for i, r in clean[clean.cluster.map(sizes) < MIN_CLUSTER].iterrows():
        sims = cosine_similarity(Xc[i], Xc).ravel()
        sims[(clean.cluster == r.cluster).values] = -1
        nb = clean.loc[int(sims.argmax())]
        for f in FREETEXT:
            if r[f] in all_ph and nb[f] not in all_ph:
                record(i, f, nb[f], "placeholder_nearest_neighbor")
    return clean, pd.DataFrame(log)


def pick_holdout(clean):
    info = clean.groupby("cluster").agg(system=("vehicle_system", "first"), severity=("severity", "first"),
                                        n=("input", "size")).reset_index()
    pool = info[(info.n >= MIN_CLUSTER) & (info.severity != "cosmetic")]
    idx = pool.set_index("cluster")
    for seed in range(5000):
        rng = np.random.default_rng(seed)
        h = [int(rng.choice(g.cluster.values)) for _, g in pool.groupby("system")]
        rest = pool[~pool.cluster.isin(h)]
        h += [int(c) for c in rng.choice(rest.cluster.values, size=2, replace=False)]
        sev = idx.loc[h, "severity"].value_counts()
        if all(sev.get(k, 0) >= v for k, v in HOLDOUT_TARGET.items()):
            print(f"  holdout seed {seed}: clusters {sorted(h)}")
            return h
    raise RuntimeError("no valid holdout found")


def build_slices(df, tr, vec, clean, held):
    X_tr = vec.transform(tr.core)
    meta = []
    for s in ["val", "test"]:
        d = df[df.split == s].reset_index(drop=True)
        sim = cosine_similarity(vec.transform(d.core), X_tr)
        best, nn = sim.max(1), sim.argmax(1)
        for i in range(len(d)):
            meta.append({"split": s, "row_id": i, "nn_sim": round(float(best[i]), 3),
                         "nearest_train_cluster": int(tr.cluster[nn[i]]),
                         "slice": "seen" if best[i] >= SEEN_THR else "novel"})
    sl = pd.DataFrame(meta)
    sl["split"] = sl.split.replace({"val": "val_seen"})
    sl["group"] = "c" + sl.nearest_train_cluster.astype(str)
    for s, src in [("val_seen", "val"), ("test", "test")]:       # novel rows: group among themselves
        m = (sl.split == s) & (sl.slice == "novel")
        if m.sum():
            texts = df[df.split == src].set_index("row_id").loc[sl[m].row_id, "core"]
            _, lab = connected_components(cosine_similarity(vec.transform(texts)) >= CLUSTER_THR, directed=False)
            sl.loc[m, "group"] = [f"novel_{s}_{k}" for k in lab]

    held_rows = clean[clean.cluster.isin(held)].reset_index(drop=True)
    vn = pd.DataFrame({"split": "val_novel", "row_id": range(len(held_rows)), "nn_sim": np.nan,
                       "nearest_train_cluster": held_rows.cluster.values, "slice": "novel",
                       "group": "c" + held_rows.cluster.astype(str).values})
    # val_seen re-tagged relative to the DEV training set (held-out concerns are novel there)
    vs = sl[sl.split == "val_seen"].copy()
    vs["split"] = "val_seen_dev"
    vs["group"] = "c" + vs.nearest_train_cluster.astype(str)
    vs["slice"] = np.where(vs.group.isin({f"c{c}" for c in held}), "novel", "seen")
    return pd.concat([sl, vn, vs], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--log_out", default="outputs/cleaning_log.csv")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_frame(a.raw_dir)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(df.core)
    tr = df[df.split == "train"].reset_index(drop=True)
    S = cosine_similarity(vec.transform(tr.core))
    np.fill_diagonal(S, 0)
    n, tr["cluster"] = connected_components(S >= CLUSTER_THR, directed=False)
    print(f"Train: {len(tr)} rows -> {n} concerns")

    clean, log = clean_labels(tr, vec)
    print(f"Cleaning: {log.new.notna().sum()} label changes on {log[log.new.notna()].row_id.nunique()} rows")
    held = pick_holdout(clean)
    is_held = clean.cluster.isin(held)

    print("Writing:")
    write_jsonl(clean, out / "train_clean_full.jsonl")
    write_jsonl(tr, out / "train_raw_full.jsonl")
    write_jsonl(clean[~is_held], out / "train_clean_dev.jsonl")
    write_jsonl(tr[~is_held], out / "train_raw_dev.jsonl")
    write_jsonl(clean[is_held], out / "val_novel.jsonl")
    write_jsonl(df[df.split == "val"], out / "val_seen.jsonl")
    tr[["row_id", "cluster", "core"]].to_csv(out / "train_clusters.csv", index=False)
    pd.Series(held, name="cluster").to_csv(out / "val_novel_clusters.csv", index=False)
    build_slices(df, tr, vec, clean, held).to_csv(out / "eval_slices.csv", index=False)
    Path(a.log_out).parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(a.log_out, index=False)
    print(f"  cleaning log -> {a.log_out}")


if __name__ == "__main__":
    main()
