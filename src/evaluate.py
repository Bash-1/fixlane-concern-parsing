"""
Evaluation for the Fixlane concern-parsing task.

Prediction files are JSONL, one line per gold row:
    {"row_id": 0, "raw": "<raw model text>", "latency_s": 0.41}

CLI:
    python src/evaluate.py --gold data/processed/val_seen.jsonl \
        --pred outputs/preds/gpt4o__val_seen.jsonl --split val_seen --judge
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, threading, time
import concurrent.futures as cf
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

SYSTEMS = ["engine", "electrical", "brakes", "suspension", "hvac", "drivetrain", "body", "other"]
SEVERITIES = ["cosmetic", "comfort", "drivability", "safety_critical"]  # least -> most severe
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
FIELDS = ["vehicle_system", "primary_symptom", "severity", "suggested_diagnostic"]

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5-2026-04-23")
JUDGE_CACHE = Path(os.environ.get("JUDGE_CACHE", "outputs/judge_cache.json"))


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- 1. parsing
def parse_output(raw):
    """Returns (obj or None, strict_json, parsed, schema_valid).
    strict_json: the whole text is valid JSON. parsed: a JSON object could be extracted
    (tolerates code fences / chatter). schema_valid: exact keys + allowed enum values."""
    if not isinstance(raw, str):
        return None, False, False, False
    text, obj, strict = raw.strip(), None, False
    try:
        obj, strict = json.loads(text), True
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return None, False, False, False
    diag = obj.get("suggested_diagnostic", "")
    schema = (set(obj) == set(FIELDS)
              and obj.get("vehicle_system") in SYSTEMS
              and obj.get("severity") in SEVERITIES
              and isinstance(obj.get("primary_symptom"), str) and bool(obj["primary_symptom"].strip())
              and (diag is None or (isinstance(diag, str) and bool(diag.strip()))))
    return obj, strict, True, bool(schema)


# ------------------------------------------------------ 2. text similarity
_embedder = None

def _embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

def text_similarity(gold, pred, present):
    """Cosine similarity per row. Both null (and the key present) -> 1.0; one side missing -> 0.0."""
    out = np.zeros(len(gold))
    idx = []
    for i, (g, p, ok) in enumerate(zip(gold, pred, present)):
        if g is None and p is None and ok:
            out[i] = 1.0
        elif isinstance(g, str) and isinstance(p, str) and p.strip():
            idx.append(i)
    if idx:
        a = _embed([gold[i] for i in idx])
        b = _embed([pred[i] for i in idx])
        out[idx] = (a * b).sum(1)
    return out


# ------------------------------------------------------------ 3. LLM judge
JUDGE_SYSTEM = """You grade a system that parses vehicle repair concerns written by technicians and service advisors (many vehicles are EVs).
For ONE concern you see the reference (gold) labels and a model's predicted labels. Grade two free-text fields.

primary_symptom (short noun phrase for the main symptom):
  2 = same symptom as the reference, including key context (location, condition, when it happens). Wording may differ.
  1 = right general symptom but missing important context, adding wrong detail, or too vague.
  0 = wrong symptom, a different issue, a generic placeholder, or missing.

suggested_diagnostic (short imperative first diagnostic step, or null):
  2 = same diagnostic direction as the reference, OR an equally appropriate first step a skilled technician would take for this concern.
  1 = related and plausible but clearly weaker: less specific, or targets a secondary cause.
  0 = wrong direction, irrelevant, unsafe, or missing when the reference has one. If both are null, give 2.

Judge meaning, not wording or length. Be strict and consistent."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "symptom_score": {"type": "integer", "enum": [0, 1, 2]},
        "diagnostic_score": {"type": "integer", "enum": [0, 1, 2]},
        "rationale": {"type": "string"},
    },
    "required": ["symptom_score", "diagnostic_score", "rationale"],
    "additionalProperties": False,
}

_lock = threading.Lock()

def judge_items(items, max_workers=8):
    """items: dicts with input, gold_symptom, gold_diag, pred_symptom, pred_diag. Cached on disk."""
    from openai import OpenAI
    client = OpenAI(max_retries=10)
    cache = json.loads(JUDGE_CACHE.read_text()) if JUDGE_CACHE.exists() else {}

    def call(it):
        key = hashlib.sha1(json.dumps([JUDGE_MODEL, JUDGE_SYSTEM, it], sort_keys=True).encode()).hexdigest()
        if key in cache:
            return cache[key]
        user = (f"Concern: {it['input']}\n\nReference:\n"
                f"  primary_symptom: {json.dumps(it['gold_symptom'])}\n"
                f"  suggested_diagnostic: {json.dumps(it['gold_diag'])}\n\nPrediction:\n"
                f"  primary_symptom: {json.dumps(it['pred_symptom'])}\n"
                f"  suggested_diagnostic: {json.dumps(it['pred_diag'])}")
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "grade", "strict": True, "schema": JUDGE_SCHEMA}})
                res = json.loads(resp.choices[0].message.content)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        with _lock:
            cache[key] = res
        return res

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(call, items))
    JUDGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_CACHE.write_text(json.dumps(cache))
    return results


# ------------------------------------------------------------- 4. bootstrap
def bootstrap_ci(values, groups=None, n_boot=2000, seed=0):
    """Mean with a 95% CI. Resamples whole groups (concerns), not rows, because rows that are
    rewordings of the same concern are not independent."""
    v = np.asarray(values, dtype=float)
    groups = np.arange(len(v)) if groups is None else np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats[b] = v[np.concatenate([members[k] for k in pick])].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)

def paired_diff(rows_a, rows_b, col, n_boot=2000):
    """Mean difference (A - B) of a per-row metric with a cluster-bootstrap 95% CI."""
    m = rows_a[["row_id", "group", col]].merge(rows_b[["row_id", col]], on="row_id", suffixes=("_a", "_b"))
    return bootstrap_ci(m[f"{col}_a"].astype(float) - m[f"{col}_b"].astype(float), m.group, n_boot)


# ------------------------------------------------------------ 5. evaluate
def _clean_label(x, labels):
    return x if isinstance(x, str) and x in labels else "__invalid__"

def macro_f1(gold, pred, labels):
    present = [l for l in labels if l in set(gold)]
    return float(f1_score(list(gold), [_clean_label(p, labels) for p in pred],
                          labels=present, average="macro", zero_division=0))

def evaluate(gold_path, pred_path, split=None, slices_path="data/processed/eval_slices.csv",
             judge=False, max_workers=8):
    gold = load_jsonl(gold_path)
    preds = {p["row_id"]: p for p in load_jsonl(pred_path)}
    missing = [i for i in range(len(gold)) if i not in preds]
    if missing:
        raise ValueError(f"{len(missing)} gold rows have no prediction, e.g. {missing[:5]}")

    rows = []
    for i, g in enumerate(gold):
        raw = preds[i].get("raw")
        obj, strict, parsed, schema = parse_output(raw)
        obj = obj or {}
        go = g["output"]
        r = {"row_id": i, "input": g["input"], "raw": raw,
             "strict_json": strict, "parsed": parsed, "schema_valid": schema,
             "latency_s": preds[i].get("latency_s", np.nan)}
        for f in FIELDS:
            r[f"gold_{f}"], r[f"pred_{f}"], r[f"has_{f}"] = go[f], obj.get(f), f in obj
        ps, ss = obj.get("severity"), obj.get("vehicle_system")
        r["sys_correct"] = isinstance(ss, str) and ss == go["vehicle_system"]
        r["sev_correct"] = isinstance(ps, str) and ps == go["severity"]
        r["sev_diff"] = (SEV_RANK[ps] - SEV_RANK[go["severity"]]) if isinstance(ps, str) and ps in SEV_RANK else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)

    df["sym_sim"] = text_similarity(df.gold_primary_symptom.tolist(), df.pred_primary_symptom.tolist(),
                                    df.has_primary_symptom.tolist())
    df["diag_sim"] = text_similarity(df.gold_suggested_diagnostic.tolist(), df.pred_suggested_diagnostic.tolist(),
                                     df.has_suggested_diagnostic.tolist())

    if judge:
        df["sym_judge"], df["diag_judge"], df["judge_rationale"] = 0.0, 0.0, ""
        def _same(a, b):
            return (a is None and b is None) or (isinstance(a, str) and isinstance(b, str)
                                                 and a.strip().lower() == b.strip().lower())
        exact = df.parsed & np.array([_same(g, p) and _same(gd, pdg) for g, p, gd, pdg in zip(
            df.gold_primary_symptom, df.pred_primary_symptom,
            df.gold_suggested_diagnostic, df.pred_suggested_diagnostic)], dtype=bool)
        df.loc[exact, ["sym_judge", "diag_judge"]] = 1.0
        df.loc[exact, "judge_rationale"] = "exact match (not sent to judge)"
        ok = df.index[df.parsed & ~exact]
        items = [{"input": df.at[i, "input"],
                  "gold_symptom": df.at[i, "gold_primary_symptom"], "gold_diag": df.at[i, "gold_suggested_diagnostic"],
                  "pred_symptom": df.at[i, "pred_primary_symptom"], "pred_diag": df.at[i, "pred_suggested_diagnostic"]}
                 for i in ok]
        for i, res in zip(ok, judge_items(items, max_workers)):
            df.at[i, "sym_judge"] = res["symptom_score"] / 2      # scale 0-1
            df.at[i, "diag_judge"] = res["diagnostic_score"] / 2
            df.at[i, "judge_rationale"] = res["rationale"]

    df["slice"], df["group"] = "all", df.row_id.astype(str)
    if split and slices_path and os.path.exists(slices_path):
        sl = pd.read_csv(slices_path)
        sl = sl[sl.split == split][["row_id", "slice", "group"]]
        if len(sl):
            df = df.drop(columns=["slice", "group"]).merge(sl, on="row_id", how="left")
            df["slice"] = df.slice.fillna("all")
            df["group"] = df.group.fillna(df.row_id).astype(str)
    return df


def summarize(df):
    out = {"n": int(len(df)), "n_concerns": int(df.group.nunique())}
    cols = [("strict_json", "strict_json"), ("schema_valid", "schema_valid"),
            ("system_acc", "sys_correct"), ("severity_acc", "sev_correct"),
            ("symptom_sim", "sym_sim"), ("diag_sim", "diag_sim")]
    if "sym_judge" in df:
        cols += [("symptom_judge", "sym_judge"), ("diag_judge", "diag_judge")]
    for name, col in cols:
        m, lo, hi = bootstrap_ci(df[col].astype(float), df.group)
        out[name] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["system_macro_f1"] = round(macro_f1(df.gold_vehicle_system, df.pred_vehicle_system, SYSTEMS), 4)
    out["severity_macro_f1"] = round(macro_f1(df.gold_severity, df.pred_severity, SEVERITIES), 4)
    out["severity_under_triage"] = round(float((df.sev_diff < 0).mean()), 4)   # predicted LESS severe than gold
    out["severity_over_triage"] = round(float((df.sev_diff > 0).mean()), 4)
    sc = df.gold_severity == "safety_critical"
    out["safety_critical_recall"] = round(float((df.pred_severity[sc] == "safety_critical").mean()), 4) if sc.any() else None
    if df.latency_s.notna().any():
        out["latency_s_median"] = round(float(df.latency_s.median()), 3)
    return out


def per_class(df, field, labels):
    gold = df[f"gold_{field}"].tolist()
    pred = [_clean_label(p, labels) for p in df[f"pred_{field}"]]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=labels).round(3)


def confusion(df, field, labels):
    pred = [_clean_label(p, labels) for p in df[f"pred_{field}"]]
    return pd.crosstab(pd.Series(df[f"gold_{field}"].values, name="gold"), pd.Series(pred, name="pred"))


def print_report(df, title=""):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    table = {}
    for sl in ["all"] + sorted(s for s in df.slice.unique() if s != "all"):
        sub = df if sl == "all" else df[df.slice == sl]
        table[f"{sl} (n={len(sub)})"] = summarize(sub)
    keys = ["n_concerns", "strict_json", "schema_valid", "system_acc", "system_macro_f1", "severity_acc",
            "severity_macro_f1", "severity_under_triage", "safety_critical_recall", "symptom_sim", "diag_sim",
            "symptom_judge", "diag_judge", "latency_s_median"]
    fmt = lambda v: (f"{v['mean']:.3f} [{v['ci95'][0]:.2f}-{v['ci95'][1]:.2f}]" if isinstance(v, dict)
                     else ("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))))
    rows = {k: {c: fmt(s.get(k)) for c, s in table.items()} for k in keys if any(k in s for s in table.values())}
    print(pd.DataFrame(rows).T.to_string())
    return table


def save(df, name, split, out_dir="outputs/eval"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/{name}__{split}_rows.csv", index=False)
    summary = {"all": summarize(df)}
    for sl in df.slice.unique():
        if sl != "all":
            summary[sl] = summarize(df[df.slice == sl])
    Path(f"{out_dir}/{name}__{split}_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True, help="split name in eval_slices.csv, e.g. val_seen / val_novel / test")
    ap.add_argument("--name", default=None, help="model name for output files (default: from pred filename)")
    ap.add_argument("--slices", default="data/processed/eval_slices.csv")
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge on free-text fields")
    a = ap.parse_args()
    name = a.name or Path(a.pred).stem.split("__")[0]
    rows = evaluate(a.gold, a.pred, a.split, a.slices, judge=a.judge)
    print_report(rows, f"{name} on {a.split}")
    print("\nSeverity per class:\n", per_class(rows, "severity", SEVERITIES))
    print("\nSeverity confusion:\n", confusion(rows, "severity", SEVERITIES))
    save(rows, name, a.split)
