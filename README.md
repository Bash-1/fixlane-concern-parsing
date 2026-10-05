# Fixlane concern parsing: QLoRA fine-tune vs prompted GPT-4o

Parse free-form repair concerns into `{vehicle_system, primary_symptom, severity, suggested_diagnostic}`.
Fine-tuned **Qwen2.5-3B-Instruct** with **QLoRA** and compared it against **GPT-4o** (12 fixed few-shot examples, and
8 retrieved examples per input) plus two non-LLM references (majority class, nearest-neighbour copy).
The full write-up is in the report; this README covers results at a glance and how to run everything.

## Headline results: test set (100 rows: 90 seen-concern, 10 novel-concern; 70 distinct concerns)

| model | system acc | severity acc | safety_critical recall (14) | under-triage | symptom (judge) | diagnostic (judge) |
|---|---|---|---|---|---|---|
| majority class | 0.14 | 0.55 | 0.00 | 0.14 | - | - |
| kNN copy (no model) | 0.94 | 0.95 | 1.00 | 0.00 | 0.91 | 0.91 |
| GPT-4o, 12 static shots | 0.94 | 0.78 | 0.36 | 0.19 | 0.97 | 0.82 |
| GPT-4o, 8 retrieved shots | **0.99** | 0.92 | 0.93 | 0.08 | **0.99** | 0.90 |
| **Qwen2.5-3B QLoRA (ours)** | 0.90 | **0.97** | **1.00** | **0.00** | 0.97 | **0.94** |

Novel-concern slice (10 rows, all EV): GPT-4o retrieval system 0.90 / severity 0.80 / diagnostic 0.90 vs
fine-tune 0.50 / 0.70 / 0.40.

| | median latency, 1 request | cost per 1k concerns* |
|---|---|---|
| GPT-4o retrieval (API) | 0.85 s | ~$3.20 |
| Fine-tune, 4-bit + LoRA (T4) | 6.8 s | ~$0.06 (batched) |

\* GPT-4o at $2.50/$10 per 1M input/output tokens; T4 at an assumed $0.35/h, fully utilised. Verify current prices.

**In one paragraph:** on concerns resembling the training data, the fine-tune matches the best GPT-4o setup,
catches every safety-critical case with no under-triage, reproduces Fixlane's canonical phrasing, and is ~50x cheaper
in batch, but a nearest-neighbour lookup is equally accurate there. On genuinely new concerns, GPT-4o wins clearly
(system and diagnostics), because a 3B model trained on ~92 concerns has little domain knowledge of its own.
Both approaches struggle with EV high-voltage-battery conventions.

## Repo layout
~~~
src/
  common.py            schema, prompts, text normalisation, output formatting (shared)
  prepare_data.py      clustering, label cleaning, held-out split, eval slices  (raw -> data/processed)
  simple_baselines.py  majority + kNN reference baselines
  baseline_openai.py   prompted GPT-4o baseline (static or retrieval few-shot, structured outputs)
  train.py             QLoRA fine-tuning (completion-only loss; --cat_weight, --oversample)
  predict.py           batched greedy generation with base model +/- adapter
  evaluate.py          parsing, per-field metrics, LLM judge, cluster-bootstrap CIs
  benchmark.py         latency / throughput (4-bit+adapter or merged fp16)
data/raw/              dataset as provided
data/processed/        outputs of prepare_data.py
outputs/               predictions, metrics, cleaning log, LLM label audit, report tables
adapters/qwen3b_final/final/   the submitted LoRA adapter
~~~

## Setup (Colab T4 or any CUDA GPU with >= 8 GB)
~~~bash
pip install -r requirements.txt     # torch: use the CUDA build already on your machine/Colab
pip uninstall -y torchao            # Colab ships torchao 0.10, which PEFT rejects when loading onto an fp16 base
export OPENAI_API_KEY=...           # only needed for the GPT-4o baseline and the LLM judge
export HF_TOKEN=...                 # only needed to download the adapter from the (private) Hub repo
~~~
Run all commands from the repo root.

**Not in this repo:** the dataset (put Fixlane's `train/val/test.jsonl` in `data/raw/`, then run `python src/prepare_data.py`) and the LoRA weights (Hugging Face Hub, see *Using the adapter*).

## Reproduce
~~~bash
# 1. Data: cluster, clean, split (deterministic; byte-identical to the shipped data/processed)
python src/prepare_data.py

# 2. Reference baselines (free)
python src/simple_baselines.py --train data/processed/train_clean_full.jsonl --split test

# 3. GPT-4o baselines (~200 calls)
python src/baseline_openai.py --eval data/raw/test.jsonl --train data/processed/train_clean_full.jsonl \
    --mode static --k 12 --out outputs/preds/gpt4o_static__test.jsonl
python src/baseline_openai.py --eval data/raw/test.jsonl --train data/processed/train_clean_full.jsonl \
    --mode retrieval --k 8 --out outputs/preds/gpt4o_retrieval__test.jsonl

# 4. Final fine-tune (~12 min on a T4) and predictions
python src/train.py --train data/processed/train_clean_full.jsonl --output_dir adapters/qwen3b_final \
    --oversample safety_critical:3
python src/predict.py --adapter adapters/qwen3b_final/final --tag ft_final --splits test

# 5. Score (add --judge for LLM-judged free-text fields; exact matches skip the API call)
python src/evaluate.py --gold data/raw/test.jsonl --pred outputs/preds/ft_final__test.jsonl --split test --judge

# 6. Latency
python src/benchmark.py --mode 4bit --tag ft_final_4bit
~~~

**Dev experiments** (selection was done on these, never on test):
~~~bash
python src/train.py --train data/processed/train_clean_dev.jsonl --output_dir adapters/qwen3b_clean_dev                        # first FT
python src/train.py --train data/processed/train_clean_dev.jsonl --output_dir adapters/qwen3b_clean_dev_w5 --cat_weight 5      # A: label-token weighting
python src/train.py --train data/processed/train_clean_dev.jsonl --output_dir adapters/qwen3b_clean_dev_os3 --oversample safety_critical:3  # B: adopted
python src/train.py --train data/processed/train_raw_dev.jsonl --output_dir adapters/qwen3b_raw_dev_os3 --oversample safety_critical:3      # raw-data ablation
python src/predict.py --adapter adapters/<run>/final --tag <tag> --splits val_seen,val_novel
python src/evaluate.py --gold data/processed/val_novel.jsonl --pred outputs/preds/<tag>__val_novel.jsonl --split val_novel
~~~
Dev views: DEV-SEEN = `val_seen` rows of trained concerns (split `val_seen_dev`, slice `seen`);
DEV-NOVEL = `val_novel` + `val_seen` rows of held-out concerns.

## Using the adapter
Hub (private, ask for access): `Bashaarat1/fixlane-qwen2.5-3b-concern-parser-lora`, or the local copy in
`adapters/qwen3b_final/final`. The model card there has a complete loading snippet. Load the base in **4-bit NF4**
exactly as in `predict.py` and use `common.SYSTEM_PROMPT_FT` verbatim. Serve as trained: merging the adapter into an
fp16 base changed 25% of outputs (the adapter was trained against the 4-bit weights).

## Data quality, in short
- 500 train rows contain ~92 distinct concerns, each reworded ~5x (filler phrases, typos).
- Noise found: 14 single-row label flips inside otherwise-consistent concern clusters (5 system, 9 severity) and
  11 placeholder free-text values ("vehicle issue", "check vehicle systems"). Fixed by in-cluster majority vote;
  25 changes logged in `outputs/cleaning_log.csv`.
- Consistent-but-debatable conventions (e.g. EV battery/range -> drivetrain; narrow use of safety_critical) were
  **kept**: they define Fixlane's policy and the test labels follow them. GPT-5.5 audit: `outputs/llm_audit.csv`.
- Val/test labels are clean; val is ~99% rewordings of training concerns, so a group-held-out `val_novel`
  (10 concerns) was built for model selection.

## Notes and caveats
- **License:** Qwen2.5-3B-Instruct is not Apache-2.0 (Hub license: `other`; check its LICENSE). For production, use
  an Apache-2.0 base (e.g. Qwen2.5-1.5B/7B); the pipeline is model-agnostic via `--model`.
- The TF-IDF vocabulary in `prepare_data.py` is fit on input text from all splits (no labels).
- The novel test slice is 10 rows (8 concerns); treat its numbers as directional. CIs are cluster-bootstrapped by concern.
- GPT-4o latency includes network time with 3 parallel workers; fine-tune latency is pure GPU time on a T4.
