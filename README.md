# DAC-Mem: Derivability-Aware Persistent Memory for Long-Horizon LLM Agents

This repository contains a complete, runnable reference project for the paper idea:

**What Not to Remember: Tool-Conditioned Memory Writing for Long-Horizon LLM Agents**

The core idea is simple:

> Persistent memory should store what is valuable and **not** cheaply recoverable later from tools or environment state.

Instead of storing every seemingly useful fact, we score each candidate memory with:
- utility
- novelty
- confidence
- type prior
- **derivability** under a bounded future access model
- stale risk

The controller outputs one of three actions:
- `PERSIST`
- `EPHEMERAL`
- `SKIP`

This implementation is intentionally lightweight and transparent. It does **not** require closed-source APIs and can run with rule-based extraction, hybrid lexical retrieval, and optional Hugging Face extractive QA.

---

## 1. Repository layout

```text
configs/default.yaml              default hyperparameters
data/synthetic/                   tiny smoke-test datasets
scripts/run_experiment.py         run one controller on one dataset
scripts/run_all.py                compare all baselines
scripts/run_ablations.py          ablation study
scripts/tune_dacmem.py            lightweight hyperparameter search
scripts/smoke_test.py             local sanity check
src/dac_mem/                      main package
report.md                         detailed project report
```

---

## 2. Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want optional extractive QA with Hugging Face, keep `torch` and `transformers` installed.

---

## 3. Supported datasets

### LoCoMo
Official source used in the paper release:
- GitHub raw JSON: `data/locomo10.json`

### LongMemEval
Official cleaned JSON release:
- `longmemeval_s_cleaned.json`
- `longmemeval_m_cleaned.json`
- `longmemeval_oracle.json`

This project includes automatic download helpers.

---

## 4. Quick start

### Smoke test on the bundled synthetic dataset
```bash
python scripts/smoke_test.py
```

### Run DAC-Mem on synthetic data
```bash
python scripts/run_experiment.py \
  --dataset synthetic \
  --data_path data/synthetic/longmemeval_mini.json \
  --controller dac_mem \
  --reader_mode heuristic
```

### Compare all baselines on LoCoMo
```bash
python scripts/run_all.py \
  --dataset locomo \
  --data_path data/locomo10.json \
  --download \
  --limit 100
```

### Run on LongMemEval cleaned small split
```bash
python scripts/run_all.py \
  --dataset longmemeval \
  --data_path data/longmemeval_s_cleaned.json \
  --variant s_cleaned \
  --download \
  --limit 200
```

### Run ablations
```bash
python scripts/run_ablations.py \
  --dataset longmemeval \
  --data_path data/longmemeval_s_cleaned.json \
  --variant s_cleaned \
  --download \
  --limit 200
```

### Tune DAC-Mem thresholds/weights
```bash
python scripts/tune_dacmem.py \
  --dataset locomo \
  --data_path data/locomo10.json \
  --download \
  --limit 100
```

---

## 5. Implemented baselines

- `store_all`
- `relevance_only`
- `novelty_recency`
- `amac_lite` — a lightweight A-MAC-style baseline using utility, confidence, novelty, recency, and type prior
- `dac_mem` — our method

Notes:
- `amac_lite` is a clean, independent reimplementation of the **A-MAC design philosophy**, not a claim of reproducing the authors' exact official results.
- The repository currently does **not** reproduce Memory-R1, because that method depends on reinforcement learning and a larger training/evaluation stack.

---

## 6. Evaluation outputs

Each run produces a JSON file in `results/` with:
- aggregated metrics
- per-example retrieval logs
- memory statistics
- optional QA predictions

Main metrics:
- `turn_hit@k`
- `session_hit@k`
- `turn_precision@k`
- `session_precision@k`
- `persistent_size`
- `persistent_derivable_fraction`
- `persistent_stale_mean`
- `persistent_duplicate_fraction`
- optional `qa_em`, `qa_f1`

---

## 7. Why this implementation is useful

This codebase is not trying to be the most powerful memory architecture. It is trying to be:
- **principled**
- **simple**
- **transparent**
- **fast to iterate on**
- **easy to extend into a paper-quality system**

In particular, it keeps the project centered on the paper contribution:

> a new persistent-memory write principle based on future recoverability,
> not a large opaque memory stack.

---

## 8. Recommended paper-style experiment plan

### Main experiments
1. **LoCoMo 5-fold / dev-test style evaluation**
   - tune thresholds on a dev subset
   - compare Store-All, Relevance-Only, Novelty/Recency, A-MAC-lite, DAC-Mem

2. **Zero-shot transfer to LongMemEval**
   - use DAC-Mem settings tuned on LoCoMo
   - evaluate retrieval quality and memory compactness on LongMemEval

3. **Optional personalization transfer**
   - extend to PERMA or MemoryCD later

### Ablations
- remove derivability
- remove stale-risk
- binary admit/reject instead of persist/ephemeral/skip
- weaken tool probe / replace with heuristic derivability
- vary future access model

### Tuning experiments
- persist threshold
- ephemeral threshold
- derivability weight
- stale-risk weight
- retrieval top-k
- ephemeral session window

---

## 9. Suggested next upgrades

If you want to push this toward a stronger EMNLP submission, the most valuable upgrades are:
1. replace heuristic extraction with an instruction-tuned structured extractor
2. replace heuristic derivability with a true bounded recovery agent
3. add a stronger reader for downstream QA
4. expand evaluation to PERMA / MemoryCD
5. add significance testing and confidence intervals

