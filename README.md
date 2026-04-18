# MemJudge: LLM-as-Judge for Tool-Conditioned Memory Writing

**Paper:** *What Not to Remember: Tool-Conditioned Memory Writing for Long-Horizon LLM Agents*

## Core idea

> Persistent memory should store what is valuable and **not** cheaply recoverable later from tools or environment state.

**MemJudge** uses an LLM-as-judge to evaluate whether each candidate memory can be recovered from future tools (profile, calendar, artifact, search). Recoverable information is filtered out of persistent memory — it lives only in ephemeral memory or is skipped entirely.

Each candidate is assigned one of three actions:
- `PERSIST` — store in long-term memory (high value, low derivability)
- `EPHEMERAL` — keep only short-term (useful but tool-recoverable)
- `SKIP` — discard (low value)

---

## 1. Repository layout

```
configs/default.yaml        default hyperparameters
data/synthetic/              tiny smoke-test datasets
scripts/
  download.py               pre-download models & data (login node, with internet)
  run_benchmark.py          main experiment (offline on GPU node)
  smoke_test.py             local sanity check (no model, no internet)
  submit_job.sh             SLURM job script
src/dac_mem/                main package
report.md                   detailed project report
```

---

## 2. Installation

```bash
module load miniconda3
conda create -n dac_mem python=3.10 -y
conda activate dac_mem
pip install -r requirements.txt
```

---

## 3. Workflow for GPU cluster (offline compute nodes)

### Step 1: Pre-download models and datasets (login node, has internet)

```bash
module load miniconda3 && conda activate dac_mem
python scripts/download.py
```

This downloads:
- Open-source LLMs: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B
- Embedding model: BGE-large
- Datasets: LoCoMo, LongMemEval

### Step 2: Submit GPU job (offline compute node)

```bash
sbatch scripts/submit_job.sh
```

Or run interactively on a GPU node:

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset synthetic \
    --device cuda
```

### Step 3: Quick smoke test (no GPU, no network)

```bash
PYTHONPATH=src python scripts/smoke_test.py
```

---

## 4. Baselines

- `store_all` — persist everything
- `relevance_only` — utility threshold
- `novelty_recency` — novelty + recency weighted
- `amac_lite` — A-MAC 5-factor admission control
- `memory_bank` — MemoryBank-style timestamp-weighted store
- `memory_r1` — Memory-R1-inspired LLM-prompted CRUD manager (when LLM available)
- `memjudge` — **our method**

---

## 5. Metrics

Four categories:
- **Downstream QA:** `qa_em`, `qa_f1`, `judge_score` (LLM-as-judge, 1–5 scale)
- **Memory quality:** `persistent_size`, `persistent_derivable_fraction`, `persistent_stale_mean`, `persistent_duplicate_fraction`
- **Retrieval:** `turn_hit@k`, `session_hit@k`, `turn_precision@k`, `turn_recall@k`
- **Efficiency:** `latency_sec`, `retrieved_context_words`

Plus: paired bootstrap significance tests (`MemJudge vs baselines`).

---

## 6. Models supported

All via the unified `get_llm(provider, model)` interface:

| Provider | Default model | Notes |
|----------|--------------|-------|
| `huggingface` | Qwen/Qwen2.5-7B-Instruct | Transformers backend |
| `vllm` | Qwen/Qwen2.5-7B-Instruct | Fast batched inference |
| `openai` | gpt-4o | Requires `OPENAI_API_KEY` |
| `anthropic` | claude-sonnet-4-20250514 | Requires `ANTHROPIC_API_KEY` |
| `google` | gemini-1.5-pro | Requires `GOOGLE_API_KEY` |

For the paper we use small open-source models: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B.

---

## 7. Method summary (from controllers.py)

```
For each candidate memory m:
  1. Compute features (novelty, recency, utility, stale_risk)
  2. LLM-as-judge: derivability(m) = P(m can be recovered from tools)
  3. persist_score  = α·utility + δ·type - β·derivability - γ·stale_risk
     ephemeral_score = 0.65·utility + 0.55·derivability + 0.30·stale_risk
  4. If persist_score >= 0.55:   PERSIST
     elif ephemeral_score >= 0.58: EPHEMERAL
     else:                          SKIP
```

The key insight: **derivability acts as a penalty on persist_score** — high-value information that's recoverable from tools should not occupy long-term memory.
