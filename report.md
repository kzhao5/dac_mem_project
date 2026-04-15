# Project Report

## Title
**Not Everything Useful Should Be Remembered: Derivability-Aware Persistent Memory for Long-Horizon LLM Agents**

---

## 1. Motivation

Long-horizon LLM agents increasingly rely on external memory to support multi-session dialogue, personalization, task tracking, and retrieval-augmented reasoning. Recent work has made clear that long-term memory quality is a serious bottleneck for reliable conversational agents and memory-augmented systems. Two public benchmarks are especially relevant for this project:

- **LoCoMo** provides very long multi-session conversational data with annotated QA evidence and event summaries.
- **LongMemEval** evaluates five key long-term memory abilities of chat assistants: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention.

At the same time, recent methods such as **A-MAC** and **Memory-R1** show that memory admission and memory operations matter substantially; memory should not be treated as a passive append-only store.

However, most existing systems still focus on:
- what is useful
- what is novel
- what is recent
- what is reliable

and much less on the following question:

> **Should this information occupy persistent long-term memory if it can be cheaply recovered later from tools or environment state?**

This project is built around that exact gap.

---

## 2. Core idea

We propose **DAC-Mem**, a **Derivability-Aware Controller** for persistent memory writing.

The main intuition is:

> **Persistent memory should store what is valuable and not cheaply recoverable.**

Some information is indeed useful, but still does not deserve persistent storage.

Examples:
- a current file path
- a calendar update
- a version number
- a temporary branch status

These facts may be useful, but they are often:
- recoverable from tools later
- likely to become stale
- harmful if redundantly persisted

By contrast, other information is harder to re-derive and therefore more appropriate for persistent memory:
- user preferences
- decision rationales
- reasons for rejecting a plan
- subtle long-term constraints

This leads to a three-way write decision:
- **PERSIST** — store in long-term memory
- **EPHEMERAL** — keep only in short-term / working memory
- **SKIP** — do not store

The important design decision here is to **separate persistent from ephemeral** memory rather than treating admission as a simple binary choice.

---

## 3. Why this design is principled

This design was chosen because it satisfies three goals simultaneously.

### 3.1 It matches the problem structure
The actual scientific question is not “how do we build a bigger memory system?” but “what deserves persistence?” Therefore the contribution should be a **write policy**, not a large new architecture.

### 3.2 It is easy to evaluate cleanly
A write controller can be inserted in front of existing memory systems and evaluated against strong, interpretable baselines.

### 3.3 It is simple enough to run
A full RL memory manager like Memory-R1 is powerful, but much heavier to reproduce. For a course project and a first paper version, a lightweight, transparent controller is the most practical way to get clean experimental evidence.

---

## 4. Method

### 4.1 Candidate extraction
Each dialogue turn is split into sentence-level memory candidates.

Why this design:
- sentence-level candidates are atomic enough for admission decisions
- they are cheap to extract
- they preserve traceability back to original turns
- the paper contribution stays focused on write control, not extraction quality

### 4.2 Memory typing
Each candidate is assigned a type using transparent rules:
- `preference`
- `stable_profile_fact`
- `decision_rationale`
- `event_state`
- `tool_recoverable_artifact`
- `plan_intent`
- `transient_debug_state`
- `generic_fact`

Why this design:
- type is a strong prior for both derivability and stale risk
- type priors are also important in A-MAC-style admission control
- it keeps the system interpretable

### 4.3 Utility estimate
For each candidate memory `m`, we compute a lightweight utility score using:
- type prior
- specificity
- speaker importance
- confidence
- small bonus for explicit updates

This utility score is intentionally simple and does **not** try to predict future tasks perfectly. It is a proxy for whether a fact is likely to matter later.

### 4.4 Tool-conditioned derivability probe
This is the central contribution.

We define a simplified future access model with structured tools such as:
- `profile`
- `calendar`
- `artifact`

These tools are deliberately **narrow**. They do not expose the whole raw transcript. That is important, because otherwise everything would be trivially recoverable.

For each candidate memory:
1. we determine which tools are relevant to its type
2. we query the corresponding structured tool index
3. we compute a recoverability score from similarity and access cost
4. we combine this with a type-based recoverability prior

The result is a derivability score in `[0, 1]`.

Why this design:
- it operationalizes recoverability in a way that can be implemented today
- it is more principled than just asking an LLM “is this derivable?”
- it stays efficient enough for large benchmark sweeps

### 4.5 Stale-risk estimate
We estimate stale risk from:
- memory type
- temporal expressions
- artifact/version/status cues

Why this design:
- derivable information is often also more likely to become stale
- stale risk is a strong reason to avoid persistence even when utility is non-trivial

### 4.6 Final decision rule
For each item, DAC-Mem computes:

- a **persist score**
- an **ephemeral score**

The persist score rewards:
- utility
- novelty
- type prior
- confidence
- mild recency

and penalizes:
- derivability
- stale risk
- duplicates

The ephemeral score rewards:
- utility
- recency
- derivability
- stale risk

This design encodes the intended semantics:
- highly useful but hard-to-recover facts should persist
- useful but recoverable or fast-aging facts should be ephemeral
- low-value facts should be skipped

---

## 5. Baselines

We evaluate against four baselines.

### 5.1 Store-All
Persist everything.

Why included:
- strong sanity-check baseline
- shows the cost of memory pollution and stale accumulation

### 5.2 Relevance-Only
Persist items whose utility exceeds a threshold.

Why included:
- isolates the “store what seems useful” assumption
- tests whether derivability really adds something beyond utility

### 5.3 Novelty/Recency
Persist information that is new and recent.

Why included:
- many memory systems implicitly favor fresh and non-duplicate content
- helps test whether DAC-Mem is just a novelty/recency heuristic in disguise

### 5.4 A-MAC-lite
A lightweight A-MAC-style controller using:
- utility
- confidence
- novelty
- recency
- type prior

Why included:
- this is the most important recent near-neighbor baseline
- it makes the comparison scientifically meaningful

We call it **A-MAC-lite** because this implementation is an independent reimplementation of the public design principles, not a claim of reproducing the original authors’ exact code or reported numbers.

---

## 6. Datasets and benchmark choices

### 6.1 Main benchmark: LoCoMo
Why LoCoMo:
- standard long-term memory conversational benchmark
- public JSON release
- evidence turn IDs available for retrieval evaluation
- session structure is explicit

Use in our project:
- development benchmark
- cross-validated tuning of thresholds and weights
- evidence retrieval evaluation

### 6.2 Main transfer benchmark: LongMemEval
Why LongMemEval:
- broad coverage of long-term memory abilities
- includes knowledge updates and abstention
- provides evidence session labels and turn-level `has_answer` labels
- widely used in recent memory papers

Use in our project:
- out-of-domain transfer benchmark
- test whether DAC-Mem tuned on LoCoMo generalizes to a harder assistant-memory setting

### 6.3 Optional extension: PERMA / MemoryCD
Why future work only for now:
- these benchmarks are especially useful for personalization and continual preference changes
- they would strengthen the paper later
- for the first implementation, LoCoMo + LongMemEval is the best balance between legitimacy and feasibility

---

## 7. Evaluation protocol

We do **not** create a new benchmark. Instead, we use existing benchmarks plus a structured tool-access simulation.

### 7.1 Retrieval evaluation
For each example, we:
1. ingest the full interaction history
2. build persistent memory, ephemeral memory, and structured tool indexes
3. retrieve top-k items for the final question
4. compare retrieved turn/session IDs against benchmark evidence labels

Main retrieval metrics:
- `turn_hit@k`
- `session_hit@k`
- `turn_precision@k`
- `session_precision@k`
- `turn_recall@k`
- `session_recall@k`

### 7.2 Memory quality evaluation
We also measure:
- average persistent memory size
- average ephemeral memory size
- fraction of persisted items that are still derivable
- mean stale risk of persisted items
- duplicate fraction in persistent memory

These metrics are crucial because our paper is about write quality, not just answer quality.

### 7.3 Optional QA evaluation
The repository also supports optional QA metrics:
- exact match
- token-level F1

A lightweight heuristic reader is included by default, and a Hugging Face extractive reader can be enabled for stronger downstream QA experiments.

---

## 8. Main experiments

### Experiment 1: Baseline comparison on LoCoMo
Compare:
- Store-All
- Relevance-Only
- Novelty/Recency
- A-MAC-lite
- DAC-Mem

Hypothesis:
- Store-All should have high memory growth and worse precision
- A-MAC-lite should be a strong baseline
- DAC-Mem should reduce persistent memory size and derivable/stale persistence while preserving evidence recall

### Experiment 2: Zero-shot transfer to LongMemEval
Tune DAC-Mem on LoCoMo dev split and test on LongMemEval.

Hypothesis:
- derivability-aware admission should transfer because recoverability and staleness are structural properties, not dataset-specific tricks

### Experiment 3: Memory-efficiency tradeoff
Plot evidence retrieval versus persistent memory size.

Hypothesis:
- DAC-Mem should achieve a better Pareto frontier than Store-All and Relevance-Only

---

## 9. Ablations

### Ablation A: remove derivability
Set derivability weight to zero.

Why:
- directly tests the central hypothesis

### Ablation B: remove stale risk
Set stale-risk penalty to zero.

Why:
- tests whether derivability alone is enough

### Ablation C: binary admit/reject
Disable the ephemeral pathway.

Why:
- tests whether the persistent/ephemeral split is actually useful

### Ablation D: weaken the probe
Use a weaker recoverability signal instead of the full tool-conditioned probe.

Why:
- tests whether bounded recoverability matters more than a generic heuristic

### Ablation E: access-model variation
Evaluate under:
- profile/calendar only
- profile/calendar/artifact
- stricter tool budget

Why:
- derivability is only meaningful relative to a future access model

---

## 10. Hyperparameter tuning plan

We use a simple grid search for:
- persist threshold
- ephemeral threshold
- derivability weight
- stale-risk weight
- retrieval top-k
- ephemeral session window

The default tuning objective is:

`turn_hit@k + 0.5 * session_hit@k - λ * persistent_size`

Why this objective:
- our method should keep evidence recall high
- but should not win by storing everything
- the regularization term enforces the intended compactness behavior

---

## 11. Expected findings

If the method works as intended, the expected pattern is:

### Store-All
- largest persistent memory
- highest derivable fraction in persistent memory
- higher stale rate
- not necessarily better retrieval precision

### Relevance-Only
- smaller memory than Store-All
- still stores many recoverable event/state facts

### Novelty/Recency
- compact memory
- may drop older but important long-term constraints

### A-MAC-lite
- strong balanced baseline
- good control over value and reliability
- still lacks explicit recoverability reasoning

### DAC-Mem
- smaller persistent memory than Store-All and Relevance-Only
- lower stale-risk among persisted items
- lower fraction of derivable persisted facts
- similar or slightly better evidence retrieval
- best performance when ephemeral memory is enabled

---

## 12. Why this project is a good paper direction

This project is attractive because it hits a real and current gap:
- it is **not** a generic memory architecture paper
- it is **not** “just another benchmark”
- it is a clean **algorithmic control policy** with a clear scientific claim

The method is also easy to explain in one sentence:

> useful information is not enough; persistent memory should prefer information that is useful **and** difficult to recover later

That is exactly the kind of principle-driven contribution that can grow from a course project into a publishable paper.

---

## 13. Limitations

This implementation is deliberately lightweight and therefore has several limitations:

1. candidate extraction is sentence-based and heuristic
2. memory typing is rule-based rather than learned
3. the derivability probe is simulated with narrow structured tools
4. the default reader is not a strong generative QA model
5. A-MAC-lite is a principled reimplementation, not an exact official reproduction

These are acceptable for a first implementation because the goal is to validate the write-policy idea cleanly and efficiently.

---

## 14. Most important future upgrades

The strongest upgrades for a real EMNLP submission would be:

1. replace heuristic extraction with a structured LLM fact extractor
2. replace the current probe with a real bounded recovery agent
3. add a stronger reader or answer model
4. expand to PERMA / MemoryCD for personalization
5. add statistical significance testing and more formal calibration of tool budgets

---

## 15. Final summary

This project proposes a simple but principled idea:

> **Persistent memory is for the irrecoverable.**

We operationalize this idea with a lightweight controller that decides whether each textual memory candidate should be persisted, kept only ephemerally, or skipped.

The design is:
- intuitive
- efficient
- interpretable
- easy to run on public benchmarks
- well aligned with the recent academic shift from static memory stores to explicit memory control policies

