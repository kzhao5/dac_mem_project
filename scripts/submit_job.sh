#!/bin/bash
#SBATCH --job-name=dac_mem
#SBATCH --output=logs/dac_mem_%j.out
#SBATCH --error=logs/dac_mem_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

# ── Setup ──────────────────────────────────────────────────────────────────
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate dac_mem

cd /home/kzhao2/dac_mem_project
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

# ── Choose model (edit this line) ──────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# ── Run experiments ────────────────────────────────────────────────────────

echo "=== Running on synthetic ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset synthetic \
    --device cuda

echo "=== Running on LoCoMo ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset locomo \
    --device cuda \
    --limit 50

echo "=== Running on LongMemEval ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset longmemeval \
    --device cuda \
    --limit 50

echo "=== All experiments complete ==="
