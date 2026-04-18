#!/bin/bash
#SBATCH --job-name=memjudge
#SBATCH --output=logs/memjudge_%j.out
#SBATCH --error=logs/memjudge_%j.err
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ── Setup ──────────────────────────────────────────────────────────────────
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate dac_mem

cd /home/kzhao2/dac_mem_project
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

echo "=== Job info ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date:      $(date)"
echo "PWD:       $(pwd)"
echo ""

# ── Model (edit this line to switch models) ────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# ── Run experiments ────────────────────────────────────────────────────────

echo "=== [1/3] Synthetic ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset synthetic \
    --device cuda

echo ""
echo "=== [2/3] LoCoMo ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset locomo \
    --device cuda \
    --limit 50

echo ""
echo "=== [3/3] LongMemEval ==="
python scripts/run_benchmark.py \
    --model $MODEL \
    --dataset longmemeval \
    --device cuda \
    --limit 50

echo ""
echo "=== All experiments complete ==="
echo "Results saved to: results/"
