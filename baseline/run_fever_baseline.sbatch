#!/bin/bash
#SBATCH --job-name=fever-transformer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=fyy-ic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "===== GPU INFO ====="
nvidia-smi
echo "===================="

cd /projects/illinois/eng/cs/fyy/cjchong3/CS410_project/baseline
source /projects/illinois/eng/cs/fyy/cjchong3/miniconda3/etc/profile.d/conda.sh
conda activate kv_qwen_fix

mkdir -p logs results

echo "===== FEVER BASELINE (claim only), 500 samples ====="
# python llm_fever_eval.py \
#     --mode baseline \
#     --limit 500 \
#     --batch-size 32

# Other modes (uncomment / swap above):
#
# Retrieval w/ transformer-retrieved docs (longer prompts -> smaller batch)
python llm_fever_eval.py \
    --mode retrieval \
    --method transformer \
    --limit 500 \
    --batch-size 4
#
# Retrieval w/ BM25 (when available)
# python llm_fever_eval.py \
#     --mode retrieval \
#     --method bm25 \
#     --limit 500 \
#     --batch-size 4
#
# Retrieval w/ dense retriever (when available)
# python llm_fever_eval.py \
#     --mode retrieval \
#     --method dense \
#     --limit 500 \
#     --batch-size 4

# sbatch run_fever_baseline.sbatch
# squeue -u cjchong3
# cat logs/fever-baseline-{Job id}.out
# scancel -u cjchong3
