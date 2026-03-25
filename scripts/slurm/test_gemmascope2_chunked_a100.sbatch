#!/bin/bash
# Edit account/partition lines below for your Ascend allocation if needed.
#SBATCH --job-name=ct-gs2-chunked
#SBATCH --account=PAS3272
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "== job metadata =="
date
hostname
pwd

echo "== uv environment =="
uv --version

echo "== gpu info =="
nvidia-smi || true

echo "== running full test suite =="
uv run pytest -vv -s --log-cli-level=INFO
