#!/bin/bash
# Edit account/partition lines below for your allocation if needed.
#SBATCH --job-name=ct-phase0-stats
#SBATCH --account=PAS3272
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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

echo "== running Phase 0 stats and encoder gather regression tests =="
uv run pytest -vv -s --log-cli-level=INFO \
  tests/test_phase0_stats.py \
  tests/test_cross_layer_transcoder.py -k "encode_sparse_with_lazy_encoder or gather_encoder_vectors_preserves_sparse_order" \
  tests/test_double_pass_sparsification.py
