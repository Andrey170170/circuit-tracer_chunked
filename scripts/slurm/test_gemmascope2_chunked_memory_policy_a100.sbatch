#!/bin/bash
# Edit account/partition lines below for your allocation if needed.
#SBATCH --job-name=ct-gs2-vram-p0p1
#SBATCH --account=PAS3272
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
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

echo "== running CPU-side memory policy regression tests =="
uv run pytest -vv -s --log-cli-level=INFO \
  tests/test_chunked_decoder_optimizations.py -k "encoder_vectors_are_cpu_staged or error_vector_prefetch_window or replay_windows_match_full_replay or guardrail or cleanup_is_idempotent"

echo "== running GemmaScope-2 exact chunked GPU memory policy tests =="
uv run pytest -vv -s --log-cli-level=INFO \
  tests/test_gemmascope2_chunked_gpu.py \
  tests/test_gemmascope2_chunked_memory_policy_gpu.py
