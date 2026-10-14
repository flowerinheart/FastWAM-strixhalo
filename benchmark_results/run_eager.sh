#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
python benchmark_results/run_rocm.py \
 task=libero_uncond_2cam224_1e-4 \
 ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
 model.redirect_common_files=false \
 model.action_scheduler.infer_shift=5.0 \
 +DRYRUN.compile_action_infer=false +DRYRUN.warmup=3 +DRYRUN.iters=10
