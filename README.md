# FastWAM on AMD Strix Halo — ROCm

| Configuration | Median | P95 |
|---|---:|---:|
| Original eager execution | 640.2 ms | 649.2 ms |
| Context embedding and K/V caching | 562.5 ms | 575.8 ms |
| Caching + `torch.compile` | 399.7 ms | 404.6 ms |
| Verification in a separate process | 412.8 ms | 420.7 ms |
| Final reproduction script run | **393.1 ms** | **403.3 ms** |

This repository is forked from [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM), the original Fast-WAM implementation. This fork documents and reproduces inference optimization on **AMD Radeon 8060S / Strix Halo (`gfx1151`)** with ROCm. The final run reduces median latency by approximately **38.6%** relative to the original eager baseline.

[中文说明](README_zh.md)

## Test configuration

Measured on October 14, 2026. Each table row uses 3 warmup iterations and 50 measured iterations with GPU synchronization, excluding model loading and initial compilation.

- LIBERO `libero_uncond_2cam224.pt` release checkpoint.
- BF16, batch size 1; two synthetic 224×224 camera images concatenated to 224×448; 8-dimensional proprioception.
- 10 denoising steps, scheduler shift 5.0; output shape `(32, 7)`.
- Timing includes T5 text encoding, VAE encoding, video cache prefill and action inference. Robot capture, communication and control execution are excluded.
- Peak PyTorch allocated GPU memory: approximately **23.325 GiB**.
- Math SDPA is used because the default fused SDPA path failed with `hipErrorInvalidValue` on this nightly/GPU combination.

The optimization computes context embeddings and each action layer's cross-attention K/V once per inference, then reuses them across the 10 denoising steps. The cache includes the current proprioception and does not persist between calls. The compiled path uses the upstream `torch.compile` implementation.

Caching alone produced exactly identical action outputs in three full-inference comparisons. Compilation introduced a maximum absolute difference of **0.01172** across three comparisons, passing `rtol=0.02, atol=0.01`; outputs were finite. LIBERO task success rates have not been evaluated in this fork.

## ROCm environment

Validated environment:

| Component | Version |
|---|---|
| Python | 3.12.14 |
| PyTorch | `2.15.0a0+rocm10.1.0a20260902` |
| torchvision | `0.30.0a0+rocm10.1.0a20260902` |
| ROCm wheel | `10.1.0a20260902` |
| HIP reported by PyTorch | `7.16.26351` |
| GPU architecture | `gfx1151` |

Use the existing validated environment on the test machine:

```bash
conda activate rocm-nightly
```

For another machine, create a Python 3.12 environment and install a matching ROCm PyTorch/torchvision nightly pair supporting gfx1151, following the [ROCm installation documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/). Historical nightly wheels may not remain available; newer builds require revalidation. Do not install CUDA wheels.

```bash
git clone https://github.com/flowerinheart/FastWAM-strixhalo.git
cd FastWAM-strixhalo
python -m pip install -r requirements-rocm.txt
# The upstream package metadata contains CUDA/training pins; bypass those.
python -m pip install -e . --no-deps
python -c 'import torch; print(torch.__version__, torch.version.hip); print(torch.cuda.get_device_name(0))'
```

## Download inference weights

Downloads total approximately 25 GB. The trained checkpoint already contains both experts; preprocessing an ActionDiT backbone or downloading base DiT shards is unnecessary.

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
for filename in ['libero_uncond_2cam224.pt', 'libero_uncond_2cam224_dataset_stats.json']:
    hf_hub_download('yuanty/fastwam', filename, local_dir='checkpoints/fastwam_release')
for filename in ['Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth']:
    hf_hub_download('Wan-AI/Wan2.2-TI2V-5B', filename,
                    local_dir='checkpoints/Wan-AI/Wan2.2-TI2V-5B')
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', allow_patterns='google/umt5-xxl/*',
                  local_dir='checkpoints/Wan-AI/Wan2.1-T2V-1.3B')
PY
```

The commands use original Wan weights with `model.redirect_common_files=false`; the upstream converted-weight repository returned 404 during testing.

## Reproduce latency and output checks

Run from the repository root with the ROCm environment active:

```bash
# Final optimized configuration: math SDPA + context cache + compilation.
bash benchmark_results/optimization/run_optimized.sh

# Separate measurements, each with 50 iterations.
python benchmark_results/optimization/benchmark_opt.py --label baseline_repeat --no-cache
python benchmark_results/optimization/benchmark_opt.py --label cache_repeat --verify
python benchmark_results/optimization/benchmark_opt.py --label compiled_repeat --compile --verify
```

Initial compilation can take several minutes. Report steady-state timings after warmup. The benchmark writes individual timings, median and P95 to JSON, plus output-comparison results. Disable internal caching with `cache_action_context=False` in Python or `+DRYRUN.cache_action_context=false` in the upstream dryrun CLI. Enable compilation with `compile_action_infer=True` or `+DRYRUN.compile_action_infer=true`.

Recorded results: [baseline](benchmark_results/optimization/baseline.json), [cache](benchmark_results/optimization/context_cache.json), [compiled](benchmark_results/optimization/context_cache_compile.json), [independent repeat](benchmark_results/optimization/final_repeat.json), [final script](benchmark_results/optimization/repro.json), [output checks](benchmark_results/optimization/final_repeat_correctness.json).

## Reproduce profiler and roofline analysis

```bash
# Baseline trace, excluding model loading and warmup.
python benchmark_results/profiler/profile_fastwam.py
# Optimized trace written to benchmark_results/optimization/profiler/.
python benchmark_results/profiler/profile_fastwam.py --optimized
```

Trace collection records CPU/GPU activity, tensor shapes, FLOPs and memory. Profiling adds overhead; use unprofiled latency for performance comparisons. Open `fastwam.trace.json` in Perfetto. Large traces and model weights are not committed.

With a checkout of [rocm-scripts](https://github.com/FaaSApps/rocm-scripts), run its roofline tool:

```bash
# Replace this path with your checkout.
ROOFLINE_TOOL=/home/haojiema/code/rocm/rocm-scripts/tools/roofline_trace.py
python "$ROOFLINE_TOOL" \
  --trace benchmark_results/profiler/fastwam.trace.json \
  --gpu gfx1151 --end-to-end-ms 603.1475939671509
python "$ROOFLINE_TOOL" \
  --trace benchmark_results/profiler/fastwam.trace.json \
  --gpu gfx1151 --end-to-end-ms 603.1475939671509 --report-by-op
python "$ROOFLINE_TOOL" \
  --trace benchmark_results/optimization/profiler/fastwam.trace.json \
  --gpu gfx1151 --end-to-end-ms 412.79619600391015 --report-by-op
```

Replace `--end-to-end-ms` with your own unprofiled median when rerunning. Baseline profiling measured 603.15 ms unprofiled and 826.06 ms with profiler; those runs are separate from the 50-iteration table.

GPU kernel time decreased from **522.19 ms to 362.29 ms**, and kernel count from **36,577 to 14,103**. GEMM time decreased from **380.60 ms to 282.59 ms**; elementwise time from **79.12 ms to 31.69 ms**. The baseline's largest hotspot was repeated context K/V projection, accounting for **102.52 ms** across 600 calls.

Roofline estimates depend on modeled traffic and hardware ceilings. Optimized traces have only **28.2%** analyzable coverage because compiled kernels lose some CPU shape mappings; their theoretical limits should not be interpreted as measured achievable latency.

Reports: [baseline summary](benchmark_results/profiler/roofline_summary.txt), [baseline per-op](benchmark_results/profiler/roofline_by_op.txt), [optimized summary](benchmark_results/optimization/profiler/roofline_summary.txt), [optimized per-op](benchmark_results/optimization/profiler/roofline_by_op.txt).
