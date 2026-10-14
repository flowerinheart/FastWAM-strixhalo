# FastWAM — Strix Halo / ROCm

| 配置 | 中位数 | P95 |
|---|---:|---:|
| 原始 eager | 640.2 ms | 649.2 ms |
| 缓存 context embedding 和 K/V | 562.5 ms | 575.8 ms |
| 缓存 + `torch.compile` | 399.7 ms | 404.6 ms |
| 独立进程复测 | 412.8 ms | 420.7 ms |
| 最终复现脚本运行 | **393.1 ms** | **403.3 ms** |

本仓库 fork 自 [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)，用于记录和复现 AMD Radeon 8060S（Strix Halo / gfx1151）上的 ROCm 推理优化。每组预热 3 次、测量 50 次。最终中位数相比原始 eager 降低约 38.6%。

测试为 LIBERO checkpoint、BF16、batch=1、双相机 224×224、8 维 proprio、10 步去噪、shift=5.0，输出 32×7 动作。耗时包含每次 T5 和 VAE 编码，排除模型加载、首次编译、预热和机器人 IO。峰值 allocated 显存约 23.325 GiB。

优化在单次推理内缓存 context embedding 和各层 K/V，结合原仓库编译路径。仅缓存时三组完整输出逐元素一致；编译版最大绝对误差为 0.01172，通过数值容差检查，尚未验证 LIBERO 成功率。使用 math SDPA 绕过本机默认后端的 hipErrorInvalidValue。

## 环境和权重

实测使用 Python 3.12.14、PyTorch `2.15.0a0+rocm10.1.0a20260902`、torchvision `0.30.0a0+rocm10.1.0a20260902`。先安装支持 gfx1151 的匹配 ROCm nightly torch/torchvision，再安装推理依赖。新 nightly 版本需要重新验证。

```bash
conda activate rocm-nightly
git clone https://github.com/flowerinheart/FastWAM-strixhalo.git
cd FastWAM-strixhalo
python -m pip install -r requirements-rocm.txt
python -m pip install -e . --no-deps
```

原始 pyproject 包含 CUDA 和训练依赖 pin，因此 editable 安装使用 `--no-deps`。权重共约 25 GB；下载命令见 [英文 README 的权重教程](README.md#download-inference-weights)。

## 复现

下载权重后，从仓库根目录执行：

```bash
bash benchmark_results/optimization/run_optimized.sh
python benchmark_results/optimization/benchmark_opt.py --label baseline_repeat --no-cache
python benchmark_results/optimization/benchmark_opt.py --label cache_repeat --verify
python benchmark_results/optimization/benchmark_opt.py --label compiled_repeat --compile --verify
```

首次编译可能需要数分钟。脚本记录每次 latency、中位数、P95 和输出误差。

```bash
python benchmark_results/profiler/profile_fastwam.py
python benchmark_results/profiler/profile_fastwam.py --optimized
```

这两条命令生成基线和优化后的 profiler trace。权重及大型 trace 不提交到 Git。Roofline 命令和测试数据链接见 [英文复现教程](README.md#reproduce-profiler-and-roofline-analysis)。分析时端到端延迟应使用未开启 profiler 的实测中位数。

优化后 GPU kernel 总时间从 522.19 ms 降到 362.29 ms，kernel 数从 36,577 降到 14,103。编译 trace 的 roofline 可分析覆盖率仅 28.2%，理论下限不能当作下一轮可保证达到的目标。
