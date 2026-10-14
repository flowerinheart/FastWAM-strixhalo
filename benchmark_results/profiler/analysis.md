# FastWAM torch profiler 与 roofline 分析

采集日期：2026-10-14。使用 LIBERO uncond release，BF16、batch=1、双相机 224×224（拼接为 224×448）、8 维 proprio、32×7 动作输出、10 步去噪、shift=5.0，eager + math SDPA。GPU Radeon 8060S / gfx1151，环境 rocm-nightly。

## 延迟与采集方法

3 次预热，5 次未开启 profiler 的推理：中位数 **603.15 ms**，平均 **610.78 ms**。随后采集 1 次完整推理，profiler 内耗时 **826.06 ms**。记录 CPU/GPU 活动、input shapes、FLOPs、memory；不采集 Python stacks。模型加载和预热不在 trace 内。峰值 allocated 显存 **23.325 GiB**，动作输出有限。

profiler 有明显开销，826 ms 不能作为正常运行延迟。roofline 使用本次未开启 profiler 的中位数 603.15 ms；GPU kernel 时间取自 trace。其 80.95 ms “Host” 是两者相减的估计，混合了两个测量条件，不能视为直接测得的纯 CPU 时间。

## GPU 阶段分布

通过 kernel External id 找到 CPU op，再归属到包含该 op 的阶段 annotation，累计 GPU kernel duration（不是 annotation span）。

| 阶段 | GPU kernel 时间 | 占比 | kernel 数 |
|---|---:|---:|---:|
| 10 步动作去噪 | 323.19 ms | 61.89% | 31,410 |
| T5 文本编码 | 84.34 ms | 16.15% | 1,600 |
| 视频缓存 prefill | 83.37 ms | 15.97% | 3,060 |
| VAE 编码 | 29.87 ms | 5.72% | 438 |
| 其他 | 1.42 ms | 0.27% | 69 |
| 合计 | 522.19 ms | 100% | 36,577 |

PyTorch key_averages 的 FastWAM annotation 行包含 GPU range spans、异步排队和嵌套事件，部分百分比超过 100%；请用上表分析阶段占比，不要将 annotation 行和 kernel 行累加。

## Roofline 结果

已调用指定 `/home/haojiema/code/rocm/rocm-scripts/tools/roofline_trace.py`，分别输出 summary 和 report-by-op，均退出 0。

| 类型 | GPU 时间 | 占总 GPU 时间 | 工具估算效率 |
|---|---:|---:|---:|
| GEMM | 380.60 ms | 72.9% | 48.4% |
| Elementwise | 79.12 ms | 15.2% | 64.8% |
| Convolution | 20.02 ms | 3.8% | 50.0%（大部分缺形状） |
| Reduction | 15.47 ms | 3.0% | 32.1% |

GEMM 中 380.11 ms 被工具判为 memory-bound。这里是基于 FLOPs/bytes 与设定硬件上限的模型分类，不是硬件计数器测得的带宽饱和结论。math SDPA 会拆成 GEMM、softmax/reduction 和 elementwise，报告没有独立 ATTN 行不代表模型未使用注意力。

工具采用 gfx1151 的 BF16 WMMA 43 TFLOPS、read bandwidth 230 GiB/s、copy bandwidth 216 GiB/s。可估算工作覆盖 **87.0%** GPU 时间；**40.78 ms（7.8%）**匹配类型但缺 shape 数据，另有 **26.98 ms（5.2%）**排除类型。7,868 个 kernel 的估算效率超过 100%，工具按缓存/Winograd 等影响将其截到 100%。因此这些效率用于筛选优化目标，不是严格的硬件利用率。

模型估算的 roofline GPU 下限为 **294.83 ms**，端到端下限 **375.78 ms**；若可分析工作达到 80% roofline，端到端目标为 **432.55 ms**，相对当前中位数约有 **170.60 ms / 28.3%** 改善空间。这是理论估计，未通过优化实验验证。

## 优先关注的算子

1. **动作去噪交叉注意力的 context K/V 投影**：`aten::addmm`，A BF16 `[129,1024]`，B `[1024,3072]`，600 次，共 **102.52 ms（19.6% GPU 时间）**；平均 **170.86 µs**，估算 4.78 TFLOPS / 40.3 GiB/s、**17.5% roofline**。工具估计达到 80% roofline 可回收 **80.18 ms**。全部归属 action denoise。次数与 30 层 × K/V 两个投影 × 10 步一致；代码在 `CrossAttention.forward` 中每步重新执行 `k(ctx)`/`v(ctx)`。可优先评估 kernel selection，并将固定 context 的每层 K/V（包含 K norm）在一次 infer 内缓存。这需要确认 context 在去噪步骤间不变并验证数值一致性。
2. **T5 大矩阵乘法**：`aten::mm`，`[128,4096] @ [4096,4096]`，96 次、**28.33 ms**、效率 **49.1%**，gain@80% **11.01 ms**。另外两组 T5 FFN `[128,10240] @ [10240,4096]`、`[128,4096] @ [4096,10240]` 共 **45.06 ms**。相同任务指令反复使用时，可以缓存 T5 embedding；当前基线每次都重新编码。
3. **小 M 动作投影**：`[32,1024] @ [1024,3072]`，1,200 次、**47.08 ms**、效率 **68.4%**，gain@80% **7.27 ms**。其单次执行短，kernel tuning、融合及 compile/CUDA graph 值得评估；本次没有启用 compile/graph。

默认融合 SDPA 路径此前遇到 hipErrorInvalidValue，本次沿用可工作的 math 后端，结论适用于这个兼容配置。未来若融合 SDPA 可稳定运行，需要重新采集，不能直接外推本报告。

## 结果文件与复现

- `fastwam.trace.json`：原始 Chrome trace，可用 Perfetto 打开。
- `fastwam.trace.json.gz`：压缩版同一 trace。
- `roofline_summary.txt` / `roofline_by_op.txt`：指定工具的原始输出。
- `stage_breakdown.json`：阶段 kernel 累计时间与热点归属。
- `metadata.json`：环境、计时和采集配置。
- `self_cuda_time_total.txt` / `self_cpu_time_total.txt`：torch profiler 原始算子表。
- `roofline_commands.json`：实际分析命令；`roofline_filters.json`：工具筛选配置。
- `profile_fastwam.py`：可复现采集脚本，使用 rocm-nightly 的 python 执行。

```bash
cd /home/haojiema/code/amd_robot/models/FastWAM
/home/haojiema/miniconda3/envs/rocm-nightly/bin/python benchmark_results/profiler/profile_fastwam.py
/home/haojiema/miniconda3/envs/rocm-nightly/bin/python /home/haojiema/code/rocm/rocm-scripts/tools/roofline_trace.py --trace benchmark_results/profiler/fastwam.trace.json --gpu gfx1151 --end-to-end-ms 603.1476 --report-by-op
```
