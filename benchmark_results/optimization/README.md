# FastWAM 推理优化结果

目标：相同 BF16、batch=1、双相机 224×224、32×7 动作、10 步去噪，包含 T5 和 VAE 编码，将稳态延迟降到约 450 ms。已实现 context embedding / 每层 cross-attention K/V 的单次推理内缓存，配合仓库已有 torch.compile 路径。

| 配置 | 次数 | 中位数 | P95 |
|---|---:|---:|---:|
| 原始 eager，无 context 缓存 | 50 | 640.23 ms | 649.20 ms |
| eager + context 缓存 | 50 | 562.45 ms | 575.84 ms |
| compile + context 缓存 | 50 | 399.70 ms | 404.61 ms |
| 独立进程复测 compile + context 缓存 | 50 | 412.80 ms | 420.69 ms |
| 最终用户复现入口 | 50 | 393.10 ms | 403.30 ms |

最终复测比原始基线中位数降低 35.5%。所有测量使用 Radeon 8060S、rocm-nightly、math SDPA，包含每次 T5 编码；没有跨调用的 T5 缓存，也没有减少去噪步数。模型加载、首次编译、预热不计入稳态延迟。初次编译可能需要数分钟。

## 修改

- CrossAttention 提供 context K/V 投影入口，可接收已投影并归一化的 K/V。
- ActionDiT.prepare 可接收已计算的 context embedding。
- FastWAM.infer_action 默认 cache_action_context=true，仅在该调用内缓存包含当前 proprio 的 context，随后 10 步复用；关闭可对照旧路径。
- MoT tensor action 路径逐层使用对应 K/V，其他路径保留即时投影。
- dryrun 增加 DRYRUN.cache_action_context 开关；compile 仍由既有 compile_action_infer 控制。

## 验证和限制

缓存 eager 对照三组不同 prompt、图像、proprio、seed 的完整动作输出逐元素相同。compile 版三组对照最大绝对误差为 0.0078125 / 0.01171875 / 0.01171875，通过 rtol=0.02、atol=0.01 的检查；所有输出无 NaN/Inf。额外 CPU CrossAttention 测试覆盖 batch 1/2 及带 mask，缓存前后逐元素相同。Python compileall 和 git diff --check 通过。

尚未验证 LIBERO 任务成功率，编译数值差异对策略表现的影响未知。模型原训练/其他 variant 的完整测试未运行。此目标是模型稳态 latency，不含机器人 IO。

没有找到 hipblaslt-bench，因此暂未做 kernel 选型；context 缓存加现有 compile 已达到 latency 目标，没有继续改写 RoPE、尝试 fused SDPA 或跨任务 T5 缓存。

## 优化后 profiler

profiler/fastwam.trace.json 和 roofline_summary.txt / roofline_by_op.txt 已生成。GPU kernel 总时间从旧 trace 的 522.19 ms 降到 362.29 ms；GEMM 从 380.60 ms 降到 282.59 ms，elementwise 从 79.12 ms 降到 31.69 ms。kernel 数从 36,577 降到 14,103。编译/graph 路径导致多数 kernel 缺少可映射 shapes，roofline 有效覆盖仅 28.2%，不能把新报告的估算下限当作下一轮可靠目标。

## 复现

```bash
bash benchmark_results/optimization/run_optimized.sh
```

详细测速与正确性验证：

```bash
conda activate rocm-nightly
cd /home/haojiema/code/amd_robot/models/FastWAM
python benchmark_results/optimization/benchmark_opt.py --label baseline_repeat --no-cache
python benchmark_results/optimization/benchmark_opt.py --label cache_repeat --verify
python benchmark_results/optimization/benchmark_opt.py --label compiled_repeat --compile --verify
python benchmark_results/profiler/profile_fastwam.py --optimized
```

baseline.json、context_cache.json、context_cache_compile.json、final_repeat.json 保存每次原始 latency；*_correctness.json 保存输出误差检查。profile_optimized.log 保存优化后 trace 采集记录。
