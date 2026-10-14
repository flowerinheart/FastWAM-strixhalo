# Strix Halo benchmark artifacts

See the repository [README](../README.md) for ROCm setup, weight downloads, and reproduction commands.

- [Optimization measurements and validation](optimization/README.md)
- [Baseline profiler and roofline analysis](profiler/analysis.md)
- [Initial 10-iteration eager measurements](results.json)
- [Validated environment](environment.json)

Raw traces, model outputs, logs, and diagnostic scripts remain local. The original measurements predate the context-cache optimization; use `--no-cache` to reproduce the original eager configuration.
