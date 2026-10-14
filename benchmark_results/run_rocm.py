"""Run the upstream benchmark with math SDPA on ROCm and output checks.

The default fused SDPA path failed on Radeon 8060S with this nightly build.
Output validation and saving occur after the upstream timed region.
"""
import logging
logging.basicConfig(level=logging.INFO)
import importlib.util
from pathlib import Path
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

script = Path(__file__).resolve().parents[1] / 'scripts/dryrun_fastwam.py'
spec = importlib.util.spec_from_file_location('dryrun_fastwam', script)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
original = benchmark._timed_infer_with_output

def checked_infer(model, kwargs, device):
    elapsed, action = original(model, kwargs, device)
    if not torch.isfinite(action).all().item():
        raise RuntimeError('Action output contains NaN or Inf')
    torch.save(action.cpu(), Path(__file__).parent / 'last_action.pt')
    return elapsed, action

benchmark._timed_infer_with_output = checked_infer
from hydra import compose, initialize_config_dir
import sys
with initialize_config_dir(config_dir=str(script.parent.parent / "configs"), version_base="1.3"):
    cfg = compose(config_name="sim_libero", overrides=sys.argv[1:])
benchmark.main.__wrapped__(cfg)
