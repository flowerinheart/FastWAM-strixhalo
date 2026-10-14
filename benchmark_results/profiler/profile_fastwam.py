"""Profile one warmed FastWAM inference with shapes and CPU/GPU activity."""
import argparse,functools,json,logging,os,statistics,time
parser=argparse.ArgumentParser()
parser.add_argument("--optimized",action="store_true")
args=parser.parse_args()
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.profiler import profile, ProfilerActivity, record_function
logging.basicConfig(level=logging.INFO)
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent if not args.optimized else root/'benchmark_results/optimization/profiler'
out.mkdir(parents=True,exist_ok=True)
os.environ['DIFFSYNTH_MODEL_BASE_PATH']=str(root/'checkpoints')
os.environ['DIFFSYNTH_DOWNLOAD_SOURCE']='huggingface'
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
with initialize_config_dir(config_dir=str(root/'configs'),version_base='1.3'):
 cfg=compose(config_name='sim_libero',overrides=['task=libero_uncond_2cam224_1e-4','model.redirect_common_files=false','model.action_scheduler.infer_shift=5.0'])
model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda')
model.load_checkpoint(str(root/'checkpoints/fastwam_release/libero_uncond_2cam224.pt'))
model.eval()
torch.manual_seed(42)
kwargs=dict(prompt='pick up the object',input_image=torch.rand(1,3,224,448)*2-1,action_horizon=32,proprio=torch.randn(1,8),num_inference_steps=10,seed=42,rand_device='cpu',compile_action_infer=args.optimized,cache_action_context=args.optimized)
def infer():
 with torch.no_grad():return model.infer_action(**kwargs)
for i in range(3):
 infer();torch.cuda.synchronize();print('warmup',i+1,flush=True)
timings=[]
for i in range(5):
 torch.cuda.synchronize();start=time.perf_counter();infer();torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
print('unprofiled ms',timings,flush=True)
def annotate(obj,name,label):
 original=getattr(obj,name)
 @functools.wraps(original)
 def wrapped(*args,**kw):
  with record_function(label):return original(*args,**kw)
 setattr(obj,name,wrapped)
annotate(model,'encode_prompt','FastWAM/text_encoding')
annotate(model,'_encode_input_image_latents_tensor','FastWAM/vae_encoding')
annotate(model.mot,'prefill_video_cache_tensor','FastWAM/video_cache_prefill')
annotate(model,'_denoise_action_with_video_cache','FastWAM/action_denoise_step')
torch.cuda.reset_peak_memory_stats()
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=False,with_flops=True,profile_memory=True) as prof:
 with record_function('FastWAM/inference'):
  start=time.perf_counter();result=infer();torch.cuda.synchronize();profile_ms=(time.perf_counter()-start)*1000
prof.export_chrome_trace(str(out/'fastwam.trace.json'))
for key in ['self_cuda_time_total','self_cpu_time_total']:
 try:table=prof.key_averages(group_by_input_shape=True).table(sort_by=key,row_limit=60)
 except Exception:table=prof.key_averages(group_by_input_shape=True).table(sort_by='self_device_time_total',row_limit=60)
 (out/f'{key}.txt').write_text(table)
torch.save(result['action'],out/'action.pt')
meta=dict(unprofiled_ms=timings,unprofiled_median_ms=statistics.median(timings),unprofiled_mean_ms=statistics.mean(timings),profiled_ms=profile_ms,peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,action_shape=list(result['action'].shape),action_finite=bool(result['action'].isfinite().all()),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,hip=torch.version.hip,sdpa='math',dtype='bf16',warmup=3,profile_iterations=1,denoise_steps=10,compile=args.optimized,cache_action_context=args.optimized,record_shapes=True,profile_memory=True)
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2),flush=True)
