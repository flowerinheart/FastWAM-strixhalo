"""Benchmark warmed FastWAM and compare cached/compiled outputs to eager."""
import argparse,functools,json,logging,os,statistics,time
parser=argparse.ArgumentParser()
parser.add_argument("--label",default="baseline")
parser.add_argument("--iters",type=int,default=50)
parser.add_argument("--verify",action="store_true")
parser.add_argument("--compile",action="store_true")
parser.add_argument("--no-cache",action="store_true")
args=parser.parse_args()
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
logging.basicConfig(level=logging.INFO)
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
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
kwargs=dict(prompt='pick up the object',input_image=torch.rand(1,3,224,448)*2-1,action_horizon=32,proprio=torch.randn(1,8),num_inference_steps=10,seed=42,rand_device='cpu',compile_action_infer=args.compile,cache_action_context=not args.no_cache)
def infer():
 with torch.no_grad():return model.infer_action(**kwargs)
if args.verify:
 checks=[]
 for seed,prompt in [(42,'pick up the object'),(13,'place the cup on the table'),(71,'open the drawer')]:
  kwargs['seed']=seed
  kwargs['prompt']=prompt
  kwargs['proprio']=torch.randn(1,8)
  kwargs['input_image']=torch.rand(1,3,224,448)*2-1
  kwargs['cache_action_context']=False
  kwargs['compile_action_infer']=False
  reference=infer()['action']
  kwargs['cache_action_context']=True
  kwargs['compile_action_infer']=args.compile
  cached=infer()['action']
  error=(reference-cached).abs().max().item()
  torch.testing.assert_close(cached,reference,rtol=0.02 if args.compile else 0,atol=0.01 if args.compile else 0)
  checks.append(dict(seed=seed,prompt=prompt,max_abs_error=error))
 (out/f'{args.label}_correctness.json').write_text(json.dumps(checks,indent=2))
 torch.manual_seed(42)
 kwargs.update(cache_action_context=not args.no_cache,seed=42,prompt='pick up the object',input_image=torch.rand(1,3,224,448)*2-1,proprio=torch.randn(1,8))
 print('correctness',checks,flush=True)
for i in range(3):
 infer();torch.cuda.synchronize();print('warmup',i+1,flush=True)
timings=[]
for i in range(args.iters):
 torch.cuda.synchronize();start=time.perf_counter();infer();torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000)
print('unprofiled ms',timings,flush=True)

result=infer()
assert torch.isfinite(result['action']).all()
torch.save(result['action'],out/f'{args.label}_action.pt')
ordered=sorted(timings)
meta=dict(label=args.label,timings_ms=timings,median_ms=statistics.median(timings),mean_ms=statistics.mean(timings),p95_ms=ordered[min(len(ordered)-1,int(len(ordered)*.95))],min_ms=min(timings),max_ms=max(timings),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,sdpa='math',denoise_steps=10,compile=args.compile,cache_action_context=not args.no_cache)
(out/f'{args.label}.json').write_text(json.dumps(meta,indent=2))
print(json.dumps({k:v for k,v in meta.items() if k!='timings_ms'},indent=2),flush=True)
