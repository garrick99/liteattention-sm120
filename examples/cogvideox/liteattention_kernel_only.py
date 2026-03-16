import torch
import torch.nn.functional as F
import time
import os
import sys
import types
import importlib.util

dtype = torch.bfloat16
LA_ROOT = os.path.expanduser('~/liteattention-sm120')

# Minimal LA load — just flash_attn_func
pkg = types.ModuleType('lite_attention')
pkg.__path__ = [os.path.join(LA_ROOT, 'hopper')]
sys.modules['lite_attention'] = pkg
spec = importlib.util.spec_from_file_location('lite_attention._C', os.path.join(LA_ROOT, 'hopper/_C.abi3.so'))
cmod = importlib.util.module_from_spec(spec)
sys.modules['lite_attention._C'] = cmod
spec.loader.exec_module(cmod)
pkg._C = cmod
ipkg = types.ModuleType('lite_attention._internal')
ipkg.__path__ = [os.path.join(LA_ROOT, 'hopper/_internal')]
sys.modules['lite_attention._internal'] = ipkg
spec2 = importlib.util.spec_from_file_location('lite_attention._internal.flash_attn_interface',
    os.path.join(LA_ROOT, 'hopper/_internal/flash_attn_interface.py'))
iface = importlib.util.module_from_spec(spec2)
sys.modules['lite_attention._internal.flash_attn_interface'] = iface
spec2.loader.exec_module(iface)
flash_attn_func = iface.flash_attn_func
print('LA kernel loaded')

la_calls = 0
_orig = F.scaled_dot_product_attention

def la_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    global la_calls
    if query.dim() != 4 or query.dtype != torch.bfloat16 or not query.is_cuda:
        return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    B, H, S, D = query.shape
    if D > 256 or D < 64 or attn_mask is not None:
        return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    out = flash_attn_func(q, k, v, causal=is_causal)
    la_calls += 1
    return out.transpose(1, 2)

print('Loading CogVideoX-5b...')
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-5b', torch_dtype=dtype)
pipe.enable_model_cpu_offload()

F.scaled_dot_product_attention = la_sdpa
print('Patched. Generating...')

prompt = 'A golden retriever running through a sunlit meadow with wildflowers, cinematic quality, 4K'
t0 = time.time()
with torch.no_grad():
    video = pipe(
        prompt=prompt, num_frames=16, guidance_scale=6.0,
        num_inference_steps=30,
        generator=torch.Generator('cpu').manual_seed(42),
    ).frames[0]
gen_time = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f'Done in {gen_time:.1f}s | Peak VRAM: {peak:.1f} GB | LA calls: {la_calls}')

os.makedirs(os.path.expanduser('~/cogvideo_output'), exist_ok=True)
export_to_video(video, os.path.expanduser('~/cogvideo_output/5b_clean_la.mp4'), fps=8)
print('Saved to ~/cogvideo_output/5b_clean_la.mp4')
F.scaled_dot_product_attention = _orig
