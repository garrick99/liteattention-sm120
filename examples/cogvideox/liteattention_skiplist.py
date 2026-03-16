import torch
import torch.nn.functional as F
import time
import os
import sys
import types
import importlib.util

dtype = torch.bfloat16
LA_ROOT = os.path.expanduser('~/liteattention-sm120')

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
exec(open(os.path.join(LA_ROOT, 'hopper/__init__.py')).read(), pkg.__dict__)
LA = pkg.LiteAttention
print('Loaded')

# Per-CALL skip list: each invocation within a step gets its own skip list
# keyed by a call counter within the step, not by shape
call_in_step = [0]
step_idx = [0]

# skip_lists[call_index] = (skip_tensor, phase)
skip_lists = {}
la_calls = 0
la_time = 0.0

_orig = F.scaled_dot_product_attention

def la_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    global la_calls, la_time

    if query.dim() != 4 or query.dtype != torch.bfloat16 or not query.is_cuda:
        return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    B, H, S, D = query.shape
    if D > 256 or D < 64 or attn_mask is not None:
        return _orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()

    ci = call_in_step[0]
    call_in_step[0] += 1

    # First 2 steps: no skip list
    if step_idx[0] < 2:
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, causal=is_causal)
        la_time += time.perf_counter() - t0
        la_calls += 1
        return out.transpose(1, 2)

    # Get or create per-call skip list
    if ci not in skip_lists:
        sl = LA.init_skip_list(B, (S, S), H, D, False, torch.bfloat16, q.device, None, False)
        skip_lists[ci] = [sl, 0]

    sl_data = skip_lists[ci]
    sl, phase = sl_data[0], sl_data[1]

    if phase == 0:
        read, write = sl[0], sl[1]
    else:
        read, write = sl[1], sl[0]

    t0 = time.perf_counter()
    out = flash_attn_func(q, k, v, causal=is_causal,
                          attn_read_list=read, attn_write_list=write,
                          thr=-2.0, reverse_skip_list=False, phase=False)
    la_time += time.perf_counter() - t0
    la_calls += 1

    sl_data[1] = 1 - phase
    return out.transpose(1, 2)

def step_cb(pipe, step_index, timestep, callback_kwargs):
    step_idx[0] = step_index
    call_in_step[0] = 0  # reset per-step call counter
    return callback_kwargs

print('Loading CogVideoX-5b...')
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-5b', torch_dtype=dtype)
pipe.enable_model_cpu_offload()

F.scaled_dot_product_attention = la_sdpa
print(f'Patched (per-layer skip lists)')

prompt = 'A golden retriever running through a sunlit meadow with wildflowers, cinematic quality, 4K'
print(f'Generating: {prompt}')

t0 = time.time()
with torch.no_grad():
    video = pipe(
        prompt=prompt, num_frames=16, guidance_scale=6.0,
        num_inference_steps=30,
        generator=torch.Generator('cpu').manual_seed(42),
        callback_on_step_end=step_cb,
    ).frames[0]

gen_time = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f'Done in {gen_time:.1f}s | Peak VRAM: {peak:.1f} GB | LA calls: {la_calls}')
print(f'Unique skip list instances: {len(skip_lists)}')
if la_calls > 0:
    print(f'LA avg: {la_time/la_calls*1000:.2f}ms | Total: {la_time:.2f}s')

os.makedirs(os.path.expanduser('~/cogvideo_output'), exist_ok=True)
export_to_video(video, os.path.expanduser('~/cogvideo_output/5b_perlayer.mp4'), fps=8)
print('Saved to ~/cogvideo_output/5b_perlayer.mp4')
F.scaled_dot_product_attention = _orig
