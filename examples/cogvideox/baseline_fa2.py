import torch
import time
import os

dtype = torch.bfloat16

print('Loading CogVideoX-5b...')
t0 = time.time()

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    'THUDM/CogVideoX-5b',
    torch_dtype=dtype,
)
pipe.enable_model_cpu_offload()

load_time = time.time() - t0
print(f'Loaded in {load_time:.1f}s')

prompt = 'A golden retriever running through a sunlit meadow with wildflowers, cinematic quality, 4K'
print(f'Generating (FA2 baseline): {prompt}')

t0 = time.time()
with torch.no_grad():
    video = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=6.0,
        num_inference_steps=30,
        generator=torch.Generator('cpu').manual_seed(42),
    ).frames[0]

gen_time = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024**3
print(f'Done in {gen_time:.1f}s | Peak VRAM: {peak:.1f} GB')

os.makedirs(os.path.expanduser('~/cogvideo_output'), exist_ok=True)
export_to_video(video, os.path.expanduser('~/cogvideo_output/5b_baseline.mp4'), fps=8)
print('Saved to ~/cogvideo_output/5b_baseline.mp4')
