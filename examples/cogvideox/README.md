# CogVideoX Integration Examples

Real video model integration of LiteAttention with CogVideoX (THUDM/CogVideoX-2b and CogVideoX-5b).

## Scripts

### baseline_fa2.py
Standard CogVideoX with PyTorch SDPA (FlashAttention 2 backend). 
This is the reference for quality and timing comparison.

### liteattention_kernel_only.py
Drop-in replacement of SDPA with LiteAttention's flash_attn_func kernel.
No skip lists — mathematically equivalent to FA2. Used to verify kernel correctness.

### liteattention_skiplist.py  
Full LiteAttention integration with per-layer skip lists.
Each attention layer in the transformer gets its own skip list instance.
Skip lists are disabled for the first 2 denoising steps (composition phase)
and activated for the remaining steps (refinement phase).

## Key Findings

### Per-Layer Skip List Isolation
CogVideoX-5b has 42 attention layers, all with the same tensor shape (2, 48, 5626, 64).
Skip lists MUST be keyed per-layer (by call order within a step), not by tensor shape.
Sharing skip lists across layers produces garbled output because each layer has 
different attention patterns.

### SM120 reverse_skip_list Bug
The reverse skip list iteration path in mainloop_fwd_sm80.hpp produces incorrect 
results on SM120 (RTX 5090). Workaround: force reverse_skip_list=False on SM120.
Forward iteration works correctly.

### Threshold Tuning
- threshold=-2.0: Too aggressive. Fine details (fur, grass, small flowers) become blocky.
- threshold=-5.0: Good balance. Detail preserved, purple flower visible.
- threshold=-8.0: Conservative. Nearly identical to baseline.
- Omer's calibration tool (per-layer binary search) is the proper solution.

### Artifact Propagation
Skip lists can lock
