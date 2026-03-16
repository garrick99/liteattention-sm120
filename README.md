# LiteAttention SM120 Port & Performance Report

**Date:** 2026-03-15
**Author:** garrick99
**Hardware:** NVIDIA RTX 5090 (SM120), WSL2 Ubuntu 24.04
**Codebase:** LiteAttention v0.4.0 + SM89 patches + SM120 port
**Build:** CUDA 12.8, PyTorch 2.10+cu128

---

## Executive Summary

LiteAttention is now fully operational on SM120 (RTX 5090) with both forward skip list and backward tile skipping. On video inference workloads with temporal redundancy:

- **7-17x faster per-step** than baseline FA3 (45-110us vs 784us)
- **Baseline forward: 204 TFLOPS** (49% of RTX 5090 tensor peak)
- **Output quality:** cos_sim 0.990 at threshold=-1.0 with 2% inter-step noise
- **Backward: 1,926us** (26% faster than SM89/RTX 4090)
- Includes all SM89 patches: backward tile skipping, skip list fixes, bug reports

---

## 1. SM120 Port — Architecture

SM120 (Blackwell consumer) differs significantly from SM90 (Hopper) and SM100 (Blackwell datacenter):

| Feature | SM80/SM89 | SM90 (Hopper) | SM100 (B200) | SM120 (RTX 5090) |
|---------|-----------|---------------|--------------|------------------|
| MMA Instruction | mma.sync | WGMMA | UMMA+tcgen05 | Extended mma.sync |
| TMA | No | Yes | Yes | No |
| WGMMA | No | Yes | No | No |
| TMEM | No | No | Yes | No |

**SM120 uses the SM8x HMMA code path**, not the SM90 TMA/GMMA path. Key changes:

### Build Configuration
- `compute_120,code=sm_120` — native SM120 compilation target
- `LITE_ATTENTION_DISABLE_SM80=FALSE` — enables SM80 kernel instantiations (compiled to SM120)
- `LITE_ATTENTION_DISABLE_BACKWARD=FALSE` — enables backward kernels
- CUDA 12.8 toolkit (matches PyTorch 2.10+cu128)
- `MAX_JOBS=4` — CUTLASS templates consume ~6-8GB RAM each

### Code Changes (SM120-specific, on top of SM89 patches)
1. **`static_switch.h`** — Added SM120 to ARCH_SWITCH: `if (ARCH == 86 || ARCH == 89 || ARCH == 120)` routes to SM8x code path
2. **`utils.h`** — Extended `enable_sm80_to_sm89` kernel guard to include SM120 (`__CUDA_ARCH__ >= 1200`). Without this fix, kernels were no-ops on SM120.
3. **`flash_api.cpp`** — All `params.arch >= 90` checks changed to `(params.arch >= 90 && params.arch < 120)` to exclude SM120 from SM90 TMA path
4. **`lite_attention.py`** — Python tile size dispatch: SM120 uses `get_tile_size_fwd_sm8x()` not SM90
5. **`setup.py`** — SM80 arch remapped from `compute_89` to `compute_120`

---

## 2. Inference Performance (Forward Only)

### Baseline — FA3 (no skip list)

| Config | LA Time | TFLOPS | SDPA Time | Speedup vs SDPA |
|--------|---------|--------|-----------|-----------------|
| B=1 S=1024 H=16 D=128 | 58 us | 147.2 | 57 us | 0.98x |
| B=1 S=2048 H=16 D=128 | 201 us | 171.1 | 215 us | 1.07x |
| B=1 S=4096 H=16 D=128 | 784 us | 175.4 | 845 us | 1.08x |
| B=1 S=8192 H=16 D=128 | 2,760 us | 199.2 | 2,970 us | 1.08x |
| B=2 S=4096 H=32 D=128 | 2,690 us | 204.3 | 2,888 us | 1.07x |

Peak: **204.3 TFLOPS** (49% of RTX 5090 tensor peak).

### Skip List — Video Inference Simulation

B=1 S=4096 H=16 D=128, threshold=-1.0, 2% Gaussian noise per step:

| Step | Time | Speedup | Cos Sim |
|------|------|---------|---------|
| 0 (warmup) | 21,328 us | 0.04x | 0.999980 |
| 1 (1st skip) | 756 us | 1.03x | 0.996645 |
| 2 (settled) | 110 us | 7.12x | 0.990105 |
| 3 | 56 us | 13.95x | 0.990105 |
| 8 (best) | 45 us | **17.33x** | 0.990105 |

**Settled per-step: 45-110us = 7-17x faster than baseline FA3.**

---

## 3. SM120 vs SM89 Comparison

| Metric | SM89 (RTX 4090) | SM120 (RTX 5090) | Improvement |
|--------|-----------------|------------------|-------------|
| Baseline forward (S=4096) | 889 us | 784 us | +14% |
| Skip list settled | 134 us (6.6x) | 45-110 us (7-17x) | 2-3x faster |
| Backward (S=4096) | 2,594 us | 1,926 us | +26% |
| Fwd+Bwd total | 3,499 us | 2,699 us | +23% |

---

## 4. Backward Tile Skipping

Carried forward from SM89 port. Block-sparsity support in SM80 backward kernel via `is_m_block_active()` bitmask check using `__ldg` (read-only L2 cache path).

**B=1 S=4096 H=16 D=128 (baseline, no block mask):**
- Backward: 1,926 us
- Forward + Backward: 2,699 us

---

## 5. Build Instructions

```bash
# WSL2 Ubuntu 24.04, CUDA 12.8, PyTorch 2.10+cu128
source ~/liteattention_env/bin/activate
cd ~/liteattention-sm120

export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

LITE_ATTENTION_DISABLE_SM80=FALSE LITE_ATTENTION_DISABLE_BACKWARD=FALSE MAX_JOBS=4 python setup.py build_ext --inplace
```

---

## 6. Files Modified (vs upstream v0.4.0)

| File | Change |
|------|--------|
| `setup.py` | SM120 arch target, INT8 skipable SM90 instantiations |
| `hopper/instantiations/flash_fwd_hdim*_int8_skipable_sm90.cu` | 5 new files |
| `hopper/_internal/cpp/static_switch.h` | ARCH_SWITCH: SM120 -> Arch=86 |
| `hopper/_internal/cpp/utils.h` | enable_sm80_to_sm89 extended for SM120 |
| `hopper/_internal/cpp/flash_api.cpp` | SM120 arch checks, block_mask param, SM8x tile size binding |
| `hopper/_internal/cpp/mainloop_bwd_sm80.hpp` | is_m_block_active() + skip checks |
| `hopper/_internal/cpp/mainloop_bwd_sm90_tma_gmma_ws.hpp` | Interface compatibility |
| `hopper/_internal/cpp/flash_bwd_launch_template.h` | Pass block_mask to mainloop |
| `hopper/_internal/cpp/flash.h` | block_mask fields in Flash_bwd_params |
| `hopper/_internal/cpp/mainloop_fwd_sm80.hpp` | Skip list range_first + deferred range_end fixes |
| `hopper/_internal/flash_attn_interface.py` | Autograd gradient count fix, block_mask flow |
| `hopper/lite_attention.py` | SM120 tile size dispatch |
