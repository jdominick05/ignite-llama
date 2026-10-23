> **Status (2026-10-23): nice to have, not scheduled.** Written by the user as a handoff. Model work after the
> Flash-Next (qwen4exp) experiments targets **int8, concretely Q8_0** (the reference quant every comparison is made against), not FP8: gfx11 has native iu8 WMMA and `v_dot4_i32_iu8`, and
> no FP8 hardware, so int8 weights (and later int8 activations) get real compute from the silicon, while FP8 here is
> only a storage format decoded to fp16. Worth doing for loading FP8-native checkpoints as-is.

# Native FP8 (E4M3) weight support for RDNA3 (gfx11): spec

## Goal

Load FP8 E4M3 model weights and run them natively on RDNA3 (gfx1100, 7900 XTX / 7900 XT). Do not requantize: decode the FP8 bytes in registers, right before the math.

This is a W8A16 scheme:
- Weights are stored as FP8 E4M3.
- Activations stay fp16.
- Accumulation is fp32.

## Why

Most official checkpoints ship in FP8 (DeepSeek, Qwen FP8 variants, GLM, compressed-tensors/vLLM releases). Upstream llama.cpp has no native FP8 type. It converts FP8 checkpoints by dequantizing and requantizing them. The only native FP8 GGUF support found so far is a WIP fork:
- Fork: `nisparks/llama.cpp`, branch `wip/deepseek-v4-support`
- Type: `GGML_TYPE_F8_E4M3_B128`, ggml type id 42 in that fork

That fork targets NVIDIA. Nobody has a tuned gfx11 path.

## Hardware facts (gfx11)

- There is **no FP8 hardware**: no FP8 WMMA and no FP8 conversion instructions. That support arrived with RDNA4 (gfx12) and CDNA3.
- WMMA input types are fp16, bf16, iu8 and iu4. Use `v_wmma_f32_16x16x16_f16`, which takes fp16 inputs and accumulates in fp32.
- WMMA on gfx11 is wave32. The A and B operands are 16 halves per lane, and the inputs must be **replicated across both half-waves**: lanes 0–15 mirror lanes 16–31.
- Do **not** use ROCm's `__hip_fp8_e4m3` / `__hip_fp8_e5m2` types on gfx11. They fall back to generic, branchy software conversion and are slow.

## Format: E4M3FN

- Bit layout: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
- There is no infinity.
- The only NaN encodings are `0x7F` and `0xFF`.
- Max magnitude is 448 (`0x7E`).
- Subnormals: when exp == 0, the value is `mant/8 * 2^-6`.

## Core decode: 4× E4M3 → 2× half2

Output is the true value × 2^-8. The bias difference between fp16 (15) and E4M3 (7) is 8. The missing ×256 is folded into the scale later.

```cpp
__device__ __forceinline__ void e4m3x4_to_h2x2(uint32_t x, uint32_t& lo, uint32_t& hi) {
    // Put each byte in the high byte of a 16-bit lane. Should compile to v_perm_b32.
    uint32_t a = ((x & 0x000000FFu) << 8) | ((x & 0x0000FF00u) << 16);
    uint32_t b = ((x & 0x00FF0000u) >> 8) |  (x & 0xFF000000u);
    // Keep the sign bit; shift exponent+mantissa right by 1. There is no cross-lane bleed.
    // Should compile to v_bfi_b32.
    lo = (a & 0x80008000u) | ((a & 0x7F007F00u) >> 1);
    hi = (b & 0x80008000u) | ((b & 0x7F007F00u) >> 1);
}
```

Output order:
- `lo` = {byte0, byte1}
- `hi` = {byte2, byte3}
- Low half of each word = lower byte index.

Properties of this decode:
- **Lossless in fp16.** Decoded magnitudes fall in [2^-17, 1.75], all representable as fp16 normals or subnormals. E4M3 subnormals come out correct automatically.
- NaN (`0x7F`) decodes to 480/256 instead of NaN. This is acceptable for weights; document it.
- Check the ISA (`--save-temps`, or disassemble the code object). The target is about 3 VALU ops per half2. If the compiler emits worse code, use `__builtin_amdgcn_perm(0, x, 0x010C000C)` for `a` and `0x030C020C` for `b`.

Bonus path, E5M2 → fp16: E5M2 is exactly the top byte of an fp16, so the byte-placement step alone is the full decode. Apply no shift and no scale.

## Block format: F8_E4M3_B128

- 128 E4M3 bytes plus one E8M0 scale byte.
- E8M0 is a pure power of two: value = 2^(e − 127). `0xFF` = NaN.
- **TODO before implementing:** confirm the exact byte layout against the fork source: scale before or after the codes, struct packing and padding, and alignment. Read `ggml-common.h`, `ggml.c` type traits and `gguf-py` constants on that branch. Match it exactly so existing GGUFs load.

Folding the scale:
- Effective block scale = 2^(e − 127) × 2^8 = 2^(e − 119).
- Because it is a power of two, you can build it as an fp32 directly: `__uint_as_float((e - 119 + 127) << 23)` = `__uint_as_float((uint32_t)(e + 8) << 23)`.
- Guard `e == 0xFF` (NaN) and any overflow.

Also support compressed-tensors / safetensors FP8 checkpoints later:
- Scales there are per-tensor or per-channel fp32 `weight_scale`. DeepSeek-style checkpoints use 128×128 block scales.
- Pre-multiply by 256 at load time.
- **Ignore `input_scale`.** Activations stay fp16.

## Kernel structure

### GEMV (token generation, batch 1; bandwidth-bound — this is where the speedup is)

1. Load weights as wide vectors (u32x4 = 16 FP8 values) from global memory, coalesced.
2. Decode in registers.
3. Compute the dot product against fp16 activations with `v_dot2_f32_f16` (`__builtin_amdgcn_fdot2`) into an fp32 accumulator.
4. Every 128 K-elements, apply the block scale: `acc += partial * scale` in fp32. Reset `partial`.
5. Expected throughput is about Q8_0 speed, roughly 2× fewer bytes than fp16.

### GEMM (prefill)

1. Stage the FP8 tiles global → LDS. Keep them **as FP8 in LDS**, since that saves LDS capacity and bandwidth.
2. Decode LDS → registers right before the WMMA, producing the fp16 A/B fragments (with half-wave replication).
3. One 128-wide K block = 8 WMMAs (K = 16 each). Accumulate into a per-block fp32 temporary, then `acc = fma(tmp, scale, acc)`. Alternatively, apply the scale per output row if the scales are per-row.
4. Expect fp16-WMMA-class throughput. There is no compute speedup over fp16; the win is memory.

**Never** do a separate dequant-to-fp16 pass that writes to global memory. That destroys the bandwidth win.

## Validation (do these before any perf work)

1. **Exhaustive decode test.** For all 256 byte values, compare the decode output × 256 against a reference E4M3FN → fp32 decoder:
   - Expect exact equality, bit-for-bit, after conversion to fp32.
   - Skip `0x7F` and `0xFF`; assert the documented 480 value for them instead.
   - Include ±0, every subnormal, and 448.
2. **Packing test.** Random u32 inputs: verify the byte → lane order matches the documented layout.
3. **GEMV test.** Compare against a CPU fp32 reference:
   - Shapes: K ∈ {128, 4096, 14336}, N ∈ {1, 4096}.
   - Use random E8M0 scales in a sane range.
   - Pass criterion: max relative error ≲ 1e-3.
4. **GEMM test.** Same as the GEMV test, with M ∈ {16, 512, 2048}, including non-multiples of the tile size.
5. **End to end.** Load a real `F8_E4M3_B128` GGUF and compare perplexity / KL-divergence against the reference implementation's logits on a fixed prompt set.
6. **Perf.**
   - GEMV: report achieved GB/s versus the 7900 XTX peak (~960 GB/s).
   - GEMM: report TFLOPS versus the fp16 WMMA baseline.

## Non-goals (for now)

- FP8 activation quantization or FP8 KV cache.
- RDNA4 native FP8 paths.
- Converting FP8 to int8 for iu8 WMMA. It is possible but lossy and needs requantization; revisit later if prefill throughput becomes the bottleneck.
