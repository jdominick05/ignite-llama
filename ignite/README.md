# ignite-llama

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) tuned for single-user decode of
**Qwen3.8 Flash-Next** (`qwen4exp`) on AMD RDNA3 with ROCm/HIP. It is upstream release **b11719** plus a stack of
patches, one commit each. Nothing here has been submitted upstream.

Model: [jdominick05/Qwen3.8-Flash-Next-REAP-288-44GB-262k-GGUF](https://huggingface.co/jdominick05/Qwen3.8-Flash-Next-REAP-288-44GB-262k-GGUF)
(FN-R1, 288-expert REAP prune, 63 GB). It also runs on upstream llama.cpp, more slowly.

## Results

FN-R1 on an RX 7900 XTX (24 GB) + RX 7900 XT (20 GB), layer split, Linux, TheRock ROCm 10.2.0a20261011.
Flags: `-sm layer -ts 530/470 -fa 1 -ctk q8_0 -ctv q8_0 -b 2048 -ub 256 -t 8`, GPU DPM level `high`.

| build | tg128 at d0 | tg128 at d64k |
|---|---|---|
| this branch with its RDNA3 kernel patches off (upstream b11719 kernels) | 36.3 tok/s | 30.1 tok/s |
| this branch | **58.7 tok/s** (+62%) | **43.4 tok/s** (+44%) |

Behind `llama-server` with Qwen Code, decode runs at the same speed: 57.6 tok/s on a short prompt, 43.7 tok/s at
55k context. Every step with its A/B is in [`perf/SCOREBOARD.md`](perf/SCOREBOARD.md), including the changes that
were measured and rejected.

**Correctness.** Each patch was checked with `test-backend-ops` (new test cases for the fused and megakernel
paths, plus a mutation check that each test actually runs the new path) and with a KL-divergence run against the
per-node build (`perf/fn-kld.sh`). The megakernels and fusions are bit-identical to the kernels they replace. The
MMVQ small-K change alters the reduction order, which is a 1-ulp-level difference.

## What changed

Decode of a 3B-active MoE is mostly launch overhead: about 2,900 kernels per token at the start, many under 2.5 us.
Most of the gain comes from launching fewer kernels.

- **Decode megakernels** (`ggml-cuda/megakernel.cu`). Persistent kernels with a grid barrier run whole graph
  segments as one launch:
  - the qwen4exp hyper-connection mixer;
  - the gated-delta-net (linear attention) middle segment;
  - the MoE router matvec + top-k;
  - the FFN tail (shared-expert gate, expert sum, residual);
  - the QSA attention mask chain.

  Each segment is matched in the CUDA graph loop and reuses the existing kernels' device code. The matcher falls
  back to the per-node path whenever a pattern doesn't match.
- **Fusions**: gated `hc_post`, scale -> unary, sigmoid -> mul -> add, and two MMVQ matmuls on one input as one
  launch (qkv + z, hc down + inject, shared-expert gate + up).
- **q8_1 reuse**: the quantized copy of a matvec input is cached within a graph, and the megakernels write it
  directly.
- **RDNA3 MMVQ**: rows per block for small-K matvecs (K = 320 / 640 in this model), and MoE MMVQ for 5-8 token
  verify batches.
- **Server and drafting**: context-checkpoint eviction that keeps checkpoints spread out, an `ngram-suffix`
  drafter, and an adaptive `ngram-mod` draft length.
- **qwen4exp graph**: launch-saving reorders. Optionally the n-gram (PLE) table can be stored as grouped super-rows
  so 256-block quant types fit.

## Knobs

All are on by default on RDNA3. Set the environment variable to `0` to turn one off.

| variable | turns off |
|---|---|
| `GGML_CUDA_MEGAKERNEL` | all decode megakernels |
| `GGML_CUDA_MEGAKERNEL_GDN` / `_FFN` / `_ROUTER` / `_SEQ` | one megakernel segment each |
| `GGML_CUDA_MMVQ_SMALL_K` | the RDNA3 small-K MMVQ geometry |
| `GGML_CUDA_Q8_1_CACHE` | the q8_1 input cache |
| `GGML_CUDA_MMVQ_PAIR` | two-matmul MMVQ launches |
| `GGML_CUDA_SIGMOID_MUL_ADD` | the shared-expert gate fusion |

`GGML_CUDA_MMVQ_MAX_BATCH=N` sets the dense MMVQ -> MMQ batch cutoff (8 helps drafter verify on this model).

## Build

Linux, ROCm with a gfx1100 GPU. The numbers above come from TheRock's ROCm 10.2.0a20261011 (AMD clang 24); older
ROCm compilers generated noticeably slower kernels for this fork.

```sh
R=/path/to/rocm   # TheRock tarball or a ROCm install
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=$R/llvm/bin/clang -DCMAKE_CXX_COMPILER=$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++
cmake --build build -j
```

## Serve

The model needs about 44 GB of VRAM at 262k context with q8_0 KV. Load with `-lm dio`: with mmap, a 63 GB file on a
machine with less RAM than that thrashes the page cache.

```sh
llama-server -m Qwen3.8-Flash-Next-REAP-288-FN-R1-00001-of-00002.gguf -lm dio -c 262144 \
  -dev ROCm0,ROCm1 -sm layer -ts 550,450 -ngl 99 --fit off -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 256 \
  --cache-ram 0 -ctxcp 8 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --spec-type ngram-mod
```

Put the card that drives your display second, and give it less of the model.

## Reproduce

- Decode A/B: `ignite/perf/fn-bench.sh <build> [KNOB=0 KNOB=1]` alternates two configs and reports repetitions
  2..N, because the first runs slow.
- Trace: `perf/fn-trace.sh` (rocprofv3 + kernel inventory).
- KLD: `perf/fn-kld.sh` (bring your own eval text).

## Notes

- **AI-assisted.** The patches were written with Claude (Anthropic) under my direction. Each commit carries an
  `Assisted-by` trailer, and each was benchmarked and tested before it was kept. Following llama.cpp's contribution
  policy, none of this is submitted upstream as-is.
- **License.** MIT, like llama.cpp. The model has its own license: Qwen Community License 1.0.
