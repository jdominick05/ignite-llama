# Flash-Next FN-R1 decode scoreboard

Target: 150 tok/s delivered on real agent requests (plain decode is the headline; n-gram drafting is a bonus).
Ceiling with layer split: 4.92 GB/token, 5.6 ms at peak (178 tok/s), 6.6 ms at 85% (151 tok/s) (`bw_theory.py`).

Setup unless noted: Linux, TheRock 10.2.0a20261011, XTX (ROCm0) + XT (ROCm1), `-sm layer -ts 530/470 -fa 1
-ctk q8_0 -ctv q8_0 -b 2048 -ub 256 -t 8 -lm dio`, tg128, `-r 3`. Logs in `~/llm/logs`.

| date | build / change | d0 tok/s | d64k tok/s | ms/token d0 | kernels/token | log |
|---|---|---|---|---|---|---|
| 2026-10-21 | Windows b11719 ignite (reference) | ~32 | ~27 | 31.8 | ~3,400 / graph pair | handoff |
| 2026-10-22 | Linux base, graphs off | 30.67 | 25.55 | 32.6 | - | fn-linux-base-20261022-0016 |
| 2026-10-22 | Linux base | 33.69 | 27.08 | 29.7 | - | fn-linux-base-20261022-0016 |
| 2026-10-22 | + fusion-wip (uncommitted) | 35.01 | 27.93 | 28.6 | 3,460 | fn-linux-fuse-20261022-0022 |
| 2026-10-22 | + RDNA3 MMVQ small-K (75fcd6764) on base | 37.11 / 37.63 | 29.44 / 29.40 | 26.6 | - | fn-linux-smallk-ab-20261022-0039 |
| 2026-10-22 | fusion-wip + small-K | 38.75 | 30.14 | 25.8 | 3,460 | fn-linux-fuse-smallk-20261022-0048 |

**Quiet-machine rebaseline (2026-10-22 01:15, desktop idle).** Rows above were measured while the desktop was in use;
CPU contention cost 5-12% (decode is partly CPU/launch bound). Relative A/B results held. Use these as the reference:

| date | build / change | d0 tok/s | d64k tok/s | ms/token d0 | log |
|---|---|---|---|---|---|
| 2026-10-22 | base (96599b4a6, SMALL_K=0) | 36.27 / 36.29 | 30.11 / 30.14 | 27.6 | fn-bench-20261022-0114* |
| 2026-10-22 | + small-K (96599b4a6) | 40.92 / 40.99 | 33.26 / 33.27 | 24.4 | fn-bench-20261022-0114* |
| 2026-10-22 | fusion-wip, SMALL_K=0 | 37.91 / 37.96 | 31.22 / 31.27 | 26.3 | fn-bench-20261022-012* |
| 2026-10-22 | fusion-wip + small-K | 42.96 / 42.99 | 34.60 / 34.61 | 23.3 | fn-bench-20261022-012* |
| 2026-10-22 | fusion-wip + small-K, `-sm tensor` | 42.29 | 34.18 | 23.6 | fn-linux-smtensor-20261022-0106 |
| 2026-10-22 | fusion split into 3 commits (b83fb0afd, 9cafbca90, 00f411418) + small-K | 42.93 / 42.11 | 34.45 / 34.13 | 23.3 | q8_1-cache-validate-20261022-01* |
| 2026-10-22 | + q8_1 src1 cache (cb37a7edf) | 44.79 / 43.81 / 44.73 / 44.77 | 35.31 / 35.36 / 35.77 / 35.77 | 22.4 | q8_1-cache-validate, graph-validate |
| 2026-10-22 | + qwen4exp conv-state cpy, alpha bias (30863089f) | **45.18 / 45.19** | **36.05 / 35.78** | **22.1** | graph-validate-20261022-01* |

| 2026-10-22 | + shared-expert gate sigmoid->mul->add fused (79d54711c) | **45.76 / 45.78** | **36.44 / 36.38** | 21.8 | sma-validate2-20261022-0224 |
| 2026-10-22 | + GDN z expanded next to qkv (85865ffa2; noise-level, slow post-GDN slot only moves) | 45.83 / 45.83 | 36.46 / 36.45 | 21.8 | zorder-validate-20261022-0239 |
| 2026-10-22 | + RDNA3 MMID MMVQ up to batch 8 for IQ3_XXS/IQ4_NL (b2e7ac253; drafter verify only, tg unchanged) | 45.90 / 45.75 | 36.50 / 36.38 | 21.8 | fn-verify-mmid8, pair-validate |
| 2026-10-22 | + one MMVQ launch for two matmuls on one src1, hc down + inject (358b67523) | 46.37 / 46.33 / 46.38 | 36.80 / 36.77 | 21.6 | pair-validate-20261022-0350 |
| 2026-10-22 | + GDN qkv + z paired (one launch) | **46.83 / 46.83** | **37.04 / 37.00** | 21.4 | qkvz-validate-20261022-0414 |

A/B notes: 85865ffa2 passed the both-runs rule by 0.1% (noise level). 358b67523's first d0 candidate run was
disturbed (45.12 +- 2.31) and lost; a d0-only rerun won both (46.33 / 46.38 vs 45.83 / 45.84).
Drafting rows below are single runs of 8 agent requests at temp 1.0 whose outputs diverge between configs: the
agent-request deltas (+-1-4%) are indicative only; the rewrite gains (2-4x) are solid.

Kernels/token: 3,460 -> 2,991 (quantize_q8_1 737 -> 437). Commits are on local branch `ignite-llama-wip`.
Calibration: removing 96 launches gave 0.28 ms/token, ~2.9 us per launch; launch overhead is ~8-9 ms of the 21.8 ms
token. Both GPUs read `power_dpm_force_performance_level = high` (already set before these runs, mclk at max).

Checked and ruled out as noise sources: model page cache (cold vs warm after POSIX_FADV_DONTNEED: 42.87 vs 42.85,
0 major faults), GPU clocks (mclk at top state in both runs).

## Per-token budget (latest trace)

fusion-wip + small-K, `rocprof-fuse-smallk-d0-20261022-0048` (median per-call times):
kernels ~15.7 ms (mmvq ~10.3, other ~5.4), gaps ~10 ms. Top: Q6_K mmvq 3.28 ms (133 calls), Q8_0 mmvq 2.99 ms
(508 calls), IQ3_XXS gate+up 1.50 ms, IQ4_NL down 1.21 ms, quantize_q8_1 1.15 ms (737 calls).

## Delivered decode with drafting (Phase 6, 2026-10-22, 85865ffa2, `fn-spec.sh`, serve_bench: 3 synthetic + 8 real agent requests 8k-20k prompt, temp 1.0 seed 42)

| config | code / prose | rewrite (copy-heavy) | 8 real agent requests | all |
|---|---|---|---|---|
| none | 45.3 / 45.4 | 44.8 | **42.2** | 43.9 |
| ngram-mod (defaults, drafts up to 64) | 45.6 / 45.6 | **187.6** (87% accepted) | 40.6 (5-19% accepted) | 62.9 |
| ngram-mod + `GGML_CUDA_MMVQ_MAX_BATCH=8` | 45.5 / 45.5 | 187.7 | 41.4 | 64.7 |
| none (repeat) | 45.5 / 45.5 | 44.8 | 42.1 | 43.6 |

On ordinary agent replies ngram-mod loses 2-4% (low acceptance, rejected 64-token drafts cost a ~65-token verify);
on file rewrites it gives 4.2x. Plain decode is the main lever for agent turns; drafting pays on edit/rewrite
output. Log: `fn-spec-20261022-031617/`.

Draft length (`--spec-ngram-mod-n-max`, all with `GGML_CUDA_MMVQ_MAX_BATCH=8` and b2e7ac253, `fn-spec-20261022-033227/`):

| n-max | 8 agent requests | rewrite | all |
|---|---|---|---|
| none | 42.2 | 44.8 | 43.9 |
| 4 | **43.9** (+4%, 67-100% accepted) | 97.5 | 56.5 |
| 8 | 43.6 | 102.3 | 58.2 |
| 16 | 42.7 | 109.0 | 59.1 |
| 64 (default) | 41.3 | **187.7** | 62.3 |

Short drafts win on agent turns, long ones on rewrites: an adaptive length (grow while drafts are accepted, drop
to 4 on a miss) should get both. Until then: n-max 4-8 for agent work.

Adaptive length (`--spec-ngram-mod-n-start N`, n-max 64, `fn-spec-20261022-040156/`, single runs):

| config | 8 agent requests | rewrite | all |
|---|---|---|---|
| none | 42.7 | 45.4 | 44.4 |
| n-max 4 | 44.7 | 97.7 | 58.2 |
| **n-start 4** | **44.0** | **162.3** | **64.1** |
| n-start 2 | 43.4 | 159.2 | 61.9 |

Recommended FN serving: `--spec-type ngram-mod --spec-ngram-mod-n-start 4` with `GGML_CUDA_MMVQ_MAX_BATCH=8`.

## Rejected experiments

| date | change | result | record |
|---|---|---|---|
| 2026-10-22 | MMVQ quantizes src1 into LDS itself for K <= 640 (no quantize launch), bit-exact (0 differing bytes on 965 real-graph sites, KLD 0) | all small-K incl. MMID: tg 45.78 -> 43.89 (expert down 22.9 -> 41.1 us); without MMID: neutral at d0, -0.03 d64k. The per-block quantize prologue costs what the launch did | `~/llm/mmvq-y-f32-rejected.patch`, yf32-validate-20261022-0257 |
| 2026-10-22 | HIP runtime flags (DEV_KERNARG, PACKET_CAPTURE, DIRECT_DISPATCH, MAX_HW_QUEUES, SEGMENT_SCHEDULING, BATCH_SIZE, SDMA) | all within +-0.3% of base except GPU_MAX_HW_QUEUES=1 (-2.6%) | fn-flags-20261022-015248 |

## Verify cost (Phase 0.4, 2026-10-22, 85865ffa2, `-ub 256 -p k -n 0`, r=5)

| k | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| d0 ms/batch | 24.0 | 27.6 | 29.9 | 31.6 | 53.4 | 55.5 |
| d0 x k=1 | 1.00 | 1.15 | 1.25 | 1.32 | 2.23 | 2.31 |
| d64k ms/batch | 34.8 | 41.3 | 48.0 | 49.4 | 70.5 | 72.7 |
| d64k x k=1 | 1.00 | 1.19 | 1.38 | 1.42 | 2.03 | 2.09 |

A 4-token verify costs 1.32x one token at d0 (dense 27B was much steeper): drafting is a larger lever for FN than
the plan assumed. k=6/8 jump to ~2.2x: the RDNA3 MMVQ -> MMQ switch above batch 5 (and the MMID max batch); tune
`GGML_CUDA_MMVQ_MAX_BATCH` / MMID limits for FN's experts before tuning drafters. pp1 here (41.7 +- 8) is a noisy
prompt-path number, not tg. Log: `fn-verify-cost-20261022-0252.txt`.

Switch-point sweep (`fn-verify-sweep-20261022-0310.txt`, r=5, two base runs agree within 1%):

| config | pp6 d0 | pp8 d0 | pp6 d64k | pp8 d64k |
|---|---|---|---|---|
| base (dense MMVQ <= 5) | 112.5 | 142.5 | 86.0 | 110.0 |
| `GGML_CUDA_MMVQ_MAX_BATCH=8` | **148.5** | **177.9** | **105.6** | **132.2** |

FN wants dense MMVQ up to batch 8 (k=8 verify 55.5 -> 45.0 ms, 1.87x a single token), the opposite of the dense 27B
tuning the RDNA3 default came from. The MUL_MAT_ID limit cannot be raised by an env override: the MoE kernel's
`__launch_bounds__` come from the same table (override -> "unspecified launch failure").

## KLD noise floor (2026-10-22)

FN-R1 is extremely sensitive to rounding: a fused shared-expert kernel that differed from the unfused ops by 1 ulp
(HIP's `__fmul_rn` + add were contracted into an FMA; max |d| 3e-8) gave **mean KLD 0.026, same top 94.7%** on the
24-chunk eval2 set, above the plan's requant gate (KLD < 0.02, same top >= 97%). The QSA top-k pool selection and the
MoE top-k routing likely amplify tiny differences. Consequences:
- Kernel/graph changes must be bit-exact (gate: KLD ~0, same top 100%); use `#pragma clang fp contract(off)` in
  fused elementwise kernels.
- The Phase 4 requant gate must be re-based on this floor: compare a candidate quant against the spread of
  "1-ulp perturbed" FP runs, not against 0.02 absolute (or use a longer text and PPL as the primary metric).

## KLD at ub 256 does not see the decode path (2026-10-22)

`fn-kld.sh` defaults to `-ub 256`: matmuls go to MMQ, src1 is too big for the q8_1 cache slot, and the pair
fusion and mm epilogues need one row. So the KLD lines in the commit messages of cb37a7edf (q8_1 cache),
358b67523 (pair) and 19c07de29 (qkv+z pair), and the decode half of 30863089f, compared identical code paths.
Decode-path check: base from 00f411418 (before all of them) at `FN_KLD_UB=1 FN_KLD_CHUNKS=4`, compared with the
tip 19c07de29: mean KLD 0.000000, max 0.000055, same top 100% - identical to the control (00f411418 against its
own base: same max 0.000055, the harness floor at ub 1). All decode-path changes are exact.
Use `FN_KLD_UB=1 FN_KLD_CHUNKS=4 FN_KLD_BASE=~/llm/kld-base-fn-r1-ub1.bin` for decode-only changes.

## Trace caveat

rocprofv3 timestamps of the **second GPU (XT, Agent 2) are not usable**: 35,866 of 36,443 kernel records start before
the previous kernel on the same queue ends (one correlation id per graph launch). Call counts are fine; per-call
times, gaps and busy % of the XT are not. The XTX (Agent 1) records have no overlaps. XTX, profiled: busy 28%,
dispatch gaps 24% (~3.8 us/kernel), waiting on the XT 48% (`gapsplit.py`).

## Tools

- `fn-bench.sh <build> [KNOB=off KNOB=on]`: alternating A/B, d0 + d64k.
- `fn-trace.sh <build> [ENV=...]`: rocprofv3 trace + `kinventory.py` + `mmvq_bw.py`.
- `bw_theory.py <gguf>`: bytes/token and the bandwidth ceiling.
