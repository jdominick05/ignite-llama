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

Kernels/token: 3,460 -> 2,991 (quantize_q8_1 737 -> 437). Commits are on local branch `ignite-llama-wip`.
Calibration: removing 96 launches gave 0.28 ms/token, ~2.9 us per launch; launch overhead is ~8-9 ms of the 21.8 ms
token. Both GPUs read `power_dpm_force_performance_level = high` (already set before these runs, mclk at max).

Checked and ruled out as noise sources: model page cache (cold vs warm after POSIX_FADV_DONTNEED: 42.87 vs 42.85,
0 major faults), GPU clocks (mclk at top state in both runs).

## Per-token budget (latest trace)

fusion-wip + small-K, `rocprof-fuse-smallk-d0-20261022-0048` (median per-call times):
kernels ~15.7 ms (mmvq ~10.3, other ~5.4), gaps ~10 ms. Top: Q6_K mmvq 3.28 ms (133 calls), Q8_0 mmvq 2.99 ms
(508 calls), IQ3_XXS gate+up 1.50 ms, IQ4_NL down 1.21 ms, quantize_q8_1 1.15 ms (737 calls).

## KLD noise floor (2026-10-22)

FN-R1 is extremely sensitive to rounding: a fused shared-expert kernel that differed from the unfused ops by 1 ulp
(HIP's `__fmul_rn` + add were contracted into an FMA; max |d| 3e-8) gave **mean KLD 0.026, same top 94.7%** on the
24-chunk eval2 set, above the plan's requant gate (KLD < 0.02, same top >= 97%). The QSA top-k pool selection and the
MoE top-k routing likely amplify tiny differences. Consequences:
- Kernel/graph changes must be bit-exact (gate: KLD ~0, same top 100%); use `#pragma clang fp contract(off)` in
  fused elementwise kernels.
- The Phase 4 requant gate must be re-based on this floor: compare a candidate quant against the spread of
  "1-ulp perturbed" FP runs, not against 0.02 absolute (or use a longer text and PPL as the primary metric).

## Trace caveat

rocprofv3 timestamps of the **second GPU (XT, Agent 2) are not usable**: 35,866 of 36,443 kernel records start before
the previous kernel on the same queue ends (one correlation id per graph launch). Call counts are fine; per-call
times, gaps and busy % of the XT are not. The XTX (Agent 1) records have no overlaps. XTX, profiled: busy 28%,
dispatch gaps 24% (~3.8 us/kernel), waiting on the XT 48% (`gapsplit.py`).

## Tools

- `fn-bench.sh <build> [KNOB=off KNOB=on]`: alternating A/B, d0 + d64k.
- `fn-trace.sh <build> [ENV=...]`: rocprofv3 trace + `kinventory.py` + `mmvq_bw.py`.
- `bw_theory.py <gguf>`: bytes/token and the bandwidth ceiling.
