# Qwen3.8-27B results, 2026-10-29

The 24-layer partial-FFN sidecar was the fastest tested passing configuration on an Apple M3 Ultra with 96 GiB RAM. This is warmed benchmark performance, not cold-start latency or a general production-workload claim.

## Accuracy

| Selection | Combined Core ML cases | Worst mean KL | Largest perplexity increase |
| --- | ---: | ---: | ---: |
| 20 layers | 12 passed | 0.000775301 | 0.05727% |
| 22 layers | 12 passed | 0.000852444 | 0.04554% |
| 24 layers | 12 passed | 0.000892147 | 0.04735% |

Limits remained mean KL <= 0.001 and perplexity ratio <= 1.01. Code/prose/math were tested at 512/2048 tokens; build-guide/schema-code at 512/2048/8192. Previously reserved prompts were reused, not fresh unseen evaluation data. A 28-layer combination failed the code limit. Individual GPU controls with small nonzero differences were accepted only when they met the same numerical limits; exact equality was diagnostic.

The 24-layer selection adds 44, 45, 48 and 50 to the validated 20-layer baseline. Layer 31 remains FP16; all other selected sidecars use 8-bit palettes.

## Overall performance against ordinary Metal

Five measured repetitions per Core ML configuration, generating 128 tokens. Metal was measured three times before and three times after; reference medians pool all six. Each process/length has an excluded warmup. Runs were sequential, using identical tokenized prompts. Text is repeated to reach each benchmark length.

| Selection | 2K prefill throughput gain | 8K prefill throughput gain | 2K total latency reduction | 8K total latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 20 layers | 5.65% | 6.06% | 3.26% | 4.93% |
| 22 layers | 6.22% | 6.77% | 3.56% | 5.47% |
| 24 layers | 6.76% | 7.26% | 3.88% | 5.83% |

| Prompt | Metal prefill | 24-layer prefill | Metal total | 24-layer total |
| --- | ---: | ---: | ---: | ---: |
| 2048 tokens | 6.7963 s | 6.3660 s | 11.2500 s | 10.8131 s |
| 8192 tokens | 28.0591 s | 26.1600 s | 32.6001 s | 30.6985 s |

Relative to 20 layers, 22 adds 0.54-0.68% prefill throughput and 24 adds 1.05-1.13%. The 24-layer configuration saves 0.437 s / 1.902 s per 2K / 8K request versus Metal. Decode speed is essentially unchanged, with zero decode offloads. Reference prefill drift was 0.065% at 2K and 0.025% at 8K. Small differences between configurations should also be read alongside the sample ranges in `summary.json`.

All Core ML configurations generated the same 128-token continuations as one another. At 2K they first diverged from Metal at token 86; at 8K all 128 tokens matched Metal. Passing the numerical screen does not guarantee identical text or broad task accuracy.

## Reproduction and provenance

The base was `be6e387b9f2fda9a6fa0ca59f9607f2b6bb21db2`. The original Metal executable and runtime libraries were used for the Metal reference. Core ML runs used the same compiler flags/libraries with only compiled-package initialization changed to reuse a cache; timed inference code was unchanged. Both cache-creation and cache-reuse checks reproduced the prior 20-layer code metrics exactly. Reusing the cache grew disk by about 0.6 MiB instead of recompiling packages for each process.

`results/2026-10-29/` contains per-repetition measurements, quality rows, input/output token IDs, prompts, package/source hashes, build flags, the final 52-layer screen, and the failed 28-layer check. Original home paths are normalized to `${HOME}`. Binaries, model packages, compiled caches and personal connection details are excluded.

Run `python3 examples/ffn-coreml/verify_results.py` to recompute medians, accuracy checks, offload counts, reference drift and token comparisons from this evidence. Build and asset-generation instructions are in [README.md](README.md).
