# llama.cpp club-3090 test-bed — MoE expert cache

**Engine id:** `llamacpp-club3090` · **Status:** 🧪 Experimental · **Requires:** NVIDIA Ampere (SM 8.6) or newer

---

## ⚠️ Read this first: what this is, and what it isn't

This is a **test bed, not a product.** It is a llama.cpp fork where promising MoE-serving work from
different authors runs in **one binary, on one rig, under one measurement harness** — so the ideas can
be compared against each other instead of against a pile of incompatible forks.

Upstreaming is **each author's call, under their own name.** We are not a vendor for anyone else's work.

**The headline feature is not ours.** The MoE expert cache — hot CPU-resident experts held in spare
VRAM — is **[@leloch](https://github.com/leloch/llama.cpp)'s** design and implementation, proposed in
[ggml-org/llama.cpp#24528](https://github.com/ggml-org/llama.cpp/discussions/24528). It is **not in
mainline llama.cpp.** If you find it valuable, that credit belongs to leloch, and any upstream PR
carrying it comes from him or credits him prominently.

## TL;DR

If you are serving a **MoE larger than your VRAM** on consumer NVIDIA cards, the expert cache keeps the
experts your router actually reuses in leftover VRAM instead of re-reading them over PCIe every token.

On **2× RTX 3090**, DeepSeek-V4-Flash-0731 (UD-Q8_K_XL, 284B MoE), same session, canonical prompts:

| | stock `b10236` | this engine | Δ |
|---|---|---|---|
| **decode** (narrative) | 18.34 tok/s | **23.64 tok/s** | **~+29%** |
| **prefill** @10K | 436 tok/s | ~355 tok/s | ~−19% |

**You are trading prefill for decode.** That is the whole bargain, and whether it's a good one depends
entirely on your workload — see [Is this worth it for you?](#is-this-worth-it-for-you) below.

> The published **image** runs ~4% below a locally-built binary (CUDA 12.8 vs 13.2). The numbers above
> are **image** numbers, which is what you'd actually get.

---

## Is this worth it for you?

| Your workload | Verdict |
|---|---|
| **Long generations**, chat, reasoning, batch text | ✅ Clear win — decode dominates your wall-clock |
| **A MoE that already fits entirely in VRAM** | ❌ Pointless — there are no CPU-resident experts to cache |
| **Coding agent** re-sending a large stable prefix each turn | ⚠️ **Measure first** — you pay the prefill cost every turn and may net out negative |
| **Short prompts, short answers, latency-sensitive** | ⚠️ Prefill regression may dominate |

The agentic case is the one people get wrong. A 29% decode gain does not rescue a workload whose time
is mostly prefill.

---

## Hardware support

| Arch | Status |
|---|---|
| **Ampere SM 8.6** (3090, 3090 Ti, A5000…) | ✅ **Validated** — every number on this page |
| Ada SM 8.9 (4090) | ⚠️ **Built, never booted** |
| Blackwell SM 12.0 (5090) | ⚠️ **Built, never booted** |
| Below SM 8.6 | ❌ The cache gates itself off; no degraded path |

The image is compiled for `86;89;120`, but **we have no Ada or Blackwell hardware.** Those targets are
untested — if you run one, we'd genuinely like the numbers.

---

## Quick start

The engine ships as a digest-pinned image (digests don't move; tags do):

```
ghcr.io/noonghunna/llamacpp-club3090@sha256:17824d4483dbd60c297613bddf130f25e739d3edc7e633e4d973edc4e2165649
```

Two catalog slugs exist for DeepSeek-V4-Flash-0731:

```bash
# 2 cards (TP=2) — the validated one
bash scripts/switch.sh --force llamacpp-club3090/deepseek-flash-dual-q8-moecache

# 4 cards (TP=4) — constants inherited, see caveats
bash scripts/switch.sh --force llamacpp-club3090/deepseek-flash-multi4-q8-moecache
```

`--force` is required because the slug is `experimental`. That is deliberate, not an oversight.

**Building from source instead** — the branch is public:

```bash
git clone -b stack/club3090-moe-cache https://github.com/noonghunna/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --target llama-server -j
```

---

## The config that produced those numbers

```
--moe-cache auto          # grant free-minus-reserve per device
-devd none                # drafter on CPU, NOT on a GPU  ← load-bearing
-sm layer -ts 1,1
-ot 'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight=CPU'
-ub 2048 -b 2048
-t 28                     # ~= nproc/2
-c 204800
GGML_CUDA_MOE_CACHE_RESERVE_MB=1536
GGML_CUDA_MOE_CACHE_ADMIT_AFTER=64
```

Resulting cache state: ~5,159 experts resident across both cards, **~57% hit rate**.

---

## Tuning levers, in order of impact

### `-ub` — the biggest pool lever

Smaller ubatch frees compute-buffer VRAM, which becomes cache pool.

| `-ub` | decode | prefill | pool |
|---|---|---|---|
| 4096 | baseline | baseline | baseline |
| **2048** | **+10%** | **−12.5%** | **+1,301 experts** |

### `GGML_CUDA_MOE_CACHE_RESERVE_MB` — size it from the compute-buffer swing

The reserve is what the allocator keeps back to absorb the compute buffer's swing between graph shapes.
We measured that swing directly (**1,128 MiB** at `-ub 4096`/200K) and sized the reserve from it.

| reserve | result |
|---|---|
| 3072 (default) | baseline |
| **1536** | **~+6% decode**, +10pp hit rate — pool is bigger *and holds* |
| 1024 | **~11% SLOWER** — despite a bigger pool **and** a higher hit rate |

### `GGML_CUDA_MOE_CACHE_ADMIT_AFTER` — admission threshold

How many times an expert must be demanded before it earns a slot. Unset, admission is adaptive
(`1-complete/2-partial/8-replace`); setting it switches to **fixed**. We run **64** — when the pool
covers only a small fraction of total experts, a low bar lets one-off routing touches evict genuinely
hot residents. Tune it for your pool size; don't copy ours blindly.

(The companion `readmit_after` — the higher bar for evict-and-replace once the pool is full — is set by
`GGML_CUDA_MOE_CACHE_THROTTLE`, which doesn't match the field name.)

---

## ⚠️ Traps

**Never tune on hit rate.** It is a diagnostic, not the objective function. We have been burned by this
**three times** — most starkly at `RESERVE_MB=1024`, which showed a *bigger pool* and a *higher hit
rate* while running ~11% slower. Tune on wall-clock only.

**Never set `GGML_OP_OFFLOAD_MIN_BATCH` on a cache config.** It appears in several MoE-offload guides
and looks helpful. At `2` we measured **22.34 → 5.31 tok/s — a 4.2× loss.** It offloads the expert
`MUL_MAT_ID` ops to GPU, so no CPU-resident expert ops remain for the cache to intercept and **the cache
never allocates at all.** Unset (32) is already correct.

**`--moe-cache auto` has a 1 GiB minimum slab; an explicit budget does not.** If free-minus-reserve
can't clear 1 GiB, `auto` silently declines while `--moe-cache 24000` would have allocated fine. This
bites single-card users on very large models.

**The dead band at batches 9–31.** Above `MOE_CACHE_MAX_BATCH` (8) but below the op-offload threshold
(32), *neither* mechanism serves the batch. Harmless at `--parallel 1` with a small draft; a larger
draft or `--parallel 4` lands squarely in it. The lever is raising `MAX_BATCH`, not lowering `MIN_BATCH`.

> Every one of these shares a failure shape worth internalising: **the cache goes inert when something
> else claims its ops, and the only symptom is a missing log line.** If throughput looks like the cache
> isn't there, check that it actually allocated before you tune anything.

---

## What's in this fork

| Delta | Author | Upstream status |
|---|---|---|
| **MoE expert cache** | **[@leloch](https://github.com/leloch/llama.cpp)** | RFC [#24528](https://github.com/ggml-org/llama.cpp/discussions/24528) — **not merged** |
| L1 parallel scatter | club-3090 | offered to author |
| L3 D2H overlap | club-3090 | offered to author |
| No-budget warning + single-session gate | club-3090 | offered to author |
| Bypass warn-split | club-3090 | offered to author |
| Multi-row top-k batching | club-3090 | **interim** — drop when [ggml-org#26390](https://github.com/ggml-org/llama.cpp/pull/26390) lands |

---

## What is NOT validated

Stated plainly, because an experimental engine that hides its gaps is worse than no engine:

- **Quality has one measurement, and no same-model comparison arm.** We cannot claim the cache is
  quality-neutral. Proving that needs the identical 8-pack against stock `b10236`, which we have not run.
- **Ada and Blackwell are compiled but never booted.**
- **The 4-card slug has never run on 4 cards.** Its `RESERVE_MB` and `-ub` are *inherited from the
  2-card measurement*, not derived — per-card free VRAM differs at TP=4.
- **The reserve is tuned for 24 GB cards.** The launcher scales it upward on larger cards via a
  ratio-preserving heuristic that **no boot has exercised**. On a big card, sweep and pin
  `MOE_RESERVE_MB` explicitly.
- **Concurrency is unmeasured.** Everything here is `--parallel 1`, single stream.

---

## Upstream status

The cache is unmerged and under active discussion at
[ggml-org/llama.cpp#24528](https://github.com/ggml-org/llama.cpp/discussions/24528). Independent
validation has appeared from several rigs. If you run this, **post your numbers in that thread, not
here** — leloch's design benefits from the evidence, and the thread is where the design conversation
actually lives.

See [`../UPSTREAM.md`](../UPSTREAM.md) for the tracked row.
