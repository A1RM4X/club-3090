# Qwen3.8-27B on SGLang — W4A8 + DFlash2 (the SGLang side of the engine A/B)

SGLang is a **second engine** for Qwen3.8-27B here, alongside the primary
vLLM path (`models/qwen3.8-27b/vllm/`). It exists so the same model +
AutoRound-INT4 weights + DFlash2 drafter can be served by SGLang and
benchmarked head-to-head against the vLLM sibling — the single most useful
cross-engine comparison for a heavy-hybrid (48 linear + 16 full-attention)
model.

> **Why this is different from the Qwen3.6-27B SGLang tree (which is
> PARKED).** The 3.6 tree was parked because *EAGLE-3 external drafter was
> structurally slower than the model's built-in MTP head on Qwen3-Next*, and
> SGLang v0.5.12 had a CUTE_DSL Ampere capture-hang. **Neither applies to
> this 3.8 tree:** the drafter is **DFlash2, which is *upstream* in SGLang
> v0.5.19** (no EAGLE patch, no CUTE_DSL workaround), and the whole point is an
> **engine A/B**, not a "is SGLang better than MTP" question. So this tree is
> a live experimental path, not a parked archival one.

## TL;DR

| Variant | Path | Status |
|---|---|---|
| Quad 3090 (TP=4) DFlash2 + W4A8 | [`compose/multi4/autoround-int4/dflash2-w4a8.yml`](compose/multi4/autoround-int4/dflash2-w4a8.yml) | 🧪 Experimental — community-validated on a 4× 3090 Turbo rig (bench in [`results/sglang-q38-ar-w4a8-dflash2-tp4-20260909/`](../../../results/sglang-q38-ar-w4a8-dflash2-tp4-20260909/)). **SGLang wins c=1 decode +43%** vs the vLLM sibling; gap narrows to ~5% by c=8; SGLang TTFT lower at 4K+ (5.0s vs 8.6s @16K). SpecDecode (`--enable-metrics`): accept_len=**5.65 tok/step**, rate=**66%**. |
| SGLang AutoRound W4A8 patch (v0.5.19) | [`patches/sglang-autoround-w4a8-v0.5.19/`](patches/sglang-autoround-w4a8-v0.5.19/) | Re-cut of jb-seo's v0.5.18 W4A8 patch + the **shape-guard fix** (this PR's real fix). See its README. |

---

## Quick recipe (quad 3090, NVLink-paired)

```bash
# 0. Prereqs (must already exist on the host):
#    a. AutoRound INT4 target with the in_proj_ba data fix (see patch README).
#    b. DFlash2 drafter:   hf download incoai/Qwen3.8-27B-DFlash2 \
#                             --local-dir <MODEL_DIR>/incoai_Qwen3.8-27B-DFlash2
#    c. (Optional) a pinned chat template at <MODEL_DIR>/... or a local path.

# 1. Pull the pinned image (v0.5.19 = the DFlash2 release).
docker pull lmsysorg/sglang:v0.5.19

# 2. Boot. Point TARGET/DRAFT at your weights; NVLink rigs set NCCL_P2P_LEVEL=NVL.
cd models/qwen3.8-27b/sglang/compose
NCCL_P2P_LEVEL=NVL \
  TARGET=<host path to the AutoRound INT4 ckpt dir> \
  DRAFT=<host path to the DFlash2 ckpt dir> \
  docker compose -f multi4/autoround-int4/dflash2-w4a8.yml up -d

# 3. Verify (see the compose header for the full checklist):
#    - "[w4a8] applied + verified OK"
#    - per-rank "[Marlin] W4A8 prepared: ... arch=sm86" (many, 0 skips)
#    - "DFlash2DraftModel" + draft runner ready
#    - "KV Cache is allocated. dtype: torch.float8_e4m3fn"
#    - "Mamba Cache is allocated. max_mamba_cache_size: 84"
#    - NO "capped to ... by the mamba state cache" line
```

**NVLink is rig-specific.** The bench rig has 2× NVLink pairs (0,2)/(1,3) and
sets `NCCL_P2P_LEVEL=NVL`; the maintainer reference rig is PCIe-only and must
**not** copy that. The compose does not set it by default — it is an explicit
env the operator opts into.

---

## What the bench shows (2026-09-09, 4× 3090 Turbo, 220 W cap, fp8 KV)

Full raw output (SGLang + the paired vLLM baseline) is in
[`results/sglang-q38-ar-w4a8-dflash2-tp4-20260909/`](../../../results/sglang-q38-ar-w4a8-dflash2-tp4-20260909/).
Same model / quant / drafter (DFlash2), same FP8 KV, TP4, 262K ctx, 220 W —
**only the engine and drafter depth (vLLM 7 → SGLang 8 draft tokens) differ**,
so the decode delta is a genuine engine comparison.

- **c=1 decode: SGLang wins +43%** — 144.6 narrative / 244.8 code tok/s vs the
  vLLM 100.8 / 185.0. Single-stream is where SGLang's DFlash2 drafter +
  overlap-scheduling scheduler is most efficient (ITL p50 21.9 ms vs vLLM ~31
  ms at c=1).
- **The win narrows as concurrency rises** — +43% at c=1 → +4%/+6% by c=8 →
  +11%/+10% at c=16. vLLM's batching scales up harder; the crossover is
  between c=4 and c=8.
- **Prefill: raw TPS slightly lower** (SGLang 1683–1875 vs vLLM 1758–1910;
  −2% to −17%, worst at 1K); **but SGLang TTFT is lower at 4K+** (1.5s vs
  2.1s @4K; 2.7s vs 4.3s @8K; **5.0s vs 8.6s @16K**) — speculative prefill
  overlap reduces user-visible latency at large contexts even though the
  raw prefill throughput is marginally lower.
- **No saturation knee at c=16 on either engine** — combined decode still
  climbs c=8→c=16 (narrative 315→535, code 620→974 tok/s). That is a property
  of the DFlash2 drafter on this TP4 box, not the engine.
- **SpecDecode (Prometheus, `--enable-metrics`):** spec_accept_length =
  **5.65 tok/step**, spec_accept_rate = **66.0%** (54 accepted / 56 drafts
  over 6 requests in the 6.5 s window). Consistent with the log-level
  `accept len ≈ 4.0–5.7` range observed throughout the run.

**Caveats:** c=1 code decode is short-completion (646–800 tok actual vs 800
target; the "write quicksort" prompt doesn't always fill max_tokens — one run
stopped at 646 tok, inflating the mean). c=8 codeln TTFT (1069 ms) and c=16
codeln TTFT (2356 ms) are single-outlier rounds (median is fine). The prior
09-08 run had no `--enable-metrics`; **this 09-09 run does** — the caveat is
resolved.

---

## Why SGLang at all, and what it is NOT

- **It is not "SGLang is faster, switch."** It is an *engine A/B on the same
  workload* so the community can see which engine suits which concurrency
  shape. The practical read: SGLang is the better **single/low-concurrency**
  engine for this model; vLLM is the better **high-concurrency / heavy
  batching** engine. Pick by your `c`.
- **It does not replace the vLLM prod path.** The vLLM `dflash2.yml` sibling
  (with its 6-piece DFlash2 overlay) remains the primary 4-card path; this is
  the SGLang twin for comparison and for anyone who prefers the SGLang
  scheduler / its `--preferred-sampling-params` / thinking-kwargs ergonomics.
- **Engine pinning:** pinned to `lmsysorg/sglang:v0.5.19` because we vendor a
  patch (the W4A8 re-cut) into the container. A rolling tag would drift the
  `auto_round.py` anchor the moment upstream touches it. Bump via PR with a
  re-cut of the patch + a re-bench (AGENTS.md "Engine image pinning").
