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
| Quad 3090 (TP=4) DFlash2 + W4A8 | [`compose/multi4/autoround-int4/dflash2-w4a8.yml`](compose/multi4/autoround-int4/dflash2-w4a8.yml) | 🧪 Experimental — community-validated on a 4× 3090 Turbo rig (bench in [`results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/`](../../../results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/)). **SGLang wins c=1 decode +45%** vs the vLLM sibling; vLLM closes the gap by c=8; prefill within 2–17%. |
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

## What the bench shows (2026-09-08, 4× 3090 Turbo, 220 W cap, fp8 KV)

Full raw output (SGLang + the paired vLLM baseline) is in
[`results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/`](../../../results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/).
Same model / quant / drafter (DFlash2), same FP8 KV, TP4, 262K ctx, 220 W —
**only the engine and drafter depth (vLLM 7 → SGLang 8 draft tokens) differ**,
so the decode delta is a genuine engine comparison.

- **c=1 decode: SGLang wins +45%** — 146.6 narrative / 267.4 code tok/s vs the
  vLLM 100.8 / 185.0. Single-stream is where SGLang's DFlash2 drafter +
  overlap-scheduling scheduler is most efficient (ITL p50 21.9 ms vs vLLM ~31
  ms at c=1).
- **The win narrows as concurrency rises** — +45% at c=1 → +6%/+4% by c=8 →
  +16%/+11% at c=16. vLLM's batching scales up harder; the crossover is
  between c=4 and c=8.
- **Prefill is a wash-to-slightly-slower** (−2% to −17%), flat 1K→16K, same
  profile as vLLM.
- **No saturation knee at c=16 on either engine** — combined decode still
  climbs c=8→c=16 (narrative 313→576, code 592→975 tok/s). That is a property
  of the DFlash2 drafter on this TP4 box, not the engine.

**Caveats (kept honest in the row + results dir):** the SGLang bench ran
*without* `--enable-metrics`, so server-side Prometheus accept-length deltas
were not captured (only the qualitative SGLang `accept len` log, ≈2.7–3.8);
and one c=16 narrative round hit a ~26 s prefill/scheduler stall that
inflates that one mean TTFT (median is fine). Re-running with
`--enable-metrics` (already the default in the compose) closes the first gap.

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
