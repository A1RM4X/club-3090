# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo.

## 2026-08-19 — DFlash2 drafter (multi4 INT4): 6-piece vllm#52816 port, prefill-validated

New `vllm/qwen38-27b-multi4-dflash2` compose (TP=4, AutoRound INT4 W4A8, fp8 KV) with the
**incoai/Qwen3.8-27B-DFlash2** drafter (block-diffusion "Keep Drafting Parallel", n=7) —
the DFlash2 sibling of `multi4/autoround-int4/mtp.yml`.

**Why a patch overlay, not a drop-in.** The DFlash2 candidate selector is only exercised by
the **V2 speculator**. On the V1 runner the same checkpoint silently degrades to the base
DFlash drafter and weight-load fails with `no module named candidate_selector`. The
oceanplexian/vllm fork's PR #1 shipped only the model file; the full upstream PR is
**vllm-project/vllm#52816 (11 files)**. The `vllm-dflash2` overlay ports the 6 load-bearing
pieces (model file + base-class refactor + registry entry + `DFlash2Speculator` + the
`init_speculator` dispatch + the `use_v2_model_runner` force), all anchor-checked +
idempotent, hard-failing boot on anchor drift so a re-pinned image can't serve a half-wired
drafter.

**Memory finding (measured on a 4× RTX 3090 rig, INT4 target).** The DFlash2 codebooks
(~0.5 GiB/GPU) + 5-layer backbone eat the headroom a large-context KV cache leaves. The
OOM is an **activation-buffer** failure (scales with `--max-num-batched-tokens`, not max
context), so the fix is headroom, not a patch change:

| config | KV pool | free at boot | result |
|---|---|---|---|
| 512K ctx / gmu 0.90 | — | ~0.5 GiB | first ~8K prefill **OOM'd the engine (exit 137)** |
| 256K ctx / gmu 0.85 | 1,014,705 tok (11.3 GiB) | ~1.8 GiB | prefill survives |
| **512K ctx / gmu 0.80** | **1,049,336 tok (10.85 GiB)** | **~3 GiB** | **VALIDATED: survived the full prefill ladder (32K/65K/130K/260K) with no OOM** |

So the compose ships **512K / gmu 0.80** (the validated-stable default). At 512K the KV pool
is the binding constraint (~2 concurrent 512K sequences); drop `MAX_MODEL_LEN` to 262144 and
raise gmu toward 0.85 for more concurrency. A `MEMPROBE=1` probe (ships in the overlay) logs
the per-step peak transient activation so a contributor can size gmu to the measured need.

**Status: 🧪 Experimental — prefill-stability validated; decode-quality NOT yet benched.**
The open merge gates (to be filed via the numbers-from-your-rig template): `bench.sh`
(DFlash2 acceptance length + TPS + peak VRAM), `report.sh` rig report, `verify-full`,
`verify-stress` 7/7, `soak-continuous`. Prefill stability is proven; the acceptance-length /
TPS numbers are the remaining item. See [Issue #1064](https://github.com/noonghunna/club-3090/issues/1064)
and [PR #1060](https://github.com/noonghunna/club-3090/pull/1060).
