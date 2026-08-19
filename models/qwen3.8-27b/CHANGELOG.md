# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo.

## 2026-08-19 — DFlash2 drafter (multi4 INT4): 6-piece vllm#52816 port; 256K is the honest ceiling

New `vllm/qwen38-27b-multi4-dflash2` compose (TP=4, AutoRound INT4 W4A8, fp8 KV,
gmu 0.80, **max-model-len 262144**) with the **incoai/Qwen3.8-27B-DFlash2** drafter
(block-diffusion "Keep Drafting Parallel", n=7) — the DFlash2 sibling of
`multi4/autoround-int4/mtp.yml`.

**Why a patch overlay, not a drop-in.** The DFlash2 candidate selector is only exercised by
the **V2 speculator**. On the V1 runner the same checkpoint silently degrades to the base
DFlash drafter and weight-load fails with `no module named candidate_selector`. The
oceanplexian/vllm fork's PR #1 shipped only the model file; the full upstream PR is
**vllm-project/vllm#52816 (11 files)**. The `vllm-dflash2` overlay ports the 6 load-bearing
pieces (model file + base-class refactor + registry entry + `DFlash2Speculator` + the
`init_speculator` dispatch + the `use_v2_model_runner` force), all anchor-checked +
idempotent, hard-failing boot on anchor drift so a re-pinned image can't serve a half-wired
drafter.

**The drafter works.** Under active generation the engine logs a healthy speculative-decoding
curve: **mean acceptance length 3.03, 29% draft acceptance**, per-position
0.799/0.541/0.316/0.191/0.115/0.038/0.029 — a real, decaying acceptance distribution, better
than the MTP sibling's 2.62 on this model. (The `bench.sh` "AL 2.00 / 14.3%" reading is an
**idle-window artifact** — the drafter only reports a real acceptance length while requests are
actively generating; bench.sh's sampler caught it idle.)

**Memory finding (measured on a 4× RTX 3090 rig, INT4 target).** The DFlash2 codebooks
(~0.5 GiB/GPU) + 5-layer backbone eat the headroom a large-context KV cache leaves. The OOM
is an **activation-buffer** failure (scales with `--max-num-batched-tokens`, not max context),
so the fix is headroom, not a patch change:

| config | KV pool | free at boot | result |
|---|---|---|---|
| 512K ctx / gmu 0.90 | — | ~0.5 GiB | first ~8K prefill **OOM'd the engine (exit 137)** |
| 256K ctx / gmu 0.85 | 1,014,705 tok (11.3 GiB) | ~1.8 GiB | prefill survives |
| **512K ctx / gmu 0.80** | **1,049,336 tok (10.85 GiB)** | **~3 GiB** | survived the prefill ladder (32K/65K/130K/260K) with no OOM |

**Why the compose ships 262144, not 524288 — the 512K number is a trap.** The 512K figure
is a YaRN-free position-embedding extension, but the KV pool + vLLM's mamba-hybrid scheduler
cap a **single sequence** far below it. The `verify-stress` ceiling ladder measured the real
cliff on this config:

- **~244,470 tok → recalled the needle (PASSED)**
- **~275,000 tok → DROPPED** — the engine threw a **CUDA device-side assert** in the
  mamba-hybrid/FlashInfer attention path when it rejected the overfull sequence, then
  auto-restarted.

That assert is a **vLLM robustness wart** (a graceful reject would be the correct behavior),
**not a bug in this port**. So the practical single-sequence ceiling is ~244K, and shipping
`max-model-len=524288` would just invite anyone to boot it blind and hit the assert on any
sequence past ~244K. **262144 is the model's native `max_position_embeddings`** — the honest,
safe default. The ~1.05M-token pool at gmu 0.80 holds ~4 full 256K sequences.

**Status: 🧪 Experimental — drafter + prefill-stability validated; TPS not yet benched.**
The open merge gates (to be filed via the numbers-from-your-rig template): `bench.sh` TPS
(acceptance length is measured), `report.sh` rig report, `verify-full`, `verify-stress` 7/7,
`soak-continuous`. Prefill stability + AL are proven; the TPS/decode-quality numbers are the
remaining item. See [Issue #1064](https://github.com/noonghunna/club-3090/issues/1064) and
[PR #1060](https://github.com/noonghunna/club-3090/pull/1060).
