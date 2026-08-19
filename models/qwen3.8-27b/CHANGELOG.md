# Qwen3.8-27B — Changelog

Dated history for Qwen3.8-27B configs in this repo.

## 2026-08-19 — DFlash2 drafter (multi4 INT4): 6-piece vllm#52816 port, full gate green

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

**The drafter works — and well.** Under active generation the engine logs a strong
speculative-decoding curve: **mean acceptance length 4.68 / 7.20 / 6.85** (53–91% draft
acceptance), per-position decaying 0.79→0.39 / 1.00→0.70 / 1.00→0.62 — well above the MTP
sibling's 2.62 on this model. (The `bench.sh` "AL 2.00 / 14.3%" reading is an **idle-window
artifact**: the drafter only reports a real acceptance length while requests are actively
generating; bench.sh's sampler caught it idle. The agentic-bench and verify-full MTP-AL
checks, which run under load, show the real figures.)

**Full gate — all green (report.sh --full, 4× RTX 3090 Turbo, 220 W cap):**

| stage | result |
|---|---|
| verify-full | **9/9 PASS** (MTP-AL 4.40) |
| verify-stress | **7/7 PASS** — ceiling ladder **all 6 rungs to 240,635 tok (91% of 256K), 0 MiB VRAM growth** |
| soak-continuous | **PASS** — 0 MiB growth, 0/25 silent-empty, 100% TPS retention, p50 decode 280 |
| bench | narrative **97.9** / code **190.2** decode TPS (CV 3.6% / 6.4%); prefill **1869 @10K / 1612 @90K**; 0 MiB leak |

**Memory finding (measured).** The DFlash2 codebooks (~0.5 GiB/GPU) + 5-layer backbone eat
the headroom a large-context KV cache leaves. The OOM is an **activation-buffer** failure
(scales with `--max-num-batched-tokens`, not max context), so the fix is headroom, not a
patch change:

| config | KV pool | free at boot | result |
|---|---|---|---|
| 512K ctx / gmu 0.90 | — | ~0.5 GiB | first ~8K prefill **OOM'd the engine (exit 137)** |
| 256K ctx / gmu 0.85 | 1,014,705 tok (11.3 GiB) | ~1.8 GiB | prefill survives |
| **256K ctx / gmu 0.80** | **919,797 tok (10.24 GiB)** | **~1.8 GiB** | **full gate green; 3.51× concurrency** |

**Why the compose ships 262144, not 524288 — the 512K number is a trap.** The 512K figure
is a YaRN-free position-embedding extension, but the KV pool + vLLM's mamba-hybrid scheduler
cap a **single sequence** at ~240K (91% of 256K) — the verify-stress ceiling ladder filled
cleanly to 240,635 tokens with 0 MiB growth and no crash at 256K. **262144 is the model's
native `max_position_embeddings`** — the honest, safe default; the ~920K-token pool at gmu
0.80 holds ~3.5 full 256K sequences. (An earlier 512K/0.80 boot hit a CUDA device-side assert
on a 275K sequence — a vLLM robustness wart in the overfull-reject path, not a port bug;
256K sidesteps it entirely.)

**Interconnect:** 2× NVLink pairs (0,2)/(1,3) + PCIe P2P on the rest; vLLM custom-AR
auto-off (its NVLink-mesh gate at world>2, #786) so NCCL handles the all-reduce over the
hybrid — a healthy, expected state, not a misconfiguration. PCIe decode peak was 39% of link,
so the interconnect is not the bottleneck.

**Status: 🧪 Experimental — full functional/stress/soak/bench gate green; 8-pack quality
open** (benchlocal not installed on this rig). The drafter is proven (AL 4.68–7.20), the
256K ceiling is understood and documented, and every stability gate passes. See
[Issue #1064](https://github.com/noonghunna/club-3090/issues/1064) and
[PR #1060](https://github.com/noonghunna/club-3090/pull/1060).
