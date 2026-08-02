# Gemma 4 31B — on 2× or 4× RTX 3090

**Run [Gemma 4 31B](https://blog.google/technology/developers/gemma-4/) — with vision and tool calling — on stock vLLM.**

> ⚠️ **Single-card boot OOMs on 24 GB Ampere** regardless of KV format. Needs ≥32 GB single-card (validated on RTX 5090 by [@apnar](https://github.com/noonghunna/club-3090/discussions/67#discussioncomment-16832042)).

---

## Deployment

See [`docs/DUAL_CARD.md`](../../docs/DUAL_CARD.md) for workload-driven config picks. TL;DR:

| Config | Max ctx | Decode TPS | Best for |
|--------|---------|----------------|----------|
| `vllm/gemma-31b-dual` (default) ⭐ | **224K** | ~59 | General-purpose, vision + tools — overlay-free |
| `vllm/gemma-31b-multi-google-qat-w4a16` ⚠️ | **262K** | **80.6 / 92.3** narr/code | Official Google QAT + MTP n=2; TP=4, 257K recall-verified |

Run via:
```bash
bash scripts/launch.sh --variant vllm/gemma-31b-dual
bash scripts/switch.sh vllm/gemma-31b-multi-google-qat-w4a16
```

> **v0.24.0 consolidation (2026-07-02):** the TP=2 default remains one overlay-free **bf16** dual slug on
> `vllm-stable`. The v0.22.0 composes (`gemma-int8-mtp` = 262K int8-PTH + PR #40391, `gemma-bf16-mtp`
> = 131K, `gemma-mtp-tp1`, `gemma-31b-qat-w4a16-dual`) are **deprecated** (`switch.sh --list --all`).
> MTP is off on that v0.24.0 TP=2 profile — Gemma-4 MTP × tool-calling was broken there (vLLM #39043 / #42006). The 262K
> int8-PTH path returns overlay-free when PR #40391 merges upstream (on v0.24.0 int8-PTH allocates
> 262K but silently craters recall past ~32K, so bf16 @224K is the honest default).

---

## Models

- **Default target:** [`cyankiwi/gemma-4-31B-it-qat-AWQ-INT4`](https://huggingface.co/cyankiwi/gemma-4-31B-it-qat-AWQ-INT4)
- **Official QAT TP=4 target:** [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) (23.3 GB, vision preserved)
- **MTP draft:** [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) (0.5B / 927 MB BF16; active at n=2 on the TP=4 profile)

## Key details

| Aspect | Notes |
|--------|-------|
| **Quants** | cyankiwi QAT-AWQ INT4 (TP=2 default); official Google compressed-tensors W4A16 (TP=4) |
| **KV** | bfloat16: 224K on TP=2; native 262K on TP=4. The int8-PTH + PR #40391 path is deprecated |
| **Drafter** | TP=2: none on v0.24.0. TP=4/v0.25.1: official Gemma assistant MTP n=2, parser-clean via #45588 |
| **Vision** | ✅ Yes |
| **Tools** | ✅ `--tool-call-parser gemma4` |
| **NVLink** | Auto-detected via `NVLINK_MODE` env var |

## Upstream tracker

- [vLLM PR #41745](https://github.com/vllm-project/vllm/pull/41745) — Gemma 4 MTP support (merged)
- [vLLM PR #45588](https://github.com/vllm-project/vllm/pull/45588) — ParserEngine migration fixing Gemma MTP streaming/tool boundaries (merged; validated here with 80/80 streamed two-tool calls across n=1..4)
- [vLLM PR #40391](https://github.com/vllm-project/vllm/pull/40391) — INT8 PTH KV page-align (OPEN/unmerged). The deprecated `vllm/gemma-int8-mtp` (v0.22.0) vendors it for 262K; on v0.24.0 int8-PTH craters recall without it, so the default is bf16 (`vllm/gemma-31b-dual`). 262K int8-PTH returns when this merges.
- [Discussion #67](https://github.com/noonghunna/club-3090/discussions/67) — first Ampere consumer cross-rig data
