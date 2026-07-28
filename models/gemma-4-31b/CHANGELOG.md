# Gemma 4 31B — Changelog

Dated history for Gemma 4 31B profiles in this repository.

## 2026-07-28 — Official Google QAT W4A16 TP=4 profile ([#814](https://github.com/noonghunna/club-3090/issues/814))

Added `vllm/gemma-31b-multi-google-qat-w4a16`, using `google/gemma-4-31B-it-qat-w4a16-ct` on four RTX 3090s with stock vLLM v0.25.1, BF16 KV, native 262,144 context, and Google's official Gemma assistant drafter at MTP n=2.

The full n=1–4 sweep found n=2 best for this quant: versus no MTP, narrative/code decode changed −2.2%/+11.7% (~+4.7% equal-weight aggregate). Every arm passed 20/20 streamed two-tool calls, confirming the ParserEngine path from vLLM #45588 resolves the historical Gemma MTP parser failure on this pin.

Production gate on @Whamp's 4×3090 PCIe rig at 230 W/card:

- verify-full passed chat, tools, streaming tools, reasoning, and cascade checks;
- long-context recall passed at 257,544 prompt tokens with 1,213 MiB/card free;
- vision smoke correctly identified a blue circle on a red background;
- canonical decode: 80.61 narrative / 92.34 code TPS;
- 10K/90K prefill: 981/692 tok/s;
- continuous soak: 100/100 responses, zero errors, silent outputs, or VRAM growth;
- full 8-pack: 116/150 thinking-off and 122/150 thinking-on.

Promoted to `⚠️ Production w/ caveats`: it occupies all four GPUs, and vision plus a near-max text prompt remains unvalidated at the 1.2 GiB/card margin.
