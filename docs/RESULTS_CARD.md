# Results Card

A small, fixed format for sharing a config's **measured** results — in a discussion, an issue (see the [`numbers-from-your-rig`](../.github/ISSUE_TEMPLATE/numbers-from-your-rig.yml) template), a `learnings/` note, or a PR description.

It's the empirical counterpart to a compose's `Profile (at-a-glance)` header: the header declares *what the config is*, the Results Card reports *what it measured*. Three panels, always in this order — **Serving → Quality → Takeaways**.

> **v2 (2026-08-02).** The three panels and their order are unchanged from the original #420 format — this is an evolution, not a replacement, and a v1 card is still a valid card. What changed:
>
> - **Quality** adopts [**@henrykrinkle01**](https://github.com/henrykrinkle01)'s table from [#770](https://github.com/noonghunna/club-3090/issues/770): dispersion (Std / CV) and p50/p95 latency per pack, `repeat = N` stated up front, raw output kept in a collapsed block. Built on our pack taxonomy and canonical `/150` scale; the presentation improvements are his.
> - **Serving** carries the config facts that turned out to decide whether two cards are comparable at all — KV **quality class**, ctx/slot, power cap, the three-layer interconnect state, and what acceleration was actually engaged.
>
> Both are about *comparability*: a v1 card told you a number, and left you unable to tell a real delta from noise, or to know whether the other rig's `q4_0` number belonged in the same table as your `turbo4` one.

## When to use it

Any time you post serving/quality numbers for a `(model, engine, topology, spec-dec, KV)` config. If you're A/B-ing one knob (thinking on/off, KV format, drafter `n`, …), use two value columns in the Quality table and bold the winner per row.

## Template

````markdown
### ① Serving — <engine + pin>, <topology>

**Stack** · <engine> <pin> · <model> <weights quant> · KV **<codec> (<quality class>)** · ctx <N><, ctx/slot <M> when -np>1>
**Placement** · <N>× <GPU>, <split mode> · expert offload: <summary / none> · power cap <W> W
**Interconnect** · L1 driver P2P <granted/refused> · L2 NCCL <state> · L3 engine custom-AR <engaged/vetoed/off/n-a>
**Acceleration** · moe-cache <pool GB, steady hit %> | off · spec-dec <drafter, draft-N, accept %> | off
**Concurrency envelope** _(when probed)_ · knee N=<n>, aggregate <t/s> @knee

| Config | Spec-dec | KV / ctx | decode TPS (narr / code) | TTFT | VRAM / card |
|--------|----------|----------|--------------------------|------|-------------|
| <model + quant> | <MTP / DFlash / ngram / EAGLE / none> (draft, n=) | <k/v quant> · <max ctx> | <narr> / <code> | <ms> | <GB> |

_(engine-internal decode TPS; 3 warm + 5 measured; temp/top-k/top-p; image/pin; tracking issue. Note any spec-dec accept-rate or balance caveat.)_

### ② Quality bench, thinking <on|off>, benchlocal-cli v<X.Y.Z>, repeat = <N>

| Pack | Pass / Total | Score | Std | CV | p50 latency | p95 latency | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| toolcall-15 (v1.0.1) | a / b | c% | d% | e | f s | g s | ok |
| instructfollow-15 (v1.0.0) | a / b | c% | d% | e | f s | g s | ok |
| structoutput-15 (v1.0.0) | a / b | c% | d% | e | f s | g s | ok |
| dataextract-15 (v1.0.0) | a / b | c% | d% | e | f s | g s | ok |
| reasonmath-15 (v1.0.0) | a / b | c% | d% | e | f s | g s | ok |
| bugfind-15 (v1.0.1) | a / b | c% | d% | e | f s | g s | ok |
| hermesagent-20 (v1.0.0) | a / b | c% | d% | e | f s | g s | ok |
| cli-40 (v1.0.2) | a / b | c% | d% | e | f s | g s | ok |
| **TOTAL** | **A / B** | **C%** | | | | | |

**Equivalent to: X/150**

<details>
<summary>Raw data</summary>

```
<the full benchlocal-cli run output, verbatim>
```

</details>

**Optional reasoning/code packs** _(on top of the core 8-pack — kept separate so /150 stays intact)_:

| Pack | Pass / Total | Score | Std | CV | p50 | p95 | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| humaneval-plus-30 | a / b | c% | d% | e | f s | g s | ok |
| lcb-v6-30 | a / b | c% | d% | e | f s | g s | ok |
| aider-polyglot-30 | a / b | c% | d% | e | f s | g s | ok |

### ③ Takeaways

- **<headline verdict>** — the single most important finding (bold lead).
- <comparison / tradeoff bullet>
- <production / stability note>
- _tl;dr — one line._
````

**A/B card?** Keep the v1 two-value-column shape for the Quality table (`| Pack | setting A | setting B |`, bold the winner per row) — the dispersion columns are for characterising *one* config; an A/B is about the delta between two. State `repeat = N` for both arms either way.

## Rules that keep cards comparable

- **The core 8-pack is exactly `/150`:** `toolcall-15 + instructfollow-15 + structoutput-15 + dataextract-15 + reasonmath-15 + bugfind-15 + hermesagent-20 + cli-40` (75 + 75). **Never fold the optional packs into the 150** — `humaneval-plus-30` / `lcb-v6-30` / `aider-polyglot-30` go in their own table below. (They're 30-sample subsets; compare the A/B **delta**, not absolute % against fuller external runs.)
- **Spec-dec is its own column** — MTP / DFlash / ngram / EAGLE / none. Don't rely on the config name to carry it; the same config name can run different spec methods.
- **State the n** (warm/measured runs for TPS; `n=` for quality). Don't present a single run as ranked truth — pack noise is ±5–7, so a small total delta is a tie.
- **Reproduce the conditions** in the Serving footnote: sampling params, engine image/pin, and `thinking on/off` (benchlocal sends `enable_thinking=false` unless you pass `--enable-thinking`).

### Quality table (v2)

- **`repeat = N` is always stated**, in the heading, even when it is 1. At `repeat = 1` the **Std / CV cells render `—`** — same columns, honestly labelled, so tables stay diffable across posts and nobody has to guess whether a blank means "zero dispersion" or "not measured".
- **`repeat ≥ 3` is encouraged for headline and promotion posts.** It triples 8-pack runtime, so it is *encouraged, not gated*. The reason to want it: pack noise is ±5–7 points, and a single run gives you no way at all to tell a real regression from that. A CV column is how a reader sees which packs are stable (`toolcall` at 0.00) and which are not (`hermesagent` at 0.14) — and therefore which rows a small delta is even meaningful on.
- **p50 / p95 latency per pack**, from the per-item timings benchlocal already records. p95 is the one that matters for an agent workload: a pack whose p50 is 1.3 s and p95 is 12.9 s is a different serving experience from one flat at 4 s, and the mean hides it completely.
- **Raw output retained** in a collapsed `<details><summary>Raw data</summary>` block. Receipts complete, headline scannable — and the per-item `✗ verifier_fail` lines are what make a claimed score auditable by someone who wasn't there.
- **The `TOTAL` row and the `Equivalent to: X/150` footer are mandatory.** Packs run at `repeat = N` total to `N × 150`, which is not comparable to anything else on the rig; the `/150` normalisation is what makes it a cross-rig number.
- **`verifier_fail` means the MODEL got it wrong**, not that the grader misfired. Take the pack result at face value; don't relabel failures as harness noise in the Takeaways.

### Serving section (v2)

Measurements live in the tables; this block is **config and steady-state facts only**. Each line exists because its absence has made two cards non-comparable:

- **KV codec with its quality class** — `turbo4 (KLD 0.0009, NIAH-validated)` vs `q4_0 (speed-only)`. A sub-q8 KV number is a *speed exhibit*, not a serving result, and a card that prints only the codec name invites the reader to compare the two.
- **ctx, plus ctx/slot when `-np` > 1.** A 262K server running 8 slots is a 32K-per-request server, and "262K ctx" alone reads as the opposite.
- **Placement: GPUs, split mode, expert-offload summary, and the power cap.** The cap is not a footnote — a measured +18% has turned out to be a power artifact rather than a config win on this rig.
- **Interconnect, all three layers** (driver P2P grant / NCCL use / engine custom-AR). Auto-filled by `bench.sh`'s `=== Interconnect (three layers) ===` footer (#805) — copy it off the artifact rather than transcribing it from a different run. Note that `engine-VETOED` is an *expected* state at >2 PCIe GPUs, not a misconfiguration.
- **Acceleration state, or an explicit `off`.** moe-cache (pool GB + steady hit %), spec-dec (drafter, draft-N, acceptance %). **Acceptance without its fire rate is not a number** — a drafter at 0.992 acceptance that fired on 5 of ~20 requests contributed almost nothing.
- **Concurrency envelope when probed** — knee N and aggregate tok/s at the knee. `rebench-full.sh` runs the rungs (#805).

**Canvas-granularity (dLLM) models:** put `wall_TPS` in the throughput cells and write `decode: n/a (canvas granularity)`. A `decode_TPS` from such a run is either a divide-by-noise artifact or `wall_TPS` wearing the wrong label — see [`BENCH_CARD.md`](BENCH_CARD.md).

## Posting to a public discussion/issue

- **No internal paths or secrets** — grep your draft for absolute host paths, model-store paths, and tokens before posting.
- **Don't link files that aren't on a public branch** (experimental/untracked composes 404 — describe them in prose, or link the image tag instead).
- **Verify every link resolves** (repo paths via `git ls-tree`, image tags via `docker manifest inspect`).

## Worked examples

- **v2 Quality table** — [#770](https://github.com/noonghunna/club-3090/issues/770), Qwen3.6-27B fp8 dual-max on 2× 3090 @ 250 W, `repeat = 3`. The reference implementation this format was adopted from ([**@henrykrinkle01**](https://github.com/henrykrinkle01)). Note what the dispersion columns buy: seven packs at CV 0.00 and `hermesagent-20` at 0.14 — the one row where a small delta means nothing.
- **v1 card** — [club-3090 discussion #221](https://github.com/noonghunna/club-3090/discussions/221#discussioncomment-17140596), Qwen3.6-27B Q8 on beellama v0.3.0 DFlash, thinking ON-vs-OFF. The original #420-era format; still a valid card.
