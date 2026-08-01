# The bench.sh Result Card

The model-level 🎴 Results Card (see [RESULTS_CARD.md](RESULTS_CARD.md)) answers
*"what is this model on this rig?"*. A **bench card** answers a narrower question: *"what did THIS
`bench.sh` run measure, and can I trust it?"* Use one whenever a bench run backs a decision — a new
BENCHMARKS row, a config change, a pin bump.

Both templates share the same four-part contract:

1. **Fingerprint** — the exact config under test. A bench without its config is a rumor.
2. **Protocol** — warm/measured counts, sampling, date, power cap, engine pin.
3. **Numbers with CVs** — never a bare mean. A mean without spread can't be compared to anything.
4. **Integrity panel** — did the *measurement itself* work? Every checkbox below is a failure mode
   that has actually produced a plausible-but-wrong number on a real rig: a silent scrape failure
   reporting 0.00, a cache still warming so the mean understates steady-state, a background
   build/download stealing CPU or bandwidth mid-run, a CV quietly telling you the arms aren't
   comparable.

---

## Template 1 — SNAPSHOT (first bench of a new config)

*Use when a config gets its first canonical numbers. Feeds a BENCHMARKS row directly.*

```markdown
## 📊 bench.sh — <model> · <config-slug> · <date>

**Config:** <engine+tag> · <topology> · <quant> · <KV/ctx> · spec-dec: <drafter|none> · <key flags>
**Env:** <non-default env vars, verbatim>
**Protocol:** <N> warm + <N> measured per shape · temperature=0.6 top_p=0.95 top_k=20 · <power cap>

| shape | decode TPS | CV | wall TPS | TTFT |
|---|--:|--:|--:|--:|
| narrative (n=5) | 16.76 | 3.8% | — ⚠️ | 449 ± 12 ms |
| code (n=5)      | 16.42 | 3.8% | — ⚠️ | 543 ± 23 ms |

| prefill depth | tok/s | CV | TTFT@depth |
|--:|--:|--:|--:|
| 10K (n=2) | 127.2 | 0.1% | 81 s |
| 40K (n=2) | 91.7  | 0.5% | 7 m 17 s |

**GPU:** <VRAM/card> · <power> · <util>
**Cache (if the engine has one):** hits <start>% → <end>% (<cold | warming | plateaued>)

<!-- The block below is CONDITIONAL: include it only when the run offloaded to CPU
     (same gate the capture layer uses). It REPLACES the generic Cache line above.
     On a GPU-resident run every row here would be a dash pretending to be data.
     `bash scripts/bench.sh CARD=snapshot` fills all of it automatically. -->

### CPU offload + expert cache

**Offload:** detected via <argv -ot | --n-cpu-moe | boot log> · **RSS** <X> GiB / <Y> GiB total · swap <PASS/FAIL>
**Cache config:** cap=<MiB> · RESERVE_MB/ADMIT_AFTER/THROTTLE/MAX_BATCH=<…>

> The cache config **is** the arm identity — the pool self-limits below the requested cap, so
> pool size alone does not identify which arm a run was.

| device | pool (slots · MiB · type) | marginal hits | cumulative | evictions | skips† |
|---|---|--:|--:|--:|--:|
| CUDA0 | 3038 slots · 12911 MiB · mxfp4 | **60.3%** | 59.3% | 42171 | 13955 |
| CUDA1 | 3126 slots · 13285 MiB · mxfp4 | **80.4%** | 79.4% | 9158 | 2330 |

†`skips` = admission throttling (the insert queue exhausted under the configured `THROTTLE`) — a
**load signal, not a defect**. The hard-failure counters are fill / dispatch / collect-fail, and
those are on the integrity panel.

**PCIe** (gen<cur>/<max> ×<width> — ⚠️ if cur < max under load): decode rx <mean> / <peak> MB/s
(<%> of practical link) · prefill rx <mean> / <peak> MB/s · <note if one card dominates>

**RAM path:** derived miss demand ~<X> of ~<Y> GB/s ceiling (<Z>%) — **derived, not measured**
(misses × expert size ÷ elapsed). State the window: **decode-window** or **whole-run** (a whole-run
figure is diluted by prefill, during which the cache is not on the hot path).

### Integrity
- [ ] wall-TPS scrape agrees with engine-side timings (±5%) — if not, say which number is used and why
- [ ] CVs within rig norm (decode <5%, prefill <2%)
- [ ] quiet box: no builds / downloads / other GPU work during measured runs
- [ ] cache state steady across measured runs (a warming cache understates the mean)
- [ ] cache **hard-failure** counters zero (fill / dispatch / collect) — `skips` is load, not failure
- [ ] n-usable == n for every shape (a chat-tuned model EOSing early makes n=5 mean n=1)
- [ ] VRAM growth / leak delta — with an expert cache active this is **growth** (lazy pool + graph
      allocation on first inference); only a monotonic rise across REPEATED runs is a leak

**One-line verdict:** <what this run establishes — and what it must NOT be quoted for>
```

The example numbers above show why the integrity panel earns its place: that run's wall-TPS scrape
read 0.00 (the model answered canonical prompts briefly and stopped early, starving the token
counter), so the decode figures came from engine-side `print_timing` — a fact that belongs **on the
card**, not in whoever-ran-it's memory.

---

## Template 2 — A/B (the workhorse: one knob changed)

*Use when pricing a config change. The card IS the decision record.*

```markdown
## ⚖️ bench.sh A/B — <knob>: <arm A> vs <arm B> · <model> · <date>

**Held constant:** <everything else — engine tag, ctx, KV, drafter, env>
**Boot policy:** <same-boot | boot-per-arm ×N>   **Noise band:** ±<X>% (<how the band was established>)

**Pre-registered expectation:** <prediction written BEFORE the run — or "none", in honest ink>

| metric | A: <name> | B: <name> | Δ | verdict |
|---|--:|--:|--:|---|
| decode narr | 38.40 | 36.55 | −4.8% | outside band — real |
| decode code | 39.96 | 38.59 | −3.4% | outside band — real |
| TTFT short  | 1085 ms | 1604 ms | +47.8% | ⛔ decisive |
| prefill 10K | 902   | 1035  | +14.7% | real |
| **marginal hits** CUDA0 | 41.0% | 48.0% | +17.1% | outside band — real |
| **marginal hits** CUDA1 | 50.0% | 58.0% | +16.0% | outside band — real |
| **resource cost** (pool slots / KV headroom / VRAM) | 8887 | 7554 | **−15.0%** | the hidden cost |

<!-- The marginal-hits rows are CONDITIONAL on both arms having offloaded, and they are the rows
     that catch an under-warmed pool. Quote MARGINAL, never cumulative: a cumulative rate shows the
     BIGGER pool as worse (it spent longer cold-filling), so an A/B read off cumulative numbers
     rejects the better config. Marginal is measured over the same window in both arms. -->

**Invariants (must match across arms or the A/B is confounded):**

| field | A | B | |
|---|---|---|---|
| model / weights ftype | | | |
| served ctx | | | |
| KV cache type | | | |
| drafter | | | |
| moe-cache config (cap + RESERVE_MB / ADMIT_AFTER / THROTTLE / MAX_BATCH) | | | |
| CPU-offload method (`-ot` regex / `--n-cpu-moe`) | | | |
| serving argv (threads / ubatch / ngl / split) | | | |
| pool census per device (slots, ±5%) — only meaningful when the config above MATCHED | | | |

A **moe-cache config mismatch is a confound, exactly like a model mismatch** — it changes what is
being measured. If one of these differences *is* the knob under test, name it (`KNOB=<text>`) so it
is attributed rather than treated as an accident.

### Integrity
(the Template-1 checklist, once per arm)

**Decision:** <ADOPT / REJECT / PARK> — <one sentence, naming the decisive metric>
```

Three rows in this template exist because their absence has repeatedly cost real time:

- **Pre-registered expectation** — writing the prediction *before* the run is the only defense
  against retrofitting the story to the numbers. "None" is an acceptable entry; a prediction added
  afterward is not.
- **The resource-cost row** — config knobs that improve a headline metric by silently spending a
  persistent resource (KV headroom, an expert-cache pool, compute-buffer reservations) look like
  free wins until this row exists. Several have been adopted-then-reverted for exactly this.
- **The invariants row** — an A/B where a supposedly-constant value differs across arms isn't an
  A/B; it's two unrelated benches. Cheap to check, expensive to skip (a silently-clamped flag once
  produced three "arms" that measured identical configs).

---

**See also:** [CONTRIBUTING.md](../CONTRIBUTING.md) (bench + verify protocol) ·
[OFFLOAD_MATRIX.md](OFFLOAD_MATRIX.md) (multi-dimension sweeps — use its TSV output rather than
hand-running many A/Bs) · [RESULTS_CARD.md](RESULTS_CARD.md) (the model-level card these bench cards feed into).

---

## Rendering a card straight from the run

```bash
bash scripts/bench.sh                                   # no card (default, unchanged output)
CARD=snapshot bash scripts/bench.sh | tee run-A.log     # Template 1, ready to paste
CARD=ab BASELINE=run-A.log bash scripts/bench.sh        # Template 2 against that saved run
```

The card prints as a fenced markdown block at the end of the run. Everything the run can know is
filled in; everything that is human judgement — the one-line verdict, the A/B decision, the knob
name, "quiet box" — stays an explicit `<fill: …>` placeholder. **A card that guesses its own verdict
is worse than no card**, so the renderer will not do it.

A/B knobs: `KNOB=<text>` names the thing under test (differences matching it are attributed instead
of confounding the card) · `NOISE_BAND=<pct>` overrides the band (default: 2× the worst decode CV
across both arms, and the card says which) · `EXPECTATION="<text>"` fills the pre-registered
expectation · `BOOT_POLICY="<text>"` fills the boot-policy line.

The baseline is any previously saved `bench.sh` log. Parsing is strict: a log that is missing its
fingerprint or summary blocks fails with `baseline unparseable: <why>` rather than rendering half a
delta — and the run's own measurements are still printed.

## What `bench.sh` now fills in for you

`bench.sh` carries a capture layer (`scripts/lib/capture.sh`) that auto-fills most of the
**Fingerprint** and **Integrity** sections above. Copy these straight off the run instead of
reassembling them from memory afterwards — the reason the panel exists is that reconstructing it
later is exactly when the awkward facts get dropped.

| Card field | Where it comes from now |
|---|---|
| **Config** — KV, ctx, drafter, key flags | `CONFIG FINGERPRINT` block: KV cache type, served ctx + slots, weights ftype, drafter, and a `-m/-ot/-t/-ub/-ct/-ngl/-ts` argv fingerprint read from the serving process. No operator input needed. |
| **Env** — non-default vars | `moe-cache cfg` line: the `--moe-cache` cap plus `GGML_CUDA_MOE_CACHE_{RESERVE_MB,ADMIT_AFTER,THROTTLE,MAX_BATCH,STATS}`. The **resulting pool size does not identify the arm** — the census self-limits below the cap — so record the config, not just the census. |
| **GPU** — VRAM / util | `CAPTURE: VRAM` (idle / peak / post, per device) and `CAPTURE: PCIe` (sm%, memctl%, host cpu%). |
| **Cache** — hits start → end | `CAPTURE: EXPERT CACHE`, per device, with **marginal** and cumulative rates. |
| **resource-cost row** (A/B) | pool slots + total MiB per device, straight from the census. |

### Additions to the Integrity panel

Add these four lines to the checklist in both templates. Each is printed by the run.

- [ ] **`status: OK`** — the capture layer's own verdict (`NO_TOKENS` / `REQ_ERRORS` /
      `CACHE_DISABLED` / `INVALID_BYPASS` / `OK`). **`NO_TOKENS` means the run measured nothing** —
      a `0.00` TPS printed next to a healthy-looking TTFT is a scrape failure, not a slow model.
      Do not quote a throughput number from a non-`OK` run.
- [ ] **cache hard-failure counters zero** — `fill-fail` / `dispatch-fail` / `collect-fail`. Those
      are the real defect paths; any non-zero invalidates the cache reading for that run.
      **`skips` is NOT one of them**: it counts admission throttling (the insert queue exhausted
      under the configured `THROTTLE`), so under a low `ADMIT_AFTER` a non-zero value is normal
      steady-state pressure and a `THROTTLE` tuning signal. A live run on a healthy server showed
      skips at 0.5% of misses, tracking the per-device miss asymmetry exactly.
- [ ] **swap check PASS** — pages of the *serving process* in swap make every number in the run
      suspect, and nothing else in the pipeline notices. Needs `SERVER_PID` (auto-detected on bare
      metal).
- [ ] **n-usable == n per shape** — chat-tuned models EOS early, and a 5-run shape silently
      degenerates to n=1 while still printing `n=5`. When these differ the CV is not trustworthy;
      re-run with `FORCE_TOKENS=<n>`.
- [ ] **VRAM growth / leak delta** — labelled *growth* when an expert cache is active (its pool and
      graphs allocate lazily on first inference, so one run grows VRAM by design) and *leak*
      otherwise. Only a monotonic rise across REPEATED runs is evidence of a leak.

Two numbers on the card need their provenance stated, because both are easy to quote wrongly:

- **Cache hit rate — quote the MARGINAL one.** Cumulative rates embed the cold-fill phase, so a
  *bigger* pool reports a *lower* cumulative rate on the same traffic. Only the marginal rate across
  the measured window is comparable across boots. Keep the per-device split (a CUDA0/CUDA1
  asymmetry is routing-entropy signal; averaging it away hides it).
- **Draft acceptance — never without its fire rate.** A drafter measured at 0.992 acceptance that
  fired on 5 of ~20 requests contributes almost nothing end-to-end. The run prints both.

### Knobs worth setting

| Env | Why |
|---|---|
| `SERVER_LOG=<path>` | Bare-metal (`CONTAINER=none`) runs. Unlocks engine-side `print_timing` TPS, the client-vs-engine cross-check, expert-cache telemetry and drafter acceptance — all of which otherwise read "log scrape unavailable". |
| `ENDPOINT=chat` \| `completion` | `chat` (default) applies the model's template — the historical behaviour, so numbers stay comparable. `completion` drives raw `/v1/completions` with no template, for base models. Numbers are **not** comparable across modes. |
| `STREAM_CALIB=1` | ~3 s STREAM-triad ceiling on the host, so the derived miss-path RAM demand can be stated as a fraction ("~29 of ~99 GB/s"). Off by default. The *contention* probe (co-running STREAM during decode) is deliberately not included — it perturbs the run by construction. |
| `FORCE_TOKENS=<n>` | The fix when `n-usable < n`. |
| `CAPTURE=0` | Suppress the capture layer entirely (for a harness parsing the older output shape). |
