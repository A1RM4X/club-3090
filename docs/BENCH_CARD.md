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

### Integrity
- [ ] wall-TPS scrape agrees with engine-side timings (±5%) — if not, say which number is used and why
- [ ] CVs within rig norm (decode <5%, prefill <2%)
- [ ] quiet box: no builds / downloads / other GPU work during measured runs
- [ ] cache state steady across measured runs (a warming cache understates the mean)

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
| **resource cost** (pool slots / KV headroom / VRAM) | 8887 | 7554 | **−15.0%** | the hidden cost |

**Invariants (must match across arms or the A/B is confounded):**
cache hit% __ / __ · VRAM __ / __ · ctx __ / __ · <anything the knob shouldn't touch>

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
