# SGLang #39342 Mixed-Chunk Mamba Radix-Cache Patch (v0.5.20)

A **fail-closed** SGLang engine patch applied before launch. This is the
**v0.5.20 re-cut** of the #39342 mixed-chunk mamba radix-cache fix: the unified-
cache file was renamed `mamba_component.py` → `mamba.py` between v0.5.19 and
v0.5.20, so the C/D anchors point at the new path. All five functional edits
(A1, A2, B2, C, D) are byte-identical to the v0.5.19 cut — only the target
path changed. Verified round-tripping clean in-image at v0.5.20.

> **Supersedes the v0.5.19 cut.** The v0.5.19 patch targeted
> `.../unified_cache/components/mamba_component.py`; v0.5.20 moved that module
> to `mamba.py`. If you are on a v0.5.19 image, use the older cut.

## Status

| Patch | Upstream ref | Status | Validation |
|-------|-------------|--------|------------|
| `apply_39342.py` | sgl-project/sglang#39342 | **VALIDATED** | 90/90 clean on 4×3090 TP4 (2026-09-15) on v0.5.19; re-cut applies verbatim to v0.5.20 (file rename only). The exact trigger that crashed the unpatched stack is neutralized. |

## What it fixes

`prepare_for_extend` stamps `req.kv.mamba_last_track_seqlen` for each incoming
prefill. `merge_batch` (called from `mix_with_running`) then unconditionally
nulls the batch `mamba_track_*` tensors, so the GDN checkpoint write is
skipped. The per-req claim is never rolled back, and
`mamba.prepare_for_caching_req` (the v0.5.20 unified-cache live path) reads
`mamba_last_track_seqlen` to size the mamba donation. Result: a stale ping-pong
slot (never written for this seqlen) is donated into the unified radix tree
under a mismatched key.

The 5-edit fix (A1/A2/B2 on `schedule_batch.py`, C/D on `mamba.py`) is a
**load-bearing unit** — apply and revert them together, never partially:

- **B2** (`mix_with_running`, **head — before `merge_batch`**) clears the stale
  `mamba_last_track_seqlen` and sets the rollback flag, scoped to the **incoming
  prefill reqs only**. At the head of `mix_with_running`, `self.reqs` is only the
  incoming prefills — the running decode reqs are not merged in yet, so their
  (valid) stamps are left untouched.
- **C** (`prepare_for_caching_req`) returns **0** (not `None`) when the claim was
  rolled back, so the caller's `if cl is not None: effective_cache_len =
  min(len(token_ids), cl)` truncates to 0. In `cache_unfinished_req` that hits
  the existing `effective_cache_len <= 0` guard; in `cache_finished_req` the
  empty-key `insert()` is short-circuited by `unified_tree_core.begin_insert`.
  Either way: no insert, no assertion, no stale mamba donation.
- **D** (`commit_insert_component_data`) tolerates `mamba_value is None` instead
  of asserting — a defensive guard, not strictly reached on the return-0 path.
- **A1/A2** plumb + reset the `mamba_mixed_rollback` flag on `Req`.

**Cost (the safe analogue of pristine):** the rolled-back (mixed) prefill's KV
prefix is not cached — a cold re-prefill on the next reuse. Pristine cached a
*stale* mamba slot (corruption); this one simply does not cache the prefix
that would have been corrupted. That is the intended, correct trade-off.
Carrying the mamba track through the merge so mixed prefills still checkpoint
is upstream **sgl#39526** — a separate follow-up, out of scope here.

## Files patched

| File | Patch |
|------|-------|
| `python/sglang/srt/managers/schedule_batch.py` | #39342 (edits A1, A2, B2) |
| `python/sglang/srt/mem_cache/unified_cache/components/mamba.py` | #39342 (edits C, D) |

Both are in the v0.5.20 unified-cache path. The patch is **content-anchored**
(byte-unique anchors), **version-gated** (warns on version drift, does not
hard-fail), **idempotent** (re-run is a no-op), and **two-pass all-or-nothing**
across both files: it validates every edit in *both* files read-only first, and
only if the whole load-bearing unit is pristine-or-patched does it write
anything. A version-drifted image therefore cannot leave `schedule_batch.py`
patched while `mamba.py` is not.

## Reversibility

```
python3 apply_39342.py --revert   # undo #39342 (only what was actually applied)
```

## F2 fix note — the v1 tail rollback over-reached (corrected 2026-09-16)

v1 placed the `mix_with_running` rollback loop at the **tail** (after
`merge_batch`), where `self.reqs` = incoming prefills **plus** running decode
reqs. A running decode req carries a valid `mamba_last_track_seqlen` (stamped
in its own earlier prefill, cleared only at finish), so the loop cleared that
stamp and set `mamba_mixed_rollback = True` on **every** running decode req in a
mixed batch. Edit C then made `prepare_for_caching_req` return 0 for them, so at
finish their KV prefix was **not cached** — a cold re-prefill on every reuse.
Under sustained high concurrency with mixed chunking (the deployed scenario) that
is a pervasive TTFT / cache-hit-rate regression, invisible to an
answer-correctness or crash-only test.

**Fix:** relocate the loop to the **head** of `mix_with_running`, before
`merge_batch`, where `self.reqs` is only the incoming prefills. The decode reqs'
stamps stay intact. This script carries a transient migration edit that reverts
the v1 tail rollback before adding the head one, so an existing v1 tree lands
cleanly on v2.

## Provenance

- Authored by @A1RM4X, validated 2026-09-15 on a 4× RTX 3090 (GA102, TP4,
  PCIe) rig running Qwen3.8-27B + DFlash2 + HiCache L2 + mixed-chunk. The v0.5.20
  re-cut is a file-rename-only adaptation (mamba_component.py → mamba.py)
  verified round-tripping clean in-image at v0.5.20.

## Compose mount

Mounted at `/etc/club3090/hicache-mamba` (read-only). The compose entrypoint
runs `bash /etc/club3090/hicache-mamba/install.sh` before launching the server,
and **refuses to boot** if the patch cannot be applied or verified (fail-closed).
The compose also requires `--enable-hierarchical-cache` + `--enable-mixed-chunk`
+ `--radix-eviction-policy slru` + `--mamba-max-states-per-path 1` for the #39342
patch to be meaningful (those flags are the trigger condition).
