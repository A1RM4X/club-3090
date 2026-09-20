#!/usr/bin/env bash
# Apply the SGLang #39342 mixed-chunk mamba radix-cache patch to the v0.5.20
# engine source. Single load-bearing patch, applied FAIL-CLOSED:
#
#   apply_39342.py — sgl-project/sglang#39342 (v0.5.20 re-cut)
#     Mixed-chunk corrupts the mamba radix cache: prepare_for_extend stamps
#     mamba_last_track_seqlen; merge_batch nulls the batch track tensor;
#     prepare_for_caching_req reads the stale stamp and donates a stale mamba
#     slot into the unified radix tree under a mismatched key.
#     FIX: 5-edit load-bearing unit (schedule_batch.py A1/A2/B2 + mamba.py
#     C/D). Edit C returns 0 (not None — see the apply script's WHY block) so
#     the caller's early-return guard fires before the insert walk, skipping
#     the stale donation and keeping KV/mamba depths consistent. Edit B2 is
#     scoped to the HEAD of mix_with_running (incoming prefills only) so the
#     v1 tail placement's decode-req over-reach (the F2 cache-hit regression)
#     cannot recur; apply is two-pass all-or-nothing across both files.
#     STATUS: VALIDATED (90/90 clean under sustained mixed-chunk load; the
#     exact trigger that crashed the unpatched engine is neutralized).
#     => If it cannot apply/verify (anchor drift / half-patched), REFUSE BOOT.
#
# v0.5.20 note: the unified-cache module was renamed mamba_component.py ->
# mamba.py; this cut targets the new path. The v0.5.19 cut (mamba_component.py)
# does NOT apply to this image.
#
# Run from the compose command BEFORE `sglang serve`. Idempotent: a restart
# re-runs it and detects the already-patched state instead of failing.
set -uo pipefail

PATCH_DIR="${PATCH_DIR:-/etc/club3090/hicache-mamba}"
APPLY="$PATCH_DIR/apply_39342.py"

if [ "${1:-}" = "--verify" ]; then
  [ -f "$APPLY" ] || { echo "[hicache-mamba] $APPLY not present" >&2; exit 1; }
  if python3 "$APPLY" --check; then
    echo "[hicache-mamba] #39342 verified (all edits applied)."
  else
    echo "[hicache-mamba] #39342 --check FAILED" >&2; exit 1
  fi
  exit 0
fi

[ -f "$APPLY" ] || { echo "[hicache-mamba] ERROR: $APPLY not mounted — refusing to boot unpatched." >&2; exit 1; }

echo "[hicache-mamba] applying sgl#39342 (v0.5.20) ..."
if ! python3 "$APPLY"; then
  echo "[hicache-mamba] ERROR: apply FAILED (anchor drift?) — refusing to boot unpatched." >&2
  exit 1
fi
if ! python3 "$APPLY" --check; then
  echo "[hicache-mamba] ERROR: post-apply check FAILED — refusing to boot." >&2
  exit 1
fi
echo "[hicache-mamba] #39342 applied + verified (all edits present)."
