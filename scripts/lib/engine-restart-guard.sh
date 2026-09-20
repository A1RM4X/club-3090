#!/usr/bin/env bash
# Engine-restart guard — catches an engine that died and was restarted MID-RUN.
#
# WHY THIS EXISTS (club-3090 discussion #1076, reported by foureight84):
# a fatal EngineCore error kills the serving process, Docker's restart policy
# brings the container back, and the harness retries the in-flight scenarios
# against the rebooting engine. The run then COMPLETES AND REPORTS A PLAUSIBLE
# SCORE. Three --full runs were published at 129 / 128 / 127 before anyone
# noticed the engine had died during all three.
#
# `RestartCount` is the only honest signal: it is monotonic, it survives the
# restart, and nothing else in the output distinguishes "the model got these
# wrong" from "the engine was dead when these ran".
#
# ⚠️⚠️ UNAVAILABLE IS NOT CLEAN. If the probe cannot run — CONTAINER=none for a
# bare-metal endpoint, no docker binary, an inspect that fails — this must SAY
# SO rather than return success. Otherwise a silent skip and a genuine pass
# produce identical output, and the guard becomes the very false-clean it was
# written to prevent.
#
# Exit codes from restart_guard_check:
#   0  clean       — count unchanged
#   1  RESTARTED   — engine died mid-run; every score after it is suspect
#   2  unavailable — could not measure; caller must surface this, not ignore it

# Echo the current RestartCount, or empty when it cannot be read.
restart_guard_snapshot() {
  local container="${1:-${CONTAINER:-}}"
  [[ -n "$container" && "$container" != "none" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0
  docker inspect "$container" --format '{{.RestartCount}}' 2>/dev/null || true
}

# restart_guard_check <before> [container] [label]
restart_guard_check() {
  local before="${1:-}" container="${2:-${CONTAINER:-}}" label="${3:-run}"
  local after

  if [[ -z "$container" || "$container" == "none" ]]; then
    echo "[restart-guard] not checked: no container (CONTAINER='${container:-unset}') — a mid-${label} engine restart would be INVISIBLE" >&2
    return 2
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "[restart-guard] not checked: docker not on PATH — a mid-${label} engine restart would be INVISIBLE" >&2
    return 2
  fi
  after="$(docker inspect "$container" --format '{{.RestartCount}}' 2>/dev/null || true)"

  # Both readings must be plain integers. A container removed mid-run reads
  # empty, which is a measurement failure, NOT a clean result.
  if [[ ! "$before" =~ ^[0-9]+$ || ! "$after" =~ ^[0-9]+$ ]]; then
    echo "[restart-guard] not checked: RestartCount unreadable for '${container}' (before='${before}' after='${after}') — a mid-${label} engine restart would be INVISIBLE" >&2
    return 2
  fi

  if (( after > before )); then
    echo "" >&2
    echo "[restart-guard] ⛔ ENGINE RESTARTED DURING THIS ${label^^} — RestartCount ${before} → ${after} ($(( after - before ))x)" >&2
    echo "[restart-guard]    The engine died and Docker restarted it. Scenarios in flight were retried" >&2
    echo "[restart-guard]    against a dead or booting engine, so THIS RESULT IS NOT COMPARABLE and must" >&2
    echo "[restart-guard]    not be published. Check the engine log for the fatal error, then re-run." >&2
    echo "[restart-guard]    Override with CLUB3090_ALLOW_ENGINE_RESTART=1 only if you know why it died." >&2
    echo "" >&2
    [[ "${CLUB3090_ALLOW_ENGINE_RESTART:-0}" == "1" ]] && return 0
    return 1
  fi
  return 0
}
