#!/usr/bin/env bash
# Test for scripts/lib/engine-restart-guard.sh.
#
# The bug it locks down (club-3090 discussion #1076, reported by foureight84):
# a fatal EngineCore error killed the engine mid-run, Docker's restart policy
# brought the container back, and the harness retried the in-flight scenarios
# against a dead/booting engine. The run COMPLETED AND REPORTED A PLAUSIBLE
# SCORE — three --full runs were published at 129 / 128 / 127 before anyone
# noticed. RestartCount is the only signal that separates "the model got these
# wrong" from "the engine was not running".
#
# Validates:
#   1.  POSITIVE CONTROL: RestartCount rising ⇒ rc=1 and a loud message. A gate
#       that cannot detect the bug it was written for is theatre — this case is
#       the reason the file exists, so it is asserted first.
#   2.  NEGATIVE CONTROL: RestartCount flat ⇒ rc=0, silent. A guard that fires on
#       healthy runs would be turned off within a week.
#   3.  CONTAINER=none (bare-metal endpoint) ⇒ rc=2 AND says so. Unavailable must
#       NOT read as clean, or the guard becomes the false-clean it prevents.
#   4.  docker absent ⇒ rc=2 and says so.
#   5.  RestartCount unreadable (container removed mid-run) ⇒ rc=2, NOT rc=0.
#   6.  CLUB3090_ALLOW_ENGINE_RESTART=1 ⇒ rc=0 despite a rise (documented escape).
#   7.  A multi-restart rise reports the correct delta.
#
# Harness: mock `docker` on PATH, env-driven RestartCount. Fully offline.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n     %s\n' "$1" "${2:-}"; }

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cat > "${tmp_dir}/docker" <<'EOF'
#!/usr/bin/env bash
[[ "${1:-}" == "inspect" ]] || exit 0
# ⚠️ `${VAR-0}`, NOT `${VAR:-0}`: the colon form maps an EMPTY value to 0, which
# would make the "unreadable count" case silently test a readable 0 instead.
printf '%s\n' "${DOCKER_MOCK_RESTARTS-0}"
EOF
chmod +x "${tmp_dir}/docker"
PATH_WITH_DOCKER="${tmp_dir}:${PATH}"

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/lib/engine-restart-guard.sh"

# ⚠️ ERRFILE is set HERE, in the parent. Assigning it inside run_check would be
# lost: run_check is called in a command substitution, i.e. a subshell.
ERRFILE="${tmp_dir}/err.out"
run_check() { # run_check <before> <container> ; echoes rc, stderr -> $ERRFILE
  local before="$1" container="$2" rc=0
  : > "$ERRFILE"
  restart_guard_check "$before" "$container" "run" 2>"$ERRFILE" || rc=$?
  echo "$rc"
}

# --- 1. POSITIVE CONTROL: a rise must be caught -------------------------------
export DOCKER_MOCK_RESTARTS=4 CLUB3090_ALLOW_ENGINE_RESTART=0
PATH="$PATH_WITH_DOCKER"
rc="$(run_check 1 some-engine)"
if [[ "$rc" == "1" ]]; then ok "positive control: RestartCount 1->4 detected (rc=1)"
else bad "positive control: RestartCount 1->4 detected" "got rc=$rc, expected 1 — THE GUARD CANNOT SEE THE BUG"; fi
if command grep -q "ENGINE RESTARTED DURING THIS RUN" "$ERRFILE"; then ok "positive control: loud message emitted"
else bad "positive control: loud message" "stderr: $(head -2 "$ERRFILE")"; fi
if command grep -q "1 → 4" "$ERRFILE"; then ok "positive control: reports the delta"
else bad "positive control: reports the delta" "stderr: $(head -3 "$ERRFILE")"; fi

# --- 2. NEGATIVE CONTROL: flat count must NOT fire ----------------------------
export DOCKER_MOCK_RESTARTS=2
rc="$(run_check 2 some-engine)"
if [[ "$rc" == "0" ]]; then ok "negative control: flat RestartCount is clean (rc=0)"
else bad "negative control: flat RestartCount" "got rc=$rc, expected 0 — would fire on healthy runs"; fi
if [[ ! -s "$ERRFILE" ]]; then ok "negative control: silent on a clean run"
else bad "negative control: silent" "unexpected stderr: $(head -2 "$ERRFILE")"; fi

# --- 3. CONTAINER=none must be UNAVAILABLE, not clean -------------------------
rc="$(run_check 0 none)"
if [[ "$rc" == "2" ]]; then ok "CONTAINER=none ⇒ rc=2 (unavailable, not clean)"
else bad "CONTAINER=none ⇒ rc=2" "got rc=$rc — a skipped check must not look like a pass"; fi
if command grep -q "not checked" "$ERRFILE"; then ok "CONTAINER=none says it did not check"
else bad "CONTAINER=none says so" "stderr: $(head -2 "$ERRFILE")"; fi

# --- 4. docker absent ⇒ unavailable -------------------------------------------
rc="$(PATH=/nonexistent-bin run_check 0 some-engine)"
if [[ "$rc" == "2" ]]; then ok "docker absent ⇒ rc=2"
else bad "docker absent ⇒ rc=2" "got rc=$rc"; fi

# --- 5. unreadable count ⇒ unavailable, NOT clean -----------------------------
export DOCKER_MOCK_RESTARTS=""
rc="$(run_check 0 some-engine)"
if [[ "$rc" == "2" ]]; then ok "unreadable RestartCount ⇒ rc=2 (not a silent pass)"
else bad "unreadable RestartCount ⇒ rc=2" "got rc=$rc — container removed mid-run would read as clean"; fi

# --- 6. documented escape hatch -----------------------------------------------
export DOCKER_MOCK_RESTARTS=9 CLUB3090_ALLOW_ENGINE_RESTART=1
rc="$(run_check 0 some-engine)"
if [[ "$rc" == "0" ]]; then ok "CLUB3090_ALLOW_ENGINE_RESTART=1 overrides (rc=0)"
else bad "override" "got rc=$rc, expected 0"; fi
unset CLUB3090_ALLOW_ENGINE_RESTART

# --- 7. snapshot returns empty when uncheckable -------------------------------
out="$(CONTAINER=none restart_guard_snapshot)"
if [[ -z "$out" ]]; then ok "snapshot: empty for CONTAINER=none"
else bad "snapshot: empty for CONTAINER=none" "got '$out'"; fi

printf '\n  PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
[[ "$FAIL" -eq 0 ]]
