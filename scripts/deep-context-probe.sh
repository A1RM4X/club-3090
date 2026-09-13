#!/usr/bin/env bash
# deep-context-probe.sh — measure the regime users actually run in (club-3090#1259).
#
# THE GAP THIS FILLS
# ------------------
# Every instrument we ship stops at ~35K accumulated context:
#   • bench-agentic.sh  defaults TURNS=12, caps at 15; its header says the ramp
#                       exists "so the ~35K degrade zone is observed"
#   • bench.sh          single-turn
#   • soak-test.sh      measures GENERATED tokens, not accumulated prompt
#   • concurrency-probe short prompts by design
#   • compaction-probe  a compaction EVENT at depth, not the monotonic ramp to it
#
# @sudeposutemizligi reports ~10 tok/s at ~140K in daily use (#1232, #1052) on a
# config whose agentic numbers look healthy. Both can be true: retracting the
# #1096 "decode cliff" established there is no cliff INSIDE the measured range —
# it established nothing about 140K, because nothing we run reaches 140K.
#
# MODES
#   depth   (default) grow ONE conversation monotonically to TARGET_CTX and
#           report the per-turn curve. This is the #1259 measurement.
#   breadth open SESSIONS distinct conversations until their combined tokens
#           exceed the KV pool, forcing genuine KV eviction, then re-query the
#           earliest. Tests the stranding shape @jb-seo describes in disc #1178:
#           "kv leaf evicted -> that session's mamba checkpoint goes useless ->
#           remaining kv goes useless too".
#
# READING THE OUTPUT
#   depth   cached should track prompt_tok minus the newest turn. TTFT should
#           grow SUB-linearly with context. A knee — TTFT jumping while cached
#           stalls — is prefix reuse failing, which is the thing worth catching.
#           decode_tps is per turn so a decode-side collapse is separable from a
#           prefill-side one; it counts usage.completion_tokens, not chunks (a
#           DFlash chunk carries up to 8 tokens). It is a 48-token counting
#           reply — drafter-friendly, so under spec-dec it is a BEST-CASE
#           acceptance rate: compare it across turns, never against bench.sh.
#           kv_res / mamba_res are the
#           RESIDENT (radix-cached, reusable) share of each pool from /metrics —
#           SGLang only; vLLM publishes no mamba pool, and the columns say n/a.
#   breadth cached=0 + cold TTFT  -> clean eviction (healthy)
#           cached>0 + cold TTFT  -> STRANDED if >=50% was reported cached
#                                    (kv kept but unusable), else PARTIAL eviction
#           cached>0 + fast TTFT  -> healthy reuse
#           Only meaningful when the planned tokens EXCEED the pool — the probe
#           reads the pool from /metrics (or KV_POOL) and warns when they don't.
#
#   cached = n/a is NOT 0. Whether reuse is reportable depends on a SERVER flag,
#   identically for streaming and non-streaming (measured 2026-09-13):
#     SGLang  --enable-cache-report        (no shipped sglang compose passes it)
#     vLLM    --enable-prompt-tokens-details
#   Without it the probe falls back to the engine's prefix-hit counter on
#   /metrics (SGLang --enable-metrics; vLLM default), and a self-test right
#   after calibration — the same prompt twice — prints which source is live.
#   A reuse that happened but is not reported shows as n/a plus the flag to set.
#
# ⚠️ Run against a FRESH boot for depth. A cache already churning from other
# traffic makes degradation indistinguishable from eviction noise. Note a
# dashboard polling /health makes SGLang run a 1-token generate every second.
#
# Usage:
#   URL=http://localhost:8143 MODEL=qwen3.8-27b bash scripts/deep-context-probe.sh
#   MODE=breadth SESSIONS=12 SESSION_CTX=28000 KV_POOL=279595 bash scripts/deep-context-probe.sh
#
# Env:
#   URL           endpoint base (default: registry default port for qwen3.6-27b)
#   MODEL         served model name (required — no default; a wrong one 404s)
#   MODE          depth | breadth            (default: depth)
#   TARGET_CTX    depth: accumulated prompt tokens to reach (default: 140000)
#   TURN_TOKENS   depth: approx tokens appended per turn   (default: 4000)
#   SESSIONS      breadth: how many distinct conversations (default: 12)
#   SESSION_CTX   breadth: tokens per conversation         (default: 28000)
#   KV_POOL       breadth: engine KV pool, for the over-capacity report
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

URL="${URL:-http://localhost:8020}"
MODEL="${MODEL:-}"
MODE="${MODE:-depth}"
TARGET_CTX="${TARGET_CTX:-140000}"
TURN_TOKENS="${TURN_TOKENS:-4000}"
SESSIONS="${SESSIONS:-12}"
SESSION_CTX="${SESSION_CTX:-28000}"
KV_POOL="${KV_POOL:-0}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required (the served name). A wrong one returns HTTP 404 and reads as a dead server." >&2
  exit 2
fi
case "$MODE" in depth|breadth) ;; *) echo "MODE must be depth|breadth (got '$MODE')" >&2; exit 2 ;; esac
curl -sf -m 10 "${URL}/v1/models" >/dev/null 2>&1 || {
  echo "no server at ${URL} — check the port. ⚠️ 12 composes ignore ESTATE_PORT (#1293), so the" >&2
  echo "container may be bound to its default rather than the port you asked for; run 'docker port <c>'." >&2
  exit 1; }

exec python3 "${ROOT_DIR}/scripts/lib/deep_context_probe.py" \
  "$URL" "$MODEL" "$MODE" "$TARGET_CTX" "$TURN_TOKENS" "$SESSIONS" "$SESSION_CTX" "$KV_POOL"
