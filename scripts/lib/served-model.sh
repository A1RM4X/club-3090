#!/usr/bin/env bash
# served-model.sh — the model id an OpenAI-compatible endpoint is serving.
#
# Source it; it defines one function and runs nothing at source time.
#
# Why this exists (#1360): TabbyAPI (exllamav3) answers /v1/models with EVERY
# folder in its model directory, so `data[0].id` is whichever folder the
# filesystem lists first — a report got `modules`, and every script that took
# the first entry then labelled or targeted the wrong model. TabbyAPI's
# /v1/model (singular) returns the model actually LOADED. vLLM, llama.cpp and
# SGLang 404 on /v1/model, so for them this is exactly the old first-entry read.

export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779: python3 below must not decode with the locale codec

# club_served_model_id URL — print the served model id for URL (a base URL such
# as http://localhost:8020, with or without a trailing /v1/models), or nothing.
# Per-request timeout: CLUB_MODEL_ID_TIMEOUT_S (default 5).
club_served_model_id() {
  local base="${1%/}" t="${CLUB_MODEL_ID_TIMEOUT_S:-5}" id=""
  base="${base%/v1/models}"
  base="${base%/v1}"
  [[ -z "$base" ]] && return 0
  command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1 || return 0

  # A genuine model card only: some servers answer unknown paths with a web UI
  # or an unrelated JSON object, and neither may be read as a model id.
  id="$(curl -sf -m "$t" "${base}/v1/model" 2>/dev/null | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    ok = isinstance(d, dict) and d.get("object") == "model" and isinstance(d.get("id"), str)
    print(d["id"] if ok else "")
except Exception:
    print("")' 2>/dev/null || true)"
  if [[ -z "$id" ]]; then
    id="$(curl -sf -m "$t" "${base}/v1/models" 2>/dev/null | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin).get("data", [])
    print(d[0]["id"] if d else "")
except Exception:
    print("")' 2>/dev/null || true)"
  fi
  printf '%s' "$id"
}
