#!/usr/bin/env bash
#
# Guard for scripts/deep-context-probe.sh (club-3090#1259).
#
# The probe exists because every other instrument stops at ~35K accumulated
# context. Its value is entirely in being RUNNABLE when someone needs it at 140K,
# so this checks the things that would make it fail at that moment: a missing
# MODEL that 404s and reads as a dead server, a bad MODE, a dead endpoint, and
# the output contracts a reader depends on.
#
# The contracts are checked against a FAKE ENGINE (stdlib http.server, below)
# that reproduces the shapes measured on SGLang v0.5.19 / read from vLLM v0.29.0
# source on 2026-09-13 — in particular the false-clean this probe was caught
# producing: `prompt_tokens_details` absent (a server flag is off) being read as
# cached=0 while the engine was reusing 4,160 tokens. No GPU server required.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${ROOT}/scripts/deep-context-probe.sh"
P="${ROOT}/scripts/lib/deep_context_probe.py"
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

[[ -x "$S" ]] || bad "wrapper executable" "executable scripts/deep-context-probe.sh" "missing or not +x"
[[ -f "$P" ]] || bad "implementation present" "scripts/lib/deep_context_probe.py" "missing"
bash -n "$S" 2>/dev/null || bad "wrapper syntax" "clean bash -n" "syntax error"
python3 -c "import ast,sys; ast.parse(open('$P').read())" 2>/dev/null || bad "probe syntax" "parseable python" "syntax error"

# --- refusal: MODEL is required. A wrong/absent model 404s, which reads as a
# dead server — the exact confusion this must not create.
out="$(URL=http://127.0.0.1:9 MODEL= bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 2 ]] || bad "missing MODEL exits 2" "2" "$rc"
command grep -qi '404' <<<"$out" || bad "missing-MODEL message explains the 404 risk" "a mention of 404" "absent"
ok "refuses without MODEL, and says why it matters"

# --- refusal: bad MODE
out="$(URL=http://127.0.0.1:9 MODEL=x MODE=sideways bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 2 ]] || bad "bad MODE exits 2" "2" "$rc"
ok "refuses an unknown MODE"

# --- refusal: dead endpoint, and it must point at the ESTATE_PORT trap (#1293)
out="$(URL=http://127.0.0.1:9 MODEL=x bash "$S" 2>&1)"; rc=$?
[[ $rc -eq 1 ]] || bad "dead endpoint exits 1" "1" "$rc"
command grep -qi 'docker port' <<<"$out" || bad "dead-endpoint hint names the port trap" "a 'docker port' hint" "absent"
ok "refuses a dead endpoint and names the #1293 port trap"

# --- static contract: an unmeasurable decode window must not render as 0.0
command grep -q "'n/a'" "$P" || bad "unmeasurable decode renders n/a" "an 'n/a' branch (#1267)" "absent"
command grep -q 'def ok(win, tok)' "$P" || bad "decode window/token floor present" "a window+token guard" "absent"
ok "unmeasurable decode reports n/a, not a 0.0 that reads as silent-empty (#1267)"

# --- static contract: TTFT must be taken on the first chunk carrying choices,
# not the first chunk with CONTENT — otherwise a reasoning model's reasoning
# phase is charged to prefill (the #1096 retraction).
command grep -q 'if ttft is None: ttft = time.time() - t0' "$P" \
  || bad "TTFT anchored on first choices chunk" "ttft set when ch is truthy" "not found"
ok "TTFT anchored on the first chunk carrying choices (#1096 trap avoided)"

# ===========================================================================
# Fake engine. REPORT mode selects how prefix reuse is (not) exposed:
#   usage    prompt_tokens_details = {"cached_tokens": N} only when N > 0 (SGLang
#            --enable-cache-report shape; vLLM --enable-prompt-tokens-details is
#            the same but sends 0 explicitly)
#   metrics  field absent; /metrics publishes the SGLang prefix-hit counter
#   none     field absent; /metrics is 404 (every shipped sglang compose today)
# Every content chunk carries cumulative usage.completion_tokens (continuous
# usage stats) and one chunk carries MANY tokens (1 -> 9 -> 16, as measured under
# DFlash), so a chunk-counting decode rate reads ~5x low.
# ===========================================================================
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/fake_engine.py" <<'PYEOF'
import json, sys, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, REPORT = int(sys.argv[1]), sys.argv[2]
# Two REPORT values change the STREAM SHAPE rather than the cached-token source;
# both reproduce a decode n/a seen on a real 141K run. They report cached tokens
# exactly like "usage" so only the decode path is under test.
SHAPE = REPORT if REPORT in ("usage_once", "one_token") else "normal"
if SHAPE != "normal": REPORT = "usage"
SEEN = []; COUNTER = {"cache": 0}; LOCK = threading.Lock(); KV_TOTAL = 10000
def toks(s): return len(s) // 4 + 10
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype):
        b = body.encode(); self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, json.dumps({"data": [{"id": "fake"}]}), "application/json")
        if self.path == "/metrics" and REPORT == "metrics":
            with LOCK:
                res = min(KV_TOTAL, sum(toks(p) for p in SEEN)); n = min(33, len(SEEN))
                body = (f'sglang:realtime_tokens_total{{mode="prefill_cache",}} {COUNTER["cache"]}\n'
                        f'sglang:realtime_tokens_total{{mode="prefill_compute",}} 0\n'
                        f'sglang:realtime_tokens_created{{mode="prefill_cache",}} 1.7e9\n'
                        f'sglang:kv_used_tokens{{tp_rank="0"}} 0\nsglang:kv_evictable_tokens{{tp_rank="0"}} {res}\n'
                        f'sglang:kv_available_tokens{{tp_rank="0"}} {KV_TOTAL - res}\n'
                        f'sglang:mamba_used_tokens{{tp_rank="0"}} 0\nsglang:mamba_evictable_tokens{{tp_rank="0"}} {n}\n'
                        f'sglang:mamba_available_tokens{{tp_rank="0"}} {33 - n}\n')
            return self._send(200, body, "text/plain")
        self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
        prompt = "".join(m.get("content") or "" for m in req["messages"]); ptok = toks(prompt)
        with LOCK:
            cached = min(ptok, max([toks(p) for p in SEEN if prompt.startswith(p)] or [0]))
            COUNTER["cache"] += cached
            if prompt not in SEEN: SEEN.append(prompt)
        time.sleep(min(1.0, (ptok - cached) * 0.0001))          # emulated prefill
        maxtok = int(req.get("max_tokens", 16))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def chunk(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
        chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], "usage": None})
        steps = [(1, "OK")] if maxtok <= 8 else [(1, "one"), (9, ", two, three, four, five"), (16, ", six, seven, eight,")]
        if SHAPE == "one_token" and maxtok > 8: steps = [(1, "one")]   # model stops early: no window at all
        for cum, piece in steps:
            chunk({"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                   # usage_once: content streams but per-chunk usage never arrives,
                   # so there is no usage DELTA to difference (the turn-17 shape).
                   "usage": (None if SHAPE == "usage_once" else
                             {"prompt_tokens": ptok, "completion_tokens": cum, "total_tokens": ptok + cum})})
            time.sleep(0.15)
        ctok = steps[-1][0]
        chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": None})
        chunk({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
               "prompt_tokens_details": ({"cached_tokens": cached} if (REPORT == "usage" and cached > 0) else None)}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
PYEOF

# run_fake REPORT MODE [env...] -> output in $out. Kills the server by PID (never pkill -f).
run_fake() {
  local report="$1" mode="$2"; shift 2
  local port; port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
  python3 "$TMP/fake_engine.py" "$port" "$report" >"$TMP/fake.$report.$mode.log" 2>&1 &
  local pid=$!
  local i; for i in $(seq 1 50); do curl -sf -m 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && break; sleep 0.1; done
  out="$(env URL="http://127.0.0.1:$port" MODEL=fake MODE="$mode" "$@" bash "$S" 2>&1)"; rc=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
}
# 3rd column of the depth/breadth table rows (turn/session lines start with a number)
col_cached() { awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9,]+$/ {print $3}' <<<"$1"; }
col_decode() { awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9,]+$/ {print $5}' <<<"$1"; }

# --- contract 3 (the false-clean): field ABSENT and no counter -> cached is n/a,
# NEVER 0, and the output names the server flags that would light it up.
run_fake none depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(none) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
vals="$(col_cached "$out")"
[[ -n "$vals" ]] || bad "depth table rendered (none)" "turn rows" "none: $out"
if command grep -qvx 'n/a' <<<"$vals"; then bad "absent field renders n/a" "every cached cell n/a" "$(tr '\n' ' ' <<<"$vals")"; fi
command grep -q 'reuse HAPPENED' <<<"$out" || bad "self-test detects unreported reuse" "'reuse HAPPENED ... NOT REPORT'" "absent: $out"
command grep -q -- '--enable-cache-report' <<<"$out" || bad "names the SGLang flag" "--enable-cache-report" "absent"
command grep -q -- '--enable-prompt-tokens-details' <<<"$out" || bad "names the vLLM flag" "--enable-prompt-tokens-details" "absent"
ok "absent prompt_tokens_details renders cached=n/a (never 0) and names both engine flags"

# --- contract 4: field present -> numeric, and it tracks the growing prefix
run_fake usage depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(usage) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'cached column source: usage' <<<"$out" || bad "self-test picks the usage field" "source: usage" "absent: $out"
last="$(col_cached "$out" | tail -1 | tr -d ,)"
[[ "$last" =~ ^[0-9]+$ && "$last" -gt 0 ]] || bad "cached is numeric and >0 once the prefix repeats" ">0" "'$last'"
first="$(col_cached "$out" | head -1)"
[[ "$first" == "0" ]] || bad "first turn (nothing cached, field omitted) reads 0 once the field is proven live" "0" "'$first'"
ok "present prompt_tokens_details renders numbers; omitted-when-zero reads 0 only after the self-test proved the field live"

# --- contract 5: field absent but the engine's prefix-hit counter exists -> the
# counter delta is used and the column is labelled with that source
run_fake metrics depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(metrics) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'cached column source: metrics' <<<"$out" || bad "self-test falls back to the /metrics counter" "source: metrics" "absent: $out"
last="$(col_cached "$out" | tail -1 | tr -d ,)"
[[ "$last" =~ ^[0-9]+$ && "$last" -gt 0 ]] || bad "counter-delta cached is numeric and >0" ">0" "'$last'"
command grep -qE '^ +[0-9]+ +[0-9,]+ +[0-9,n/a]+ +[0-9.]+ +[~0-9.n/a]+ +[0-9]+% +[0-9]+/33' <<<"$out" \
  || bad "pool residency columns come from /metrics (kv %, mamba slots/total)" "'NN% n/33' columns" "absent: $out"
ok "absent field + /metrics counter -> counter delta, labelled 'metrics'; pool residency from /metrics"

# --- contract 6: decode_tps counts TOKENS (usage.completion_tokens), not chunks.
# The fake streams 16 tokens in 3 chunks ~0.15s apart: tokens/window ~ 50,
# chunks/window ~ 7. Anything under 20 means chunks were counted.
d="$(col_decode "$out" | head -1)"
python3 -c "import sys; v=float(sys.argv[1].lstrip('~')); sys.exit(0 if v > 20 else 1)" "$d" 2>/dev/null \
  || bad "decode_tps derives from completion_tokens" ">20 tok/s for 16 tokens in 3 chunks" "'$d'"
[[ "$d" != ~* ]] || bad "per-chunk usage honoured (no '~' approx marker)" "exact rate" "'$d'"
ok "decode_tps counts tokens from continuous usage, not chunks (spec-dec safe)"

# --- contract 6b (the gap the 141K run hit): an engine that streams content but
# sends usage ONCE has no usage DELTA to difference. The old code stopped there
# and printed a bare n/a for every turn past 104K — the one number the report was
# commissioned to produce. It must now fall through to the content-chunk window,
# still count TOKENS from usage, mark the row '~', and name the basis it used.
run_fake usage_once depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(usage_once) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
d="$(col_decode "$out" | head -1)"
[[ "$d" != "n/a" ]] || bad "usage-once stream still yields a decode rate" "a number" "n/a: $out"
[[ "$d" == ~* ]] || bad "a fallback basis is marked approximate" "'~NN.N'" "'$d'"
python3 -c "import sys; v=float(sys.argv[1].lstrip('~')); sys.exit(0 if v > 20 else 1)" "$d" 2>/dev/null \
  || bad "fallback still counts TOKENS not chunks" ">20 tok/s for 16 tokens in 3 chunks" "'$d'"
command grep -q 'decode basis: usage-window' <<<"$out" || bad "the fallback basis is named on the row" "'decode basis: usage-window'" "absent: $out"
ok "usage sent ONCE still measures decode (content-chunk window), marked '~' and basis named"

# --- contract 6c: when decode is genuinely unmeasurable (one token, no window),
# n/a is correct — but it must carry the stream shape that caused it, so a reader
# can tell "the model stopped early" from "the probe went blind". An n/a with no
# reason is the #1267 ambiguity one level up.
run_fake one_token depth TARGET_CTX=2500 TURN_TOKENS=800
[[ $rc -eq 0 ]] || bad "depth run against fake(one_token) exits 0" "0" "$rc: $(tail -3 <<<"$out")"
d="$(col_decode "$out" | head -1)"
[[ "$d" == "n/a" ]] || bad "a single-token reply is not timeable" "n/a" "'$d'"
command grep -q 'decode n/a: 1 completion tok' <<<"$out" || bad "n/a names the token count" "'decode n/a: 1 completion tok'" "absent: $out"
command grep -q 'finish_reason=stop' <<<"$out" || bad "n/a names the finish_reason" "finish_reason=stop" "absent: $out"
command grep -q 'short reply: 1/48 tok' <<<"$out" || bad "a reply under the cap is flagged" "'short reply: 1/48 tok'" "absent: $out"
ok "unmeasurable decode reports n/a WITH the stream shape, and flags the short reply"

# --- contract 7: breadth without eviction pressure says so, and re-queries verdict
run_fake usage breadth SESSIONS=2 SESSION_CTX=600 KV_POOL=100000
[[ $rc -eq 0 ]] || bad "breadth run exits 0" "0" "$rc: $(tail -3 <<<"$out")"
command grep -q 'do NOT exceed the pool' <<<"$out" || bad "breadth warns when planned tokens fit the pool" "a no-pressure warning" "absent: $out"
command grep -qc 'HEALTHY reuse' <<<"$out" || bad "breadth re-query verdicts rendered" "HEALTHY reuse rows" "absent: $out"
ok "breadth warns when there is no eviction pressure and classifies re-queries"

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-deep-context-probe" >&2; exit 1; fi
echo "PASS: test-deep-context-probe (club-3090#1259)"
