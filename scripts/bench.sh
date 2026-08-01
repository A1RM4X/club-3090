#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - Runs both the canonical narrative AND code prompts in one invocation.
#     This matches the README's narrative/code TPS pairing.
#   - 3 warmup + N measured runs per prompt (default 5 narrative + 5 code).
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - per-prompt summary: mean / std / CV for both TPS metrics + mean TTFT
#     + prompt-processing throughput (`PP tok/s`)
#   - shows MTP SpecDecoding metrics from docker logs at the end
#
# Why two TPS metrics:
#   - wall_TPS  = "user-perceived speed" (includes prefill cost)
#   - decode_TPS = "model decode rate" (excludes prefill)
#   For long prompts the two can differ a lot. For short prompts they
#   converge. Reporting both keeps comparisons honest across configs.
#
# Why narrative + code:
#   MTP acceptance varies wildly by prompt structure. Code (repetitive,
#   token-predictable) accepts at ~80% per position; prose (semantically
#   rich) at ~50%. Reporting only one half is misleading. README claims
#   like "66 / 85 TPS" pair them; bench should too.
#
# Prereq: stack is running and reports "Application startup complete".
#
# Env vars:
#   URL                Endpoint. Default: http://localhost:8020
#   MODEL              Served model name. Default: auto-detected from
#                      /v1/models, else qwen3.6-27b
#   CONTAINER          Container for log scraping. Default: vllm-qwen36-27b
#   RUNS               Measured runs per prompt. Default: 5
#   WARMUPS            Warm-up runs (shared across both). Default: 3
#   PROMPT_NARR        Override narrative prompt
#   PROMPT_CODE        Override code prompt
#   MAX_TOKENS_NARR    Default: 1000
#   MAX_TOKENS_CODE    Default: 800
#   ONLY               Set to "narr" or "code" to skip the other. Default: both
#   QUIET              Set to 1 to skip per-run lines (just print summary)
#   PP                 Set to 1 to add the long-prompt PP fallback probe.
#                      llama.cpp containers enable this automatically.
#   PP_FALLBACK_TOKENS Approximate filler-token target for PP=1. Default: 10000
#   PP_MAX_TOKENS      Completion cap for the PP fallback request. Default: 16
#   PREFILL_PROBE      Set to 0 to skip the canonical two-depth prefill/TTFT
#                      probe (catalog-baselines 2c). Default: 1. Each request
#                      uses a FRESH salted haystack — composes serve
#                      enable_prefix_caching, so identical prompts re-measure
#                      the CACHE HIT, not prefill. Depths exceeding the served
#                      context are skipped with a note.
#   PREFILL_DEPTHS     CSV of prompt-token depths. Default: 10000,90000 (the
#                      shallow/deep warm anchors; 90K sits inside the DeltaNet
#                      degradation regime and pairs with the verify-stress
#                      ladder's ~94K rung for anchor calibration).
#   PREFILL_RUNS       Measured runs per depth. Default: 3
#   ENABLE_THINKING    Set to 1 to send chat_template_kwargs.enable_thinking=true
#                      in bench requests. Default: 0.
#   FORCE_TOKENS       Force EXACTLY this many output tokens per run (sets
#                      max_tokens + min_tokens + ignore_eos), overriding
#                      MAX_TOKENS_NARR/CODE. Use to bench at a fixed / larger output
#                      size. Essential for DIFFUSION LMs: they self-terminate early
#                      (~1-2K words), so raising the cap alone won't lengthen the
#                      generation — you must force the length to measure sustained
#                      throughput. vLLM-oriented (ignore_eos/min_tokens). 0 = off
#                      (model decides length). Default: 0.
#
# Capture knobs (the always-on capture layer — scripts/lib/capture.sh):
#   SERVER_LOG         Path to a BARE-METAL server log. CONTAINER=none runs used to
#                      report "log scrape unavailable" and every engine-side number
#                      had to be hand-extracted; point this at the file the server
#                      writes and the engine-side TPS, expert-cache and drafter
#                      captures all light up. Default: unset (docker log scrape,
#                      or nothing).
#   SERVER_PID         Pin the serving process for /proc inspection (argv fingerprint,
#                      moe-cache env, RSS, swap). Default: auto (`pgrep -x`).
#                      SERVER_PID=none disables /proc inspection entirely.
#   ENDPOINT           chat (default) drives /v1/chat/completions with the model's
#                      chat template applied — the historical bench behaviour, kept
#                      as the default so numbers stay comparable. ENDPOINT=completion
#                      drives raw /v1/completions with NO template, for base models
#                      and for isolating template overhead.
#   STREAM_CALIB       1 = run a ~3 s STREAM-triad RAM-bandwidth ceiling calibration
#                      on the idle box before the bench, so the derived miss-path
#                      demand can be stated as a fraction of what this host delivers.
#                      Default: 0 (opt-in — it costs 3 s and needs numpy).
#                      NOTE: the CONTENTION probe (co-running STREAM during decode)
#                      is deliberately NOT here — it perturbs the run by construction
#                      and belongs in a separate opt-in arm.
#   CAPTURE            0 = skip the whole capture layer (the pre-capture output
#                      shape, for a harness that parses it). Default: 1.
#   SHORT_EOS_FRAC     A run whose completion is below this fraction of max_tokens
#                      counts as a short-EOS run and is excluded from "n-usable".
#                      Chat-tuned models EOS early and silently degenerate a 5-run
#                      shape to n=1. Default: 0.25.
#
# Usage:
#   bash scripts/bench.sh
#   ONLY=code bash scripts/bench.sh
#   PP=1 bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh
#   FORCE_TOKENS=4000 bash scripts/bench.sh   # fixed 4000-tok output (diffusion / sustained-TPS)
#   SERVER_LOG=/var/log/llama-server.log bash scripts/bench.sh   # bare metal
#   STREAM_CALIB=1 bash scripts/bench.sh      # + RAM-bandwidth ceiling

set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

# Auto-detect running container + port (URL/CONTAINER env vars still win).
# See scripts/preflight.sh::preflight_autodetect_endpoint.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint || true
fi
URL="${URL:-http://localhost:8020}"
# Resolve the served model from /v1/models when MODEL is unset (#372). The qwen
# literal below is only a last resort if detection no-ops (endpoint unreachable).
declare -F preflight_autodetect_model >/dev/null && preflight_autodetect_model
MODEL="${MODEL:-qwen3.6-27b}"
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS_NARR="${MAX_TOKENS_NARR:-1000}"
MAX_TOKENS_CODE="${MAX_TOKENS_CODE:-800}"
PROMPT_NARR="${PROMPT_NARR:-Write a detailed 800-word essay explaining transformer attention.}"
PROMPT_CODE="${PROMPT_CODE:-Write a Python implementation of quicksort with comments explaining each step.}"
ONLY="${ONLY:-both}"
QUIET="${QUIET:-0}"
PP="${PP:-0}"
PP_FALLBACK_TOKENS="${PP_FALLBACK_TOKENS:-10000}"
PP_MAX_TOKENS="${PP_MAX_TOKENS:-16}"
PREFILL_PROBE="${PREFILL_PROBE:-1}"
PREFILL_DEPTHS="${PREFILL_DEPTHS:-10000,90000}"
PREFILL_RUNS="${PREFILL_RUNS:-3}"
# The probe knobs reach the python heredoc via the environment (the argv tuple
# is full); export them here.
export PREFILL_PROBE PREFILL_DEPTHS PREFILL_RUNS
ENABLE_THINKING="${ENABLE_THINKING:-0}"
FORCE_TOKENS="${FORCE_TOKENS:-0}"
# --- capture layer knobs (see the header block) -----------------------------
CAPTURE="${CAPTURE:-1}"
ENDPOINT="${ENDPOINT:-chat}"
STREAM_CALIB="${STREAM_CALIB:-0}"
SHORT_EOS_FRAC="${SHORT_EOS_FRAC:-0.25}"
case "$ENDPOINT" in
  chat|completion) : ;;
  *) echo "ERROR: ENDPOINT must be 'chat' or 'completion' (got '$ENDPOINT')" >&2; exit 1 ;;
esac

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

ENGINE_KIND="${ENGINE_KIND:-unknown}"
if [[ "$ENGINE_KIND" == "unknown" && "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  container_image="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER}" 2>/dev/null || true)"
  container_name="$(docker inspect --format '{{.Name}}' "${CONTAINER}" 2>/dev/null || true)"
  if [[ "${container_image} ${container_name}" == *"llama.cpp"* || "${container_image} ${container_name}" == *"llama-cpp"* ]]; then
    ENGINE_KIND="llamacpp"
  elif [[ "${container_image} ${container_name}" == *"vllm"* ]]; then
    ENGINE_KIND="vllm"
  fi
fi

PP_MODE="log"
if [[ "$PP" == "1" || "$ENGINE_KIND" == "llamacpp" ]]; then
  PP_MODE="fallback"
fi

if [[ "$ENABLE_THINKING" == "1" ]]; then
  echo "[bench] thinking: enabled (request chat_template_kwargs.enable_thinking=true)" >&2
fi

if [[ "${BENCH_MOCK:-0}" == "1" ]]; then
  if [[ "$PP_MODE" == "fallback" ]]; then
    cat <<'EOF'

========== PROMPT-PROCESSING (fallback target=10000 prompt tokens, max_tokens=16) ==========
=== measured (1) ===
  run-1      wall=  3.20s  ttft= 2500ms  prompt_toks=  9876  PP_tok/s=3950.40

=== summary [prompt-processing] (n=1) ===
  PP tok/s       mean=3950.40   std=  0.00   CV= 0.0%   min=3950.40   max=3950.40
  TTFT          mean=  2500ms  std=    0ms  min=2500ms  max=2500ms
EOF
  else
    cat <<'EOF'

========== NARRATIVE (prompt=61 chars, max_tokens=1000) ==========
=== measured (1) ===
  run-1      wall=  4.20s  ttft=   120ms  toks=1000  wall_TPS=238.10  decode_TPS=245.10

=== summary [narrative] (n=1) ===
  wall_TPS       mean= 238.10   std=  0.00   CV= 0.0%   min=238.10   max=238.10
  decode_TPS     mean= 245.10   std=  0.00   CV= 0.0%   min=245.10   max=245.10
  TTFT          mean=   120ms  std=    0ms  min=120ms  max=120ms
  PP tok/s       mean=2843.21   std=  0.00   CV= 0.0%   min=2843.21   max=2843.21
EOF
  fi
  exit 0
fi

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

server_reasoning_on() {
  if curl -sf -m 3 "${URL}/props" 2>/dev/null | python3 -c '
import json, sys
try:
    obj = json.load(sys.stdin)
except Exception:
    sys.exit(1)

def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            lk = str(k).lower()
            if lk in {"reasoning", "enable_reasoning"}:
                if v is True or str(v).lower() in {"1", "true", "on", "yes"}:
                    return True
            if walk(v):
                return True
    elif isinstance(x, list):
        return any(walk(v) for v in x)
    return False
sys.exit(0 if walk(obj) else 1)
' >/dev/null 2>&1; then
    return 0
  fi
  if [[ -n "${CONTAINER:-}" && "${CONTAINER:-}" != "none" ]] \
     && command -v docker >/dev/null 2>&1 \
     && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker inspect "$CONTAINER" 2>/dev/null \
      | grep -Eq -- '(--reasoning[= ]+on|"--reasoning"[[:space:]]*,[[:space:]]*"on")' && return 0
  fi
  return 1
}

if [[ "$ENABLE_THINKING" != "1" ]] && server_reasoning_on; then
  echo "[bench] WARN: server appears to have reasoning enabled, but bench requests send enable_thinking=false. Use ENABLE_THINKING=1 for reasoning-on TPS." >&2
fi

# ===========================================================================
# CAPTURE LAYER — everything from here to the end of the run is ADDITIVE.
#
# A bench harness's failure mode is a PLAUSIBLE WRONG NUMBER, not a crash. The
# captures below exist because each has already produced one: a decode scrape
# that printed 0.00 next to a healthy TTFT; a cumulative cache hit rate that
# said a bigger pool was worse; a drafter reporting 0.992 acceptance that fired
# on 5 of ~20 requests. Shared with the sweep via scripts/lib/capture.sh.
#
# CONDITIONALITY CONTRACT:
#   ALWAYS        KV cache type · config fingerprint (argv + /props + moe-cache
#                 env + --moe-cache cap) · system RAM + swap · PCIe
#   IF OFFLOAD    expert-cache pool census / hits / evictions / health counters
#   IF OFFLOAD
#     AND CACHE   marginal (not cumulative) hit rate · derived RAM-read demand
# Every capture degrades to ONE `capture: <thing> unavailable (<why>)` line.
# ===========================================================================
CAP_ENABLED=0
CAP_T0=""; CAP_T1=""
CAP_DMON_PID=""; CAP_VRAM_PID=""; CAP_LINK_PID=""
CAP_VRAM_IDLE=""; CAP_CPU0=""
CAP_ARGV=""; CAP_OFFLOAD=""; CAP_CACHE_WANTED=0
CAP_WORK=""
if [[ "$CAPTURE" == "1" && -f "${ROOT_DIR}/scripts/lib/capture.sh" ]]; then
  # shellcheck source=lib/capture.sh
  source "${ROOT_DIR}/scripts/lib/capture.sh"
  CAP_ENABLED=1
  CAP_WORK="$(mktemp -d)"
  trap '[[ -n "${CAP_WORK:-}" ]] && rm -rf "$CAP_WORK"' EXIT
  cap_init "$CAP_WORK" || true

  CAP_ARGV="$(cap_proc_argv || true)"

  echo ""
  echo "========== CONFIG FINGERPRINT =========="
  echo "  endpoint       : ${URL}  (mode=${ENDPOINT})"
  echo "  model          : ${MODEL}"
  if [[ -n "${CAP_PID:-}" ]]; then
    echo "  serving pid    : ${CAP_PID}$([[ -n "$CAP_ARGV" ]] || echo '  (argv unreadable — different uid?)')"
  else
    echo "  serving pid    : unknown (set SERVER_PID=<pid> for argv/RSS/swap capture)"
  fi
  echo "  log source     : ${CAP_LOG_SOURCE:-none}${SERVER_LOG:+ (${SERVER_LOG})}"
  _v="$(cap_props_get "default_generation_settings.n_ctx" || true)"
  _s="$(cap_props_get "total_slots" || true)"
  [[ -n "$_v" ]] && echo "  served ctx     : ${_v}${_s:+  (slots=${_s})}"
  _v="$(cap_props_get "model_ftype" || true)"
  [[ -n "$_v" ]] && echo "  weights ftype  : ${_v}"
  _v="$(cap_props_get "default_generation_settings.params.speculative.types" || true)"
  [[ -n "$_v" ]] && echo "  drafter        : ${_v}"
  # ALWAYS-ON: KV cache type. Offload or not, a bench without its KV type is a rumor.
  _v="$(cap_kv_type "$CAP_ARGV" || true)"
  [[ -n "$_v" ]] && echo "  KV cache type  : ${_v}"
  # ALWAYS-ON: argv fingerprint — solves the fingerprint problem without operator input.
  if [[ -n "$CAP_ARGV" ]]; then
    _v="$(cap_argv_fingerprint "$CAP_ARGV" || true)"
    [[ -n "$_v" ]] && echo "  serving argv   : ${_v}"
  fi
  # ALWAYS-ON: the cache CONFIG, not just the resulting pool. The census
  # self-limits below the requested cap, so pool size alone can't identify an arm.
  _v="$(cap_moe_cache_config "$CAP_ARGV" || true)"
  [[ -n "$_v" ]] && echo "  moe-cache cfg  : ${_v}"
  case "$_v" in *"moe_cache_cap=<unset>"*) : ;; *) CAP_CACHE_WANTED=1 ;; esac
  # The GATE for every moe-cache capture below.
  CAP_OFFLOAD="$(cap_offload_detected "$CAP_ARGV" || true)"
  if [[ -n "$CAP_OFFLOAD" ]]; then
    echo "  CPU offload    : DETECTED via ${CAP_OFFLOAD}"
  else
    echo "  CPU offload    : not detected (moe-cache captures skipped)"
  fi

  # ALWAYS-ON (item 8): host RAM is the requirement launchers cannot detect, and
  # it is two lines from /proc. The swap leg is the free integrity item — pages of
  # the SERVING PROCESS in swap make every number in the run suspect.
  echo ""
  echo "========== SYSTEM RAM =========="
  _v="$(cap_ram_status || true)"
  [[ -n "$_v" ]] && echo "  ${_v}"
  _v="$(cap_swap_verdict || true)"
  [[ -n "$_v" ]] && echo "  swap check     : ${_v}"

  if [[ "$STREAM_CALIB" == "1" ]]; then
    echo ""
    echo "========== RAM BANDWIDTH CEILING (STREAM triad, idle box) =========="
    _sc="$(cap_stream_triad_gbps 3 || true)"
    CAP_STREAM="${_sc%% *}"
    _scw="${_sc##* }"
    if [[ -n "$CAP_STREAM" ]]; then
      echo "  triad ceiling  : ${CAP_STREAM} GB/s  (${_scw} concurrent workers, ~3 s, measured on THIS host)"
      echo "                   Sustained STREAM-triad, not a spec sheet figure. A box that is"
      echo "                   already serving reports a LOWER ceiling than an idle one — which"
      echo "                   is the honest denominator for a run measured under that load."
    fi
  fi

  # --- start the windowed samplers ----------------------------------------
  CAP_DMON_PID="$(cap_dmon_start "$CAP_WORK/dmon" || true)"
  CAP_LINK_PID="$(cap_pcie_link_sampler_start "$CAP_WORK/link" || true)"
  CAP_VRAM_PID="$(cap_vram_peak_start "$CAP_WORK/vpeak" || true)"
  CAP_VRAM_IDLE="$(cap_vram_used || true)"
  CAP_CPU0="$(cap_cpu_snap || true)"
  # Cache counters BEFORE the measured window, so the MARGINAL rate can be derived.
  if [[ -n "$CAP_OFFLOAD" ]]; then
    cap_refresh_log
    cap_moe_parse counters > "$CAP_WORK/counters0" 2>/dev/null || true
  fi
  # Line offset at the start of the measured window. Per-request scrapes
  # (acceptance, timings, bypass) count only what happened AFTER this point —
  # otherwise a log that already held a previous run's lines inflates every
  # count, and the drafter fire rate can exceed 100%.
  CAP_LOG_OFFSET=0
  [[ -n "$CAP_LOG" && -r "$CAP_LOG" ]] && CAP_LOG_OFFSET="$(command grep -c '' "$CAP_LOG" 2>/dev/null || echo 0)"
  export CAP_LINE_OFFSET="$CAP_LOG_OFFSET"
  # The prefill probe and PP fallback ask for ~16-token completions; a decode rate
  # measured over 16 tokens is start-up dominated. Exclude them from the decode
  # cross-check so it compares like with like.
  export CAP_MIN_DECODE_TOKENS="${CAP_MIN_DECODE_TOKENS:-32}"
  CAP_T0="$(date +%s)"
  export BENCH_PHASE_FILE="$CAP_WORK/phases" BENCH_SUMMARY_JSON="$CAP_WORK/summary.json"
  export BENCH_ENDPOINT="$ENDPOINT" BENCH_SHORT_EOS_FRAC="$SHORT_EOS_FRAC"
fi
export BENCH_ENDPOINT="${BENCH_ENDPOINT:-$ENDPOINT}"
export BENCH_SHORT_EOS_FRAC="${BENCH_SHORT_EOS_FRAC:-$SHORT_EOS_FRAC}"

python3 - "$URL" "$MODEL" "$WARMUPS" "$RUNS" "$QUIET" "$ONLY" \
            "$CONTAINER" "$PP_MODE" "$PP_FALLBACK_TOKENS" "$PP_MAX_TOKENS" \
            "$ENABLE_THINKING" "$FORCE_TOKENS" \
            "$PROMPT_NARR" "$MAX_TOKENS_NARR" \
            "$PROMPT_CODE" "$MAX_TOKENS_CODE" << 'PYEOF'
import json, os, random, re, shutil, string, subprocess, sys, time, urllib.request, statistics as s

(URL, MODEL, WARMUPS, RUNS, QUIET, ONLY,
 CONTAINER, PP_MODE, PP_FALLBACK_TOKENS, PP_MAX_TOKENS,
 ENABLE_THINKING, FORCE, PROMPT_NARR, MAX_NARR, PROMPT_CODE, MAX_CODE) = sys.argv[1:]
WARMUPS = int(WARMUPS); RUNS = int(RUNS); QUIET = int(QUIET) == 1
MAX_NARR = int(MAX_NARR); MAX_CODE = int(MAX_CODE)
PP_FALLBACK_TOKENS = int(PP_FALLBACK_TOKENS); PP_MAX_TOKENS = int(PP_MAX_TOKENS)
ENABLE_THINKING = ENABLE_THINKING == "1"
FORCE = int(FORCE)   # >0: force EXACTLY this many output tokens (max+min+ignore_eos)

# --- capture-layer plumbing (all optional; absent => historical behaviour) ---
# ENDPOINT=chat (default) keeps the historical /v1/chat/completions path with the
# model's template applied. ENDPOINT=completion drives raw /v1/completions with NO
# template — for base models, and for isolating template overhead on a chat model.
ENDPOINT_MODE = os.environ.get("BENCH_ENDPOINT", "chat")
PHASE_FILE = os.environ.get("BENCH_PHASE_FILE", "")
SUMMARY_JSON = os.environ.get("BENCH_SUMMARY_JSON", "")
try:
    SHORT_EOS_FRAC = float(os.environ.get("BENCH_SHORT_EOS_FRAC", "0.25"))
except ValueError:
    SHORT_EOS_FRAC = 0.25
SUMMARY = {"shapes": {}, "endpoint": ENDPOINT_MODE}


def progress(msg):
    """Item 6: one line per run, FLUSHED, on stderr.

    Section summaries buffer to section end (correct — a partial table is worse
    than none), but a prefill probe at depth can then sit silent for 10+ minutes
    with no way to tell it apart from a hang. stderr keeps stdout's shape intact
    for anything parsing it."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def mark_phase(label, t0, t1):
    """Record a measured window so PCIe/dmon samples can be sliced by PHASE.

    Decode and prefill PCIe traffic differ by ~100x on an offloaded MoE, so a
    whole-run mean describes neither phase."""
    if not PHASE_FILE:
        return
    try:
        with open(PHASE_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{label} {t0:.0f} {t1:.0f}\n")
    except OSError:
        pass


_TOKEN_RATIO = [None]


def token_ratio():
    """Item 7: measured tokens-per-word for THIS tokenizer.

    The haystack builders sized filler by WORD count, which is a per-tokenizer
    guess: a "40K" warm-up tokenized to 55K under DeepSeek's tokenizer (+37%).
    Prefer the server's own /tokenize; fall back to one tiny max_tokens=1 request
    and read usage.prompt_tokens; fall back to the word heuristic (ratio 1.0),
    which is exactly the historical behaviour."""
    if _TOKEN_RATIO[0] is not None:
        return _TOKEN_RATIO[0]
    sample = ("club3090 prompt processing calibration filler with stable token shape. "
              "This sentence is intentionally plain so tokenizer variance stays modest. ") * 20
    words = max(len(sample.split()), 1)
    ratio = None
    try:
        body = json.dumps({"content": sample}).encode()
        req = urllib.request.Request(f"{URL}/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            obj = json.loads(r.read().decode("utf-8", errors="ignore"))
        toks = obj.get("tokens", obj if isinstance(obj, list) else [])
        if isinstance(toks, list) and toks:
            ratio = len(toks) / words
    except Exception:
        ratio = None
    if ratio is None:
        try:
            _w, _t, _k, ptoks = run_once(sample, 1)
            if ptoks > words:            # a real count, not the word-split fallback
                ratio = ptoks / words
        except Exception:
            ratio = None
    if ratio is None or ratio <= 0:
        progress("[bench] capture: token-ratio probe unavailable (no /tokenize, probe failed) "
                 "— haystacks fall back to the word-count heuristic")
        ratio = 1.0
    else:
        progress(f"[bench] capture: tokenizer ratio {ratio:.2f} tok/word "
                 f"(haystacks sized by measured tokens, not words)")
    _TOKEN_RATIO[0] = ratio
    return ratio


def run_once(prompt, max_tokens):
    # FORCE>0: force EXACTLY FORCE output tokens (overrides the per-prompt cap) so the
    # model can't self-terminate early — required to measure sustained throughput at a
    # chosen output size on diffusion LMs (which stop ~1-2K words otherwise).
    mt = FORCE if FORCE > 0 else max_tokens
    req_body = {
        "model": MODEL,
        "max_tokens": mt,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ENDPOINT_MODE == "completion":
        # RAW completion: no chat template, no thinking kwargs (they are a chat
        # concept). Deliberately not the default — switching endpoints changes
        # the prompt the model actually sees, so numbers are NOT comparable
        # across modes.
        req_body["prompt"] = prompt
        path = "/v1/completions"
    else:
        req_body["messages"] = [{"role": "user", "content": prompt}]
        req_body["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
        path = "/v1/chat/completions"
    if FORCE > 0:
        req_body["min_tokens"] = FORCE
        req_body["ignore_eos"] = True
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(f"{URL}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                # /v1/completions puts the text at choices[0].text, not in a delta
                content = (delta.get("content") or delta.get("reasoning_content")
                           or choices[0].get("text"))
                if content and ttft is None:
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    if not prompt_tokens:
        prompt_tokens = max(1, len(prompt.split()))
    return wall, ttft, completion_tokens, prompt_tokens

def fmt(label, wall, ttft, toks):
    decode_t = max(wall - ttft, 1e-6)
    wtps = toks / wall if wall > 0 else 0
    dtps = toks / decode_t
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  toks={toks:>4d}  wall_TPS={wtps:6.2f}  decode_TPS={dtps:6.2f}"
    return wtps, dtps, ttft, line

def fmt_pp(label, wall, ttft, prompt_tokens):
    pp = prompt_tokens / max(ttft, 1e-6)
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  prompt_toks={prompt_tokens:>6d}  PP_tok/s={pp:7.2f}"
    return pp, ttft, line

def stats(name, xs, unit=""):
    m = s.mean(xs)
    sd = s.stdev(xs) if len(xs) > 1 else 0
    cv = (sd / m * 100) if m > 0 else 0
    return f"  {name:<14s} mean={m:7.2f}{unit}   std={sd:6.2f}   CV={cv:4.1f}%   min={min(xs):.2f}   max={max(xs):.2f}"

def scrape_server_log_prefill(n):
    """Item 4: the BARE-METAL half of the engine-side scrape.

    `CONTAINER=none` runs report "log scrape unavailable" and every engine-side
    number then has to be hand-extracted from a file that was sitting there the
    whole time. With SERVER_LOG set, read llama.cpp's own `prompt eval time`
    lines instead."""
    path = os.environ.get("SERVER_LOG", "")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    vals = [float(m.group(1)) for m in
            re.finditer(r"prompt eval time.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", text)]
    return vals[-max(n, 1):]


def scrape_prompt_throughput(container, n):
    vals = scrape_server_log_prefill(n)
    if vals:
        return vals
    if not container or container == "none" or shutil.which("docker") is None:
        return []
    try:
        proc = subprocess.run(
            ["docker", "logs", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    vals = [
        float(m.group(1))
        for m in re.finditer(r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", proc.stdout)
    ]
    return vals[-max(n, 1):]

def run_set(label, prompt, max_tokens):
    print(f"\n========== {label.upper()} (prompt={len(prompt)} chars, max_tokens={max_tokens}) ==========")
    print(f"=== warmups ({WARMUPS}) ===")
    for i in range(WARMUPS):
        try:
            w, t, k, _ = run_once(prompt, max_tokens)
            _, _, _, line = fmt(f"warm-{i+1}", w, t, k)
            if not QUIET:
                print(line)
        except Exception as e:
            print(f"  warm-{i+1}  FAIL: {e}")
    print(f"\n=== measured ({RUNS}) ===")
    walls, decodes, ttfts, toks_seen = [], [], [], []
    errors = 0
    mt = FORCE if FORCE > 0 else max_tokens
    phase_t0 = time.time()
    for i in range(RUNS):
        try:
            w, t, k, _ = run_once(prompt, max_tokens)
            wtps, dtps, ttft, line = fmt(f"run-{i+1}", w, t, k)
            if not QUIET:
                print(line)
            progress(f"[bench] {label} run {i+1}/{RUNS}: {k} tok in {w:.1f}s "
                     f"({dtps:.2f} decode TPS, ttft {ttft*1000:.0f}ms)")
            walls.append(wtps); decodes.append(dtps); ttfts.append(ttft)
            toks_seen.append(k)
        except Exception as e:
            errors += 1
            print(f"  run-{i+1}  FAIL: {e}")
            progress(f"[bench] {label} run {i+1}/{RUNS}: FAIL: {e}")
    mark_phase("decode", phase_t0, time.time())
    # Item 2: a chat-tuned model answers briefly and EOSes. The per-shape stats
    # then silently describe whichever one or two runs happened to run long — a
    # 5-run shape degenerates to n=1 while still printing "n=5". Count the runs
    # that are long enough to be worth a decode rate and say so.
    floor = max(1, int(mt * SHORT_EOS_FRAC))
    usable = [k for k in toks_seen if k >= floor]
    SUMMARY["shapes"][label] = {
        "n": len(walls), "n_usable": len(usable), "errors": errors,
        "max_tokens": mt, "short_eos_floor": floor,
        "tokens": toks_seen,
        "wall_tps_mean": (s.mean(walls) if walls else 0.0),
        "decode_tps_mean": (s.mean(decodes) if decodes else 0.0),
        "ttft_mean_ms": (s.mean(ttfts) * 1000 if ttfts else 0.0),
    }
    if walls:
        print(f"\n=== summary [{label}] (n={len(walls)}) ===")
        print(stats("wall_TPS",   walls))
        print(stats("decode_TPS", decodes))
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std={s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0:5.0f}ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
        print(f"  n-usable      {len(usable)}/{len(walls)}  (runs with >= {floor} tokens, "
              f"{SHORT_EOS_FRAC:.0%} of max_tokens={mt})")
        if len(usable) < len(walls):
            short = len(walls) - len(usable)
            print(f"  ⚠ WARN: {short}/{len(walls)} run(s) EOSed well below max_tokens "
                  f"(tokens={toks_seen}). Decode stats for this shape rest on "
                  f"{len(usable)} run(s), not {len(walls)} — treat the CV as unreliable.")
            print("           Fix: FORCE_TOKENS=<n> to force a fixed output length, or use a "
                  "prompt this model answers at length.")
            progress(f"[bench] WARN {label}: only {len(usable)}/{len(walls)} runs usable "
                     f"(short EOS) — see the summary block")
        if PP_MODE == "log":
            pp_vals = scrape_prompt_throughput(CONTAINER, len(walls))
            if pp_vals:
                print(stats("PP tok/s", pp_vals))
            else:
                print("  PP tok/s       n/a (engine log scrape unavailable; use PP=1 for the "
                      "long-prompt fallback, or SERVER_LOG=<path> on bare metal)")
        else:
            print("  PP tok/s       n/a (long-prompt fallback below)")

def long_prompt(target_tokens):
    filler = (
        "club3090 prompt processing calibration filler with stable token shape. "
        "This sentence is intentionally plain so tokenizer variance stays modest. "
    )
    # Item 7: size by MEASURED tokens, not words. The word heuristic is a
    # per-tokenizer guess and a "40K" haystack tokenized to 55K on DeepSeek.
    words_per_chunk = max(len(filler.split()), 1)
    chunks = max(1, int(target_tokens / (words_per_chunk * token_ratio())))
    return (
        "Read the following calibration text. Reply with one concise sentence summarizing its purpose.\n\n"
        + filler * chunks
    )

def run_pp_fallback():
    prompt = long_prompt(PP_FALLBACK_TOKENS)
    print(
        f"\n========== PROMPT-PROCESSING "
        f"(fallback target={PP_FALLBACK_TOKENS} prompt tokens, max_tokens={PP_MAX_TOKENS}) =========="
    )
    print("=== measured (1) ===")
    pp_vals, ttfts = [], []
    phase_t0 = time.time()
    try:
        w, t, _k, prompt_tokens = run_once(prompt, PP_MAX_TOKENS)
        pp, ttft, line = fmt_pp("run-1", w, t, prompt_tokens)
        print(line)
        progress(f"[bench] prompt-processing run 1/1: {prompt_tokens} prompt tok, "
                 f"{pp:.1f} PP tok/s, ttft {ttft*1000:.0f}ms")
        pp_vals.append(pp); ttfts.append(ttft)
    except Exception as e:
        print(f"  run-1      FAIL: {e}")
        progress(f"[bench] prompt-processing run 1/1: FAIL: {e}")
    mark_phase("prefill", phase_t0, time.time())
    if pp_vals:
        print("\n=== summary [prompt-processing] (n=1) ===")
        print(stats("PP tok/s", pp_vals))
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std=    0ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")

def prefill_prompt(target_tokens, salt):
    """Cache-busted long prompt (catalog-baselines 2c).

    Composes serve ``enable_prefix_caching=True`` — an identical prompt
    re-measures the PREFIX-CACHE HIT, not prefill.  vLLM's prefix cache is
    block-chained, so a unique FIRST line breaks the whole chain; the salt is
    also woven through the filler as belt-and-braces against block-aligned
    partial hits."""
    filler = (
        f"calibration {salt} segment with stable token shape for the prefill probe. "
        "This sentence is intentionally plain so tokenizer variance stays modest. "
    )
    # Item 7: divide by the MEASURED tok/word ratio so the warm-up lands on the
    # requested depth. The measured runs recalibrate from the warm-up's actual
    # count regardless — this stops the WARM-UP from overshooting (the "40K" that
    # tokenized to 55K), which is the run nothing corrects.
    words_per_chunk = max(len(filler.split()), 1)
    chunks = max(1, int(target_tokens / (words_per_chunk * token_ratio())))
    return (
        f"[probe {salt}] Read the following calibration text. Reply with the single word DONE.\n\n"
        + filler * chunks
    )

def run_prefill_probe():
    """Canonical two-depth prefill/TTFT probe (catalog-baselines 2c): warm
    engine, FRESH salted haystack per request, prefill tok/s = prompt_tokens/TTFT.
    The deep anchor pairs with the verify-stress ladder's nearest rung for
    anchor calibration (agreement certifies the ladder's whole depth curve).
    A depth the served context can't hold is SKIPPED with a note."""
    depths = [int(x) for x in os.environ.get("PREFILL_DEPTHS", "10000,90000").split(",") if x.strip()]
    n = max(1, int(os.environ.get("PREFILL_RUNS", "3")))

    def salt():
        return "".join(random.choices(string.ascii_lowercase, k=8))

    for target in depths:
        label = f"prefill-{target // 1000}k"
        print(
            f"\n========== {label.upper()} (target={target} prompt tokens, "
            f"max_tokens={PP_MAX_TOKENS}, cache-busted: fresh haystack per run) =========="
        )
        print("=== warmups (1) ===")
        # The warmup doubles as the TOKEN CALIBRATOR: the word-count heuristic
        # overshoots (salted filler tokenizes ~1.3 tok/word), so the measured
        # runs scale their request by the warmup's actual/target ratio — the
        # depth label stays honest to within a few percent.
        request_tokens = target
        try:
            w, t, _k, ptoks = run_once(prefill_prompt(target, salt()), PP_MAX_TOKENS)
            _pp, _t, line = fmt_pp("warm-1", w, t, ptoks)
            if not QUIET:
                print(line)
            if ptoks > 0:
                request_tokens = max(200, int(target * target / ptoks))
                print(f"  [calibrated: {ptoks} tok at request={target} → measured runs request {request_tokens}]")
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ("400", "maximum context", "max_model_len", "context length", "context window")):
                print(f"  SKIP: {target} tokens exceeds the served context ({msg[:100]})")
                continue
            print(f"  warm-1     FAIL: {e}")
        print(f"\n=== measured ({n}) ===")
        pps, ttfts = [], []
        phase_t0 = time.time()
        for i in range(n):
            try:
                w, t, _k, ptoks = run_once(prefill_prompt(request_tokens, salt()), PP_MAX_TOKENS)
                pp, ttft, line = fmt_pp(f"run-{i+1}", w, t, ptoks)
                if not QUIET:
                    print(line)
                progress(f"[bench] {label} run {i+1}/{n}: {ptoks} prompt tok, "
                         f"{pp:.1f} prefill tok/s, ttft {ttft*1000:.0f}ms")
                pps.append(pp)
                ttfts.append(ttft)
            except Exception as e:
                print(f"  run-{i+1}    FAIL: {e}")
                progress(f"[bench] {label} run {i+1}/{n}: FAIL: {e}")
        mark_phase("prefill", phase_t0, time.time())
        if pps:
            print(f"\n=== summary [{label}] (n={len(pps)}) ===")
            # prompt_tokens/TTFT = the CLIENT-OBSERVED rate (includes
            # tokenization + transfer + scheduling) — the user-truth number.
            print(stats("prefill tok/s", pps))
            print(
                f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  "
                f"std={(s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0):5.0f}ms  "
                f"min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms"
            )
            # Engine-internal rate from the vLLM stats log — windowed (tokens
            # per 10s stats window), so treat as approximate; the gap between
            # this and the client-observed rate is tokenization/transfer/
            # scheduling overhead, not prefill compute.
            if PP_MODE == "log":
                vals = scrape_prompt_throughput(CONTAINER, len(pps) * 2)
                vals = [v for v in vals if v > 0][-len(pps):]
                if vals:
                    print(stats("PP tok/s (engine log, windowed)", vals))
            SUMMARY["shapes"][label] = {
                "n": len(pps), "n_usable": len(pps), "errors": 0,
                "prefill_tps_mean": s.mean(pps), "ttft_mean_ms": s.mean(ttfts) * 1000,
            }

if ONLY in ("both", "narr"):
    run_set("narrative", PROMPT_NARR, MAX_NARR)
if ONLY in ("both", "code"):
    run_set("code", PROMPT_CODE, MAX_CODE)
if PP_MODE == "fallback":
    run_pp_fallback()
if os.environ.get("PREFILL_PROBE", "1") == "1":
    run_prefill_probe()

# Hand the measured totals to the shell's capture layer. stdout keeps its exact
# historical shape; the cross-checks (item 1) read this instead of re-parsing it.
if SUMMARY_JSON:
    try:
        tot = sum(sum(v.get("tokens", [])) for v in SUMMARY["shapes"].values())
        SUMMARY["total_tokens"] = tot
        SUMMARY["total_errors"] = sum(v.get("errors", 0) for v in SUMMARY["shapes"].values())
        SUMMARY["token_ratio"] = _TOKEN_RATIO[0]
        tmp = SUMMARY_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(SUMMARY, fh)
        os.replace(tmp, SUMMARY_JSON)     # atomic; a failed encode leaves no half-file
    except Exception as e:
        sys.stderr.write(f"[bench] capture: summary hand-off failed ({e})\n")
PYEOF

# ===========================================================================
# CAPTURE REPORT — everything below is additive to the historical output.
# ===========================================================================
if (( CAP_ENABLED )); then
  CAP_T1="$(date +%s)"
  CAP_ELAPSED=$(( CAP_T1 - CAP_T0 )); (( CAP_ELAPSED < 1 )) && CAP_ELAPSED=1
  cap_stop_pid "$CAP_DMON_PID"; cap_stop_pid "$CAP_LINK_PID"; cap_stop_pid "$CAP_VRAM_PID"
  cap_refresh_log
  # A run that produced no phase markers (every request failed) still needs a
  # window, or the PCIe slice would silently be empty rather than reported empty.
  [[ -s "$CAP_WORK/phases" ]] || printf 'decode %s %s\n' "$CAP_T0" "$CAP_T1" > "$CAP_WORK/phases"

  # ---- PCIe: measured used + queried available (item 10) -------------------
  echo ""
  echo "========== CAPTURE: PCIe =========="
  if [[ ! -s "$CAP_WORK/dmon" ]]; then
    cap_note_unavailable "PCIe rx/tx sampling" \
      "nvidia-smi dmon produced no output (no GPU, or dmon not supported here)"
  else
    for _ph in decode prefill; do
      command grep -q "^${_ph} " "$CAP_WORK/phases" 2>/dev/null || continue
      if _out="$(cap_dmon_phase "$CAP_WORK/dmon" "$CAP_WORK/phases" "$_ph" 2>/dev/null)"; then
        while IFS= read -r _l; do [[ -n "$_l" ]] && echo "  ${_l}"; done <<< "$_out"
      else
        # Reported, never silently omitted: a phase shorter than the 1 Hz sample
        # period genuinely has no samples, and that is a fact about the run.
        echo "  ${_ph}: no dmon samples inside the measured windows (phase shorter than the 1 Hz sample period)"
      fi
    done
    echo "  note: decode and prefill PCIe traffic differ by ~100x on an offloaded MoE —"
    echo "        a whole-run mean describes neither. 1 Hz under-reads bursts, so the"
    echo "        PEAK is reported next to the mean, not instead of it."
  fi
  # Link capability, sampled UNDER LOAD: an idle link downshifts to Gen1 by
  # design, so an idle reading manufactures false alarms.
  if _out="$(cap_pcie_link_under_load "$CAP_WORK/link" 2>/dev/null)"; then
    echo "  link (sampled under load, best observed):"
    while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_out"
    # Utilisation% against a STATED denominator — a bandwidth percentage with an
    # unstated denominator is not a number anyone can check.
    _g="$(printf '%s\n' "$_out" | command grep -oE 'gen=[0-9]+' | head -1 | cut -d= -f2 || true)"
    _w="$(printf '%s\n' "$_out" | command grep -oE 'width=[0-9]+' | head -1 | cut -d= -f2 || true)"
    _prac="$(cap_pcie_practical_mbps "${_g:-0}" "${_w:-0}" 2>/dev/null || true)"
    if [[ -n "$_prac" ]]; then
      _rx="$(cap_dmon_phase "$CAP_WORK/dmon" "$CAP_WORK/phases" decode 2>/dev/null \
             | command grep -m1 ' all ' | command grep -oE 'rx_peak=[0-9]+' | cut -d= -f2 || true)"
      echo "    denominator: ~${_prac} MB/s practical for a negotiated gen${_g} x${_w} link"
      echo "                 (per-lane raw x width x 0.85; NOT a theoretical peak)"
      [[ -n "$_rx" ]] && awk -v a="$_rx" -v b="$_prac" 'BEGIN{
        if (b > 0) printf "    decode rx peak = %s MB/s = %.1f%% of that denominator\n", a, a*100/b }'
    fi
  else
    cap_note_unavailable "PCIe link state" "nvidia-smi could not report pcie.link.* under load"
  fi

  # ---- VRAM triplet + leak delta (item 11b) --------------------------------
  echo ""
  echo "========== CAPTURE: VRAM =========="
  CAP_VRAM_POST="$(cap_vram_used || true)"
  CAP_VRAM_PEAK="$(cat "$CAP_WORK/vpeak" 2>/dev/null || true)"
  if [[ -n "$CAP_VRAM_IDLE" ]]; then
    echo "  idle=${CAP_VRAM_IDLE} MiB  peak=${CAP_VRAM_PEAK:-n/a} MiB  post=${CAP_VRAM_POST:-n/a} MiB"
    if [[ -n "$CAP_VRAM_POST" ]]; then
      # A free soak-lite leak check: VRAM that did not come back is the cheapest
      # early warning there is for a memory-policy regression.
      echo "  leak delta (post - idle) = $(( CAP_VRAM_POST - CAP_VRAM_IDLE )) MiB"
    fi
    while IFS= read -r _l; do [[ -n "$_l" ]] && echo "  ${_l} MiB (post-run)"; done \
      < <(cap_vram_per_device 2>/dev/null || true)
  else
    cap_note_unavailable "VRAM triplet" "nvidia-smi not available"
  fi

  # ---- CPU (item 11d) ------------------------------------------------------
  CAP_CPU1="$(cap_cpu_snap || true)"
  if [[ -n "$CAP_CPU0" && -n "$CAP_CPU1" ]]; then
    CAP_CPU_PCT="$(cap_cpu_pct "$CAP_CPU0" "$CAP_CPU1" || true)"
    # cpu% is the missing half of offload bottleneck forensics: a low sm% next to
    # a high cpu% says the GPU is waiting on the host, not the other way round.
    [[ -n "$CAP_CPU_PCT" ]] && echo "  host CPU busy across the run: ${CAP_CPU_PCT}%"
  fi

  # ---- draft acceptance WITH fire rate (item 11a) --------------------------
  echo ""
  echo "========== CAPTURE: DRAFT ACCEPTANCE =========="
  # Requests in the SAME windowed log the acceptance lines came from. Deriving
  # the denominator from the summary JSON instead counts only measured runs while
  # the numerator counted warm-ups too — which produced a 142% fire rate.
  # Field-exact extraction. `grep -oE 'n=[0-9]+'` ALSO matches the tail of
  # "mean=100.00", which produced a multi-line value and silently skipped the
  # whole fire-rate block.
  CAP_REQS="$(cap_moe_parse timings 2>/dev/null \
              | awk '$1=="decode"{for(i=2;i<=NF;i++) if($i ~ /^n=/){split($i,a,"="); print a[2]; exit}}' || true)"
  if cap_log_available "draft acceptance"; then
    if _acc="$(cap_moe_parse acceptance 2>/dev/null)"; then
      _fired="$(printf '%s' "$_acc" | command grep -oE 'fired=[0-9]+' | cut -d= -f2)"
      _reqs="${CAP_REQS:-}"
      echo "  ${_acc}"
      if [[ -n "$_reqs" && "${_reqs:-0}" -gt 0 ]]; then
        awk -v f="${_fired:-0}" -v r="$_reqs" 'BEGIN{
          if (f > r) f = r
          printf "  fire rate: fired on %d of %d requests in the measured window (%.1f%%)\n", f, r, f*100/r
          if (r > 0 && f * 100 / r < 50)
            print "  ⚠ WARN: acceptance WITHOUT a fire rate is a deception. This drafter was\n" \
                  "          idle on most requests, so its acceptance describes a minority of\n" \
                  "          the run and does NOT predict the end-to-end speed-up."
        }'
      fi
    else
      echo "  no drafter output in the log (spec-dec off, or the engine does not log acceptance)"
    fi
  fi

  # ---- engine-side timings + cross-check (item 1b, item 4) -----------------
  echo ""
  echo "========== CAPTURE: ENGINE-SIDE TIMINGS =========="
  if cap_log_available "engine-side timings"; then
    if _tim="$(cap_moe_parse timings 2>/dev/null)"; then
      while IFS= read -r _l; do
        case "$_l" in
          "") ;;
          dropped_short_decode*)
            echo "  ${_l}  (short probe completions excluded from the decode cross-check)" ;;
          *)  echo "  ${_l}  (tok/s, from the server's own print_timing)" ;;
        esac
      done <<< "$_tim"
      _eng="$(printf '%s\n' "$_tim" | command grep -m1 '^decode ' | command grep -oE 'median=[0-9.]+' | cut -d= -f2 || true)"
      _cli="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
xs = [v.get("decode_tps_mean", 0) for v in d.get("shapes", {}).values() if v.get("decode_tps_mean")]
print(f"{sum(xs)/len(xs):.2f}" if xs else "")
' "$CAP_WORK/summary.json" 2>/dev/null || true)"
      if [[ -n "$_eng" && -n "$_cli" ]]; then
        _div="$(cap_divergence_pct "$_eng" "$_cli" || true)"
        if [[ -n "$_div" ]]; then
          echo "  cross-check: client-observed ${_cli} vs engine-side ${_eng} tok/s -> ${_div}% divergence"
          awk -v d="$_div" 'BEGIN{ if (d+0 > 5) print "  ⚠ WARN: >5% divergence. One of these two is not measuring what its label\n          says. State WHICH number a card quotes, and why." }'
        fi
      fi
    else
      echo "  no print_timing lines in the log (engine may not emit them at this verbosity)"
    fi
  fi

  # ---- moe-cache: GATED on CPU-offload detection (item 3) ------------------
  echo ""
  echo "========== CAPTURE: EXPERT CACHE (moe-cache) =========="
  if [[ -z "$CAP_OFFLOAD" ]]; then
    echo "  skipped: no CPU offload detected — nothing for an expert cache to cache."
  elif ! cap_log_available "moe-cache telemetry"; then
    : # the notice is already printed by cap_log_available
  else
    CAP_CACHE_OK=1
    if _cen="$(cap_moe_parse census 2>/dev/null)"; then
      echo "  pool census (per device — a device can hold MULTIPLE pools; slots/bytes are summed):"
      while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_cen"
    else
      _cen=""
    fi
    # #824: the '[moe-cache] enabled:' banner still prints when only SOME devices
    # got a pool. Ground truth is the count of devices WITH a pool vs the device
    # count — not the banner, and not the raw pool-line count (multi-pool devices
    # make that count pass while a device has nothing).
    if _pd="$(cap_moe_parse pooldev 2>/dev/null)"; then
      _devs="$(printf '%s' "$_pd" | command grep -oE 'devices=[0-9]+' | cut -d= -f2)"
      _enb="$(printf '%s' "$_pd" | command grep -oE 'enabled=[0-9]+' | cut -d= -f2)"
      _nob="$(printf '%s' "$_pd" | command grep -oE 'nobudget=[0-9]+' | cut -d= -f2)"
      echo "  allocation: ${_pd}  (GPUs on this host: ${CAP_NGPU})"
      if (( CAP_CACHE_WANTED )) && [[ "${_devs:-0}" -lt "${CAP_NGPU:-0}" ]]; then
        CAP_CACHE_OK=0
        echo "  ⚠ WARN: a cache was REQUESTED but only ${_devs:-0} of ${CAP_NGPU} devices hold a pool."
        if [[ "${_nob:-0}" != "0" ]]; then
          echo "          reason: a device had no budget after its reserve. Lower RESERVE_MB."
        elif [[ "${_enb:-0}" == "1" ]]; then
          echo "          the 'enabled:' banner still printed — this is the #824 half-cached state."
        fi
        echo "          This run measured a HALF-CACHED path; do not quote it for a cache conclusion."
      fi
    fi
    cap_moe_parse counters > "$CAP_WORK/counters1" 2>/dev/null || true
    if [[ -s "$CAP_WORK/counters1" ]]; then
      [[ -s "$CAP_WORK/counters0" ]] || : > "$CAP_WORK/counters0"
      if _mar="$(cap_marginal_rates "$CAP_WORK/counters0" "$CAP_WORK/counters1" 2>/dev/null)"; then
        echo "  hit rate, per device (MARGINAL = across the measured window only):"
        while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_mar"
        echo "    note: the CUMULATIVE column embeds the cold-fill phase and is NOT"
        echo "          boot-comparable — a BIGGER pool shows a LOWER cumulative rate on"
        echo "          the same traffic. Compare boots on the MARGINAL column only."
        echo "    note: the per-device split is signal, not noise — a CUDA0/CUDA1 hit"
        echo "          asymmetry is routing entropy, and averaging it away hides it."
      fi
      echo "  health counters: $(cap_health_verdict "$CAP_WORK/counters1" || echo 'unreadable')"

      # ---- derived RAM-read demand (item 9a) — offload AND cache ----------
      if [[ -n "$_cen" ]]; then
        _exp="$(printf '%s\n' "$_cen" | command grep -oE 'expert_kib=[0-9]+' | cut -d= -f2 | head -1 || true)"
        _miss="$(printf '%s\n' "${_mar:-}" | awk '{
          for (i = 1; i <= NF; i++) {
            if ($i ~ /^dhits=/)    { split($i, a, "="); h += a[2] }
            if ($i ~ /^dlookups=/) { split($i, b, "="); t += b[2] }
          }
        } END { m = t - h; print (m > 0 ? m : 0) }' || true)"
        if [[ -n "$_exp" && -n "$_miss" && "${_miss:-0}" -gt 0 ]]; then
          _ram="$(cap_ram_rd_mbps "$_miss" "$_exp" "$CAP_ELAPSED" || true)"
          if [[ -n "$_ram" ]]; then
            echo "  derived host-RAM read demand on the miss path: ~${_ram} MB/s"
            echo "    = ${_miss} misses x ${_exp} KiB / ${CAP_ELAPSED}s"
            echo "    ⚠ DERIVED, NOT A PERF COUNTER. A lower bound on the miss path only"
            echo "      (ignores KV traffic, activations and prefetch). IMC/uncore counters"
            echo "      are root-gated on bare metal and absent in VM guests, so this is the"
            echo "      portable substitute — never quote it as measured bandwidth."
            if [[ -n "${CAP_STREAM:-}" ]]; then
              awk -v d="$_ram" -v c="${CAP_STREAM}" 'BEGIN{
                if (c > 0) printf "    ~%.0f of ~%.0f GB/s STREAM-triad ceiling (%.0f%%)\n", d/1000, c, d/10/c }'
            else
              echo "    (STREAM_CALIB=1 adds a ~3 s triad ceiling so this can be stated as a fraction)"
            fi
          fi
        fi
      fi
    fi
    _byp="$(cap_moe_parse bypass 2>/dev/null || echo 0)"
    echo "  bypass counter: ${_byp:-0}$( [[ "${_byp:-0}" != "0" ]] && echo '  ⚠ a decode DECLINED the cache — this run does not measure the cached path' )"
  fi

  # ---- integrity panel (items 1, 3c, 8, 11c) -------------------------------
  echo ""
  echo "========== CAPTURE: INTEGRITY =========="
  CAP_TOKENS="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
print(d.get("total_tokens", 0))
' "$CAP_WORK/summary.json" 2>/dev/null || echo 0)"
  CAP_ERRS="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
print(d.get("total_errors", 0))
' "$CAP_WORK/summary.json" 2>/dev/null || echo 0)"
  CAP_TPS="${_cli:-0}"
  CAP_STATUS="$(cap_status_classify "${CAP_TOKENS:-0}" "${CAP_TPS:-0}" "${CAP_ERRS:-0}" \
                  "$CAP_CACHE_WANTED" "${CAP_CACHE_OK:-1}" "${_byp:-0}")"
  echo "  status: ${CAP_STATUS}"
  case "$CAP_STATUS" in
    NO_TOKENS)
      # The exact "plausible wrong number" class #824 catalogued: wall_TPS and
      # decode_TPS printed 0.00 while TTFT printed fine, and nothing warned.
      echo "  ⚠⚠ NO_TOKENS: the run measured ${CAP_TOKENS:-0} completion tokens at ${CAP_TPS:-0} TPS."
      echo "     A TPS of 0.00 next to a healthy-looking TTFT is a SCRAPE failure, not a"
      echo "     slow model. Do NOT quote any throughput number from this run."
      echo "     Check: did the model EOS immediately? did usage arrive in the stream?"
      echo "     (stream_options.include_usage), and does the endpoint support it?" ;;
    REQ_ERRORS)
      echo "  ⚠⚠ REQ_ERRORS: ${CAP_ERRS} request(s) failed. Every aggregate above is over the"
      echo "     survivors only." ;;
    CACHE_DISABLED)
      echo "  ⚠⚠ CACHE_DISABLED: a cache was requested but is not (fully) allocated — see the"
      echo "     expert-cache section. This run measured the no-cache or half-cached path." ;;
    INVALID_BYPASS)
      echo "  ⚠⚠ INVALID_BYPASS: ${_byp} decode(s) declined the cache." ;;
    OK)
      echo "  (no capture-level defect detected — this is NOT a statement about the numbers'"
      echo "   quality, only that the measurement machinery did its job)" ;;
  esac
  echo "  swap check   : $(cap_swap_verdict 2>/dev/null || echo 'unavailable')"
  if [[ -s "$CAP_WORK/counters1" ]]; then
    echo "  cache health : $(cap_health_verdict "$CAP_WORK/counters1" 2>/dev/null || echo 'unavailable')"
  fi
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
for name, v in d.get("shapes", {}).items():
    n, u = v.get("n", 0), v.get("n_usable", 0)
    floor = v.get("short_eos_floor", 0)
    if n and u < n:
        print(f"  ⚠ n-usable   : {name} {u}/{n} runs above the short-EOS floor "
              f"({floor} tok) - the CV for this shape is not trustworthy")
' "$CAP_WORK/summary.json" || true
fi

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# MTP / spec-decode stats — only when running against a Docker container we own.
# Skipped silently in endpoint-first mode (CONTAINER=none).
if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 \
   && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi
