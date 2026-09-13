#!/usr/bin/env python3
"""Engine-agnostic long-context probe. Driven by scripts/deep-context-probe.sh.

Stdlib only, OpenAI-compatible chat API — works against vLLM and SGLang alike.
See the wrapper's header for why this exists (club-3090#1259) and how to read it.
"""
import json, os, sys, time, urllib.error, urllib.request

URL, MODEL, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
TARGET_CTX, TURN_TOKENS = int(sys.argv[4]), int(sys.argv[5])
SESSIONS, SESSION_CTX, KV_POOL = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
BASE = URL.rstrip("/")

# ---------------------------------------------------------------------------
# WHERE PREFIX REUSE IS OBSERVABLE — measured 2026-09-13, not assumed.
#
# `usage.prompt_tokens_details.cached_tokens` is gated on a SERVER flag on both
# engines, and the gate is identical for streaming and non-streaming:
#   SGLang v0.5.19  --enable-cache-report   (usage_processor.py:40/80). Even with
#                   the flag the field is OMITTED when the count is 0. 0 of the 15
#                   shipped sglang composes pass it.
#   vLLM   v0.29.0  --enable-prompt-tokens-details (launchers/cli_args.py:132,
#                   default False). 20 shipped vllm composes pass it.
# So an ABSENT field is ambiguous — "0 cached" or "not reported" — and reading it
# as 0 is exactly the false-clean this probe exists to avoid. On this rig a
# 4,212-token prompt repeated came back TTFT 3.15s -> 0.10s, `#cached-token: 4160`
# in the engine log, and `prompt_tokens_details: null` in BOTH stream modes.
#
# Fallback: both engines publish a Prometheus counter whose DELTA across one
# request is the tokens served from prefix cache. It is engine-wide (concurrent
# traffic counts too), so the column is labelled with its source:
#   SGLang  sglang:realtime_tokens_total{mode="prefill_cache"}  needs --enable-metrics (13/15 composes). Measured.
#   vLLM    vllm:prefix_cache_hits_total   v1/metrics/loggers.py:596 — source-verified, NOT live-tested here.
# A reuse self-test right after calibration (same prompt twice) decides which
# source is live. When neither is, the column renders n/a and names the flag.
# ---------------------------------------------------------------------------
_REUSE_COUNTERS = [  # (exposed counter name, required label fragment or None)
    ("sglang:realtime_tokens_total", 'mode="prefill_cache"'),
    ("vllm:prefix_cache_hits_total", None),
]
REUSE = {"source": None, "counter": None}   # source: "usage" | "metrics" | None

FLAG_HINT = ("SGLang: --enable-cache-report (+ --enable-metrics for the counter fallback); "
             "vLLM: --enable-prompt-tokens-details")

def fetch_metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None

def metric_sum(body, name, label=None):
    """Sum of every series `name{...} v` in a Prometheus text body (None if absent).
    Requires the char after the name to be `{` or a space so `foo_total` never
    matches `foo_total_created`."""
    if body is None:
        return None
    total = None
    for line in body.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name):]
        if rest[:1] not in ("{", " "):
            continue
        if label and (rest[:1] != "{" or label not in rest.split("}", 1)[0]):
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        total = (total or 0.0) + v
    return total

def detect_reuse_counter():
    body = fetch_metrics()
    for name, label in _REUSE_COUNTERS:
        if metric_sum(body, name, label) is not None:
            REUSE["counter"] = (name, label)
            return
    REUSE["counter"] = None

# ⭐ The API's cached_tokens is ATTENTION-KV ONLY. jb-seo's question (disc #1178)
# is about the KV and MAMBA pools diverging — KV leaf evicted, mamba checkpoint
# stranded — which needs the engine's own pool view. SGLang publishes both pools
# on /metrics (kv_/mamba_ {used,evictable,available}_tokens), split into ACTIVE
# (running requests) vs RESIDENT (radix-cached, evictable). vLLM publishes only
# an active-KV gauge and no mamba pool at all, so on vLLM the columns say so.
#
# ⚠️ Do NOT scrape the batch log line for this. Its `full token usage` / `mamba
# usage` are ACTIVE occupancy (`num_used = capacity - available - evictable`,
# pool_stats_observer.py:220-283): they read 0.00 after every completed turn
# while 62% of the pool was radix-resident, so a divergence heuristic on them
# can never fire. The previous CONTAINER=<name> path did exactly that — and
# `--tail 40` raced a once-per-second /health generate for the newest line.
def pool_view():
    body = fetch_metrics()
    g = lambda n: metric_sum(body, n)
    kv = [g("sglang:kv_used_tokens"), g("sglang:kv_evictable_tokens"), g("sglang:kv_available_tokens")]
    if None not in kv and sum(kv) > 0:
        out = {"engine": "sglang", "kv_total": sum(kv), "kv_active": kv[0], "kv_resident": kv[1]}
        mam = [g("sglang:mamba_used_tokens"), g("sglang:mamba_evictable_tokens"), g("sglang:mamba_available_tokens")]
        if None not in mam and sum(mam) > 0:
            out.update(mamba_total=sum(mam), mamba_active=mam[0], mamba_resident=mam[1])
        return out
    kv_pct = g("vllm:kv_cache_usage_perc")
    if kv_pct is not None:
        return {"engine": "vllm", "kv_active_frac": kv_pct}
    return None

def pool_cols(p):
    """Two fixed-width columns: KV resident%, mamba resident slots/total."""
    if not p or p["engine"] != "sglang":
        return f" {'n/a':>8} {'n/a':>10}"
    kv = f"{100 * p['kv_resident'] / p['kv_total']:.0f}%"
    if "mamba_total" in p:
        mam = f"{int(p['mamba_resident'])}/{int(p['mamba_total'])}"
    else:
        mam = "n/a"
    return f" {kv:>8} {mam:>10}"

# A per-run nonce in every prompt. Without it a second run against the same
# server finds its calibration prompt, its cold reference and its sessions
# already cached: the self-test read "repeat TTFT x1.00" (which, with no counter,
# is diagnosed as prefix caching OFF) and the breadth cold reference came back
# warm, so every re-query classified as cold-ish. Seen on the second live run.
RUN = int(time.time()) % 1000000
FILLER = ("The regional logistics audit " + str(RUN) + " for corridor %d recorded stable "
          "utilisation with no deviation worth escalating during this reporting cycle. ")

def _post(msgs, max_tokens):
    payload = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
               "temperature": 0.7, "top_p": 0.8, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False},
               # An OpenAI-compatible stream omits `usage` unless asked. Without
               # include_usage, prompt_tokens returns 0 — which silently poisons
               # calibration. continuous_usage_stats (honoured by vLLM v0.29.0
               # api_utils.py:297 and SGLang v0.5.19, both measured/verified) puts
               # cumulative completion_tokens on every chunk, which is the only way
               # to count TOKENS rather than chunks under speculative decoding.
               "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    body = json.dumps(payload).encode()
    return urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                  headers={"Content-Type": "application/json"})

def measure(msgs, max_tokens=24):
    """One turn. Returns a dict:
        ttft          first chunk carrying `choices` — NOT the first chunk with content.
                      On a reasoning model the content field stays empty through the
                      reasoning phase, and timing on it charges reasoning to prefill
                      (the #1096 retraction). None if no choices chunk ever arrived.
        dtps          decode tokens/s over the window from the first generated token to
                      the last usage chunk, counted from usage.completion_tokens — NOT
                      from chunks: under DFlash one chunk carried 8 tokens (measured),
                      so chunk-counting read ~5x low. None when unmeasurable (#1267).
        dtps_approx   True when the engine sent no per-chunk usage and the fallback
                      (completion_tokens-1)/(wall-ttft) was used instead.
        ptok, ctok    usage.prompt_tokens / completion_tokens (0 if usage missing)
        cached_usage  prompt_tokens_details.cached_tokens, or None when ABSENT
        cached_metrics  delta of the engine's prefix-hit counter across this
                      request, or None when no counter is published
        text          assistant content
    """
    m_before = fetch_metrics() if REUSE["counter"] else None
    t0 = time.time(); ttft = None; text = ""; usage = None
    first = None; last = None   # (t, cumulative completion_tokens) at the first token / last usage chunk
    with urllib.request.urlopen(_post(msgs, max_tokens), timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try: d = json.loads(line[6:])
            except Exception: continue
            now = time.time()
            u = d.get("usage")
            if u:
                usage = u
                c = u.get("completion_tokens")
                if isinstance(c, int):
                    if c >= 1 and first is None: first = (now, c)
                    last = (now, c)
            ch = (d.get("choices") or [{}])[0]
            if ch:
                if ttft is None: ttft = time.time() - t0
                text += (ch.get("delta") or {}).get("content") or ""
    wall = time.time() - t0
    u = usage or {}
    ctok = u.get("completion_tokens") or 0
    dtps = None; approx = False
    if first and last and last[1] > first[1]:
        window = last[0] - first[0]; dtok = last[1] - first[1]
    else:   # no per-chunk usage from this engine: assume the first chunk was one token
        window = wall - (ttft or 0); dtok = ctok - 1; approx = True
    # Report n/a rather than 0.0 for an unmeasurable window (#1267): a zero here
    # is indistinguishable from a genuine silent-empty turn.
    if window > 0.1 and dtok > 0:
        dtps = dtok / window
    ptd = u.get("prompt_tokens_details")
    cached_usage = ptd.get("cached_tokens") if isinstance(ptd, dict) else None
    cached_metrics = None
    if REUSE["counter"] and m_before is not None:
        a = metric_sum(m_before, *REUSE["counter"]); b = metric_sum(fetch_metrics(), *REUSE["counter"])
        if a is not None and b is not None:
            cached_metrics = int(b - a)
    return {"ttft": ttft, "dtps": dtps, "dtps_approx": approx, "ptok": u.get("prompt_tokens") or 0,
            "ctok": ctok, "cached_usage": cached_usage, "cached_metrics": cached_metrics,
            "text": text.strip()}

def cached_of(m):
    """(cached_tokens or None, source). None means UNOBSERVABLE, never 0."""
    if m["cached_usage"] is not None:
        return m["cached_usage"], "usage"
    if REUSE["source"] == "usage":
        # The field is proven live on this server and both engines omit/zero it
        # when nothing was reused (SGLang _details_if_cached; vLLM cached_tokens=0).
        return 0, "usage"
    if REUSE["source"] == "metrics" and m["cached_metrics"] is not None:
        return m["cached_metrics"], "metrics"
    return None, None

def fmt_cached(m):
    c, _ = cached_of(m)
    return "n/a" if c is None else f"{c:,}"

def fmt_dtps(m):
    if m["dtps"] is None: return "n/a"
    return f"{'~' if m['dtps_approx'] else ''}{m['dtps']:.1f}"

def fmt_ttft(m):
    return "n/a" if m["ttft"] is None else f"{m['ttft']:.2f}"

def http_error_text(e):
    if isinstance(e, urllib.error.HTTPError):
        try: body = e.read().decode("utf-8", "replace")[:300]
        except Exception: body = ""
        return f"HTTP {e.code}: {body}"
    return repr(e)

_TOK_PER_REP = None
_CAL = None   # the calibration prompt + its measurement, reused by the self-test

def calibrate():
    """Tokens per FILLER repetition, MEASURED not assumed.

    A words//N guess was 3.2x off on the first attempt here, which turned a
    28K-token session into 90K and thrashed the pool. The engine reports
    prompt_tokens; use it. 240 reps (~5K tokens here) so the self-test repeat is
    long enough to show reuse even on vLLM+MTP, which only reports full 1,600-
    token blocks and drops the last one (cached = (floor(N/1600)-1)*1600)."""
    global _TOK_PER_REP, _CAL
    if _TOK_PER_REP: return _TOK_PER_REP
    reps = 240
    prompt = (FILLER % 0) * reps
    m = measure([{"role": "user", "content": prompt}], max_tokens=1)
    if not m["ptok"]:
        # Falling back to a default here produced 6,000-repetition turns (~150K
        # tokens) and a 133s first turn that read as a slow rig rather than a
        # broken probe. Refuse instead of guessing.
        sys.exit("calibration failed: server reported prompt_tokens=0 — the stream is not "
                 "returning usage. Check stream_options/include_usage on this engine. "
                 "Refusing rather than guessing a token size.")
    _TOK_PER_REP = m["ptok"] / reps
    _CAL = (prompt, m)
    print(f"  calibration: {_TOK_PER_REP:.1f} tokens per repetition "
          f"({m['ptok']:,} tok over {reps} reps, ttft {fmt_ttft(m)}s)")
    return _TOK_PER_REP

def reuse_selftest():
    """Send the calibration prompt a second time and see WHERE the reuse shows up.
    Decides REUSE['source'] for the whole run. Never guesses: if the usage field is
    absent and no counter moved, the cached column is n/a for the run."""
    calibrate()
    prompt, first = _CAL
    rep = measure([{"role": "user", "content": prompt}], max_tokens=1)
    ratio = (rep["ttft"] / first["ttft"]) if (rep["ttft"] and first["ttft"]) else None
    ratio_s = f"x{ratio:.2f}" if ratio is not None else "n/a"
    if rep["cached_usage"] is not None:
        REUSE["source"] = "usage"
    elif rep["cached_metrics"] is not None and rep["cached_metrics"] > 0:
        REUSE["source"] = "metrics"
    usage_s = ("absent" if rep["cached_usage"] is None else f"{rep['cached_usage']:,}")
    ctr_s = ("no counter on /metrics" if not REUSE["counter"] else
             f"{REUSE['counter'][0]} delta {rep['cached_metrics']:,}" if rep["cached_metrics"] is not None
             else f"{REUSE['counter'][0]} unreadable")
    print(f"  reuse self-test (same {first['ptok']:,}-token prompt again): repeat TTFT {ratio_s}; "
          f"usage.prompt_tokens_details.cached_tokens: {usage_s}; {ctr_s}")
    if REUSE["source"]:
        print(f"  cached column source: {REUSE['source']}"
              + (" (engine-wide counter delta — concurrent traffic during a turn is counted too)"
                 if REUSE["source"] == "metrics" else ""))
        if REUSE["source"] == "usage" and rep["cached_usage"] == 0 and ratio is not None and ratio < 0.5:
            print("  ⚠️ the field is present but reads 0 while the repeat TTFT collapsed — the engine "
                  "under-reports reuse; treat the cached column with suspicion")
        return
    if ratio is not None and ratio < 0.5:
        print(f"  ⚠️ reuse HAPPENED (repeat TTFT {ratio_s}) but this server does NOT REPORT it. "
              f"The cached column is n/a for this run — NOT 0. To light it up: {FLAG_HINT}.")
    else:
        print(f"  ⚠️ the engine did not reuse an identical prompt (repeat TTFT {ratio_s}) and reports no "
              f"cached count — prefix caching looks OFF, or unobservable ({FLAG_HINT}). cached column: n/a.")

def chunk(tokens, seed):
    return (FILLER % seed) * max(1, int(tokens / calibrate()))

def run_depth():
    print(f"  DEPTH — one conversation to ~{TARGET_CTX:,} accumulated tokens, "
          f"~{TURN_TOKENS:,} per turn")
    reuse_selftest()
    p = pool_view()
    if p and p["engine"] == "sglang":
        print("  pool view: /metrics — kv_res = % of the KV pool that is radix-RESIDENT (reusable), "
              "mamba_res = resident state slots / pool. Active-use (running-request) occupancy is "
              "~0 between turns by construction and is not what is shown.")
    elif p and p["engine"] == "vllm":
        print("  pool view: n/a — vLLM publishes only an ACTIVE KV gauge (vllm:kv_cache_usage_perc) "
              "and no mamba pool metric, so residency is not observable on this engine.")
    else:
        print("  pool view: n/a — no pool gauges on /metrics (SGLang needs --enable-metrics; "
              "vLLM has no mamba pool metric).")
    if os.environ.get("CONTAINER"):
        print("  note: CONTAINER is no longer used — the batch log line reports ACTIVE occupancy, "
              "not residency (see pool_view in the source).")
    print("  decode_tps: usage.completion_tokens over first-token->last-chunk; a '~' prefix means "
          "the engine sent no per-chunk usage and the first chunk was assumed to be 1 token.")
    print(f"  {'turn':>4} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8} "
          f"{'decode_tps':>11} {'kv_res':>8} {'mamba_res':>10}")
    msgs = []; turn = 0; ptok = 0; prev_ttft = None
    while ptok < TARGET_CTX:
        turn += 1
        # The reply must be long enough to open a decode window: "reply OK" gave
        # 1-2 tokens and the decode column was n/a on every turn, by construction.
        # 48 tokens is ~0.3s even at 170 tok/s (DFlash), above the 0.1s floor.
        msgs.append({"role": "user", "content": chunk(TURN_TOKENS, turn)
                     + f"\n\nTurn {turn}: count from one to sixty in words, comma separated."})
        try:
            m = measure(msgs, max_tokens=48)
        except Exception as e:
            print(f"  turn {turn}: ERROR {http_error_text(e)}"); return
        ptok = m["ptok"]
        msgs.append({"role": "assistant", "content": m["text"] or "OK"})
        flags = ""
        if m["ttft"] is None:
            flags += "   <- NO CHOICES CHUNK (empty stream)"
        elif prev_ttft and prev_ttft > 0 and m["ttft"] > prev_ttft * 2.0:
            flags += "   <- TTFT MORE THAN DOUBLED"
        if m["ctok"] == 0:
            flags += "   <- EMPTY TURN (0 completion tokens)"
        elif not m["text"]:
            flags += f"   <- {m['ctok']} tokens but no content (reasoning-only or parser ate it)"
        print(f"  {turn:>4} {ptok:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8} "
              f"{fmt_dtps(m):>11}{pool_cols(pool_view())}{flags}", flush=True)
        prev_ttft = m["ttft"]
        if not ptok:
            print("  stopping: prompt_tokens came back 0 — usage vanished mid-run"); break
        if turn > 200:
            print("  stopping: 200 turns without reaching target"); break

def verdict(m, cold_ttft):
    """Breadth verdict. cold_ish = TTFT within 30% of the cold reference. The
    cached fraction matters: a small cached count with a cold-cost TTFT is a
    PARTIAL eviction, not a stranding — only a mostly-cached prefix that still
    costs a cold prefill is the shape jb-seo describes."""
    if m["ttft"] is None:
        return "NO RESPONSE (no choices chunk)"
    cold_ish = m["ttft"] > cold_ttft * 0.7
    cached, _ = cached_of(m)
    if cached is None:
        return ("cold-like TTFT" if cold_ish else "warm-like TTFT") + " — reuse count unobservable (see self-test)"
    if cached == 0:
        return "CLEAN eviction" if cold_ish else "fast despite cached=0 (?)"
    frac = cached / m["ptok"] if m["ptok"] else 0.0
    if cold_ish:
        return (f"STRANDED — {frac:.0%} reported cached but cold-cost" if frac >= 0.5
                else f"PARTIAL eviction — {frac:.0%} cached, cold-cost expected")
    return f"HEALTHY reuse ({frac:.0%} cached)"

def run_breadth():
    reuse_selftest()
    p = pool_view()
    pool = KV_POOL or (int(p["kv_total"]) if p and p.get("kv_total") else 0)
    total = SESSIONS * SESSION_CTX
    print(f"  BREADTH — {SESSIONS} sessions x ~{SESSION_CTX:,} tok = ~{total:,}")
    if pool:
        print(f"  KV pool {pool:,} tok ({'KV_POOL' if KV_POOL else 'from /metrics'}): "
              f"planned {total / pool:.2f}x pool")
        if total <= pool:
            print("  ⚠️ planned tokens do NOT exceed the pool — no KV eviction pressure, so the verdicts "
                  "below are NOT an eviction test. Raise SESSIONS or SESSION_CTX.")
    else:
        print("  ⚠️ KV pool unknown (KV_POOL unset and no pool gauge on /metrics) — cannot confirm "
              "eviction pressure; verdicts below are only meaningful if you know the pool was exceeded.")
    if p and p.get("mamba_total"):
        print(f"  mamba pool: {int(p['mamba_total'])} state slots — one distinct prefix per slot, so "
              f"breadth can exhaust it long before KV fills (compose header arithmetic).")
    # Seed SESSIONS+1: never a session seed, and the same digit class as them —
    # "9999" tokenised ~3 tokens/rep longer than "1" and made the reference 14%
    # longer than the sessions it is compared against.
    try:
        cold = measure([{"role": "user", "content": chunk(SESSION_CTX, SESSIONS + 1) + "\n\nReply OK."}],
                       max_tokens=8)
    except Exception as e:
        print(f"  cold reference: ERROR {http_error_text(e)}"); return
    if cold["ttft"] is None:
        print("  cold reference: no choices chunk — cannot classify anything; stopping"); return
    print(f"  cold reference: {cold['ptok']:,} tok, ttft {cold['ttft']:.2f}s (measured on an "
          f"unpressured pool; later cold prefills also pay eviction, so >= this)")
    print(f"  {'open':>8} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8} {'kv_res':>8} {'mamba_res':>10}")
    sess = []
    for i in range(1, SESSIONS + 1):
        m0 = [{"role": "user", "content": chunk(SESSION_CTX, i) + f"\n\nSession {i}: reply OK."}]
        try:
            m = measure(m0, max_tokens=8)
        except Exception as e:
            print(f"  session {i}: ERROR {http_error_text(e)}"); return
        m0.append({"role": "assistant", "content": m["text"] or "OK"})
        sess.append(m0)
        print(f"  {i:>8} {m['ptok']:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8}{pool_cols(pool_view())}", flush=True)
    print(f"  re-query (order matters: each re-query is itself a prefill that can evict the next)")
    print(f"  {'session':>8} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8}   verdict")
    for idx in sorted({0, 1, len(sess) // 2, len(sess) - 1}):
        s = list(sess[idx]); s.append({"role": "user", "content": "Follow-up: reply OK."})
        try:
            m = measure(s, max_tokens=8)
        except Exception as e:
            print(f"  {idx + 1:>8}: ERROR {http_error_text(e)}"); continue
        print(f"  {idx + 1:>8} {m['ptok']:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8}   {verdict(m, cold['ttft'])}",
              flush=True)

detect_reuse_counter()
run_depth() if MODE == "depth" else run_breadth()
