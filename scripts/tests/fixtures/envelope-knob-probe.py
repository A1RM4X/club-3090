import sys, os
sys.path.insert(0, ".")
from scripts.lib.profiles import launch_compat as lc
from scripts.lib.profiles.compat import load_profiles

profiles = load_profiles()
SPEC = "0|RTX 5090|32607|12.0;1|RTX 5090|32607|12.0"
ok = True

def check(label, cond):
    global ok
    if not cond:
        print("FAIL: " + label, file=sys.stderr); ok = False

# POSITIVE CONTROL -- the bug. Before the engine map this emitted MAX_NUM_SEQS.
# Keyed by EngineProfile.type, so a SECOND vllm id must behave identically
# without any Python edit -- that is the point of typing over id.
out = lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "sglang-stable"})
check("sglang emits its OWN spelling", out.get("MAX_RUNNING_REQUESTS") == "4")
check("sglang emits NO vLLM key", "MAX_NUM_SEQS" not in out)

# NEGATIVE CONTROL -- vLLM unchanged. Breaking the working engine beats the bug.
out = lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "vllm-stable"})
check("vllm still emits MAX_NUM_SEQS", out.get("MAX_NUM_SEQS") == "4")
check("vllm emits NO sglang key", "MAX_RUNNING_REQUESTS" not in out)

# a DIFFERENT vllm engine id must resolve identically (type-keyed, not id-keyed)
out = lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "vllm-lmcache"})
check("a second vllm id resolves the same (type-keyed)", out.get("MAX_NUM_SEQS") == "4")

# llama.cpp has no concurrency cap in its composes -> no injection
out = lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "llama-cpp-local"})
check("llama.cpp family injects nothing", out == {})

# An UNMAPPED engine injects nothing rather than a guessed key.
check("unmapped engine injects nothing",
      lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "exllamav3"}) == {})
check("missing entry injects nothing",
      lc._envelope_env(profiles, "fixture/slug", SPEC, None) == {})

# The per-engine user pin wins, and only for its own key.
os.environ["MAX_RUNNING_REQUESTS"] = "9"
check("explicit sglang pin suppresses injection",
      lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "sglang-stable"}) == {})
check("an sglang pin does NOT suppress vllm injection",
      lc._envelope_env(profiles, "fixture/slug", SPEC, {"engine": "vllm-stable"}).get("MAX_NUM_SEQS") == "4")
del os.environ["MAX_RUNNING_REQUESTS"]

sys.exit(0 if ok else 1)
