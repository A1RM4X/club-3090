#!/usr/bin/env bash
# test-offload-matrix-mocked — TIER 1 guards for scripts/offload-matrix.sh.
#
# Drives the REAL sweep end-to-end against a fake llama-server and a fake nvidia-smi,
# so every status-classification and row-integrity path is exercised deterministically
# with no GPU and no model. This is where the sweep's actual logic gets asserted: the
# defects it has already shipped were all "plausible wrong row", never a crash.
#
# HERMETIC BY CONSTRUCTION — both of these are load-bearing, not conveniences:
#   INHERIT=0                  the sweep unconditionally runs `pgrep -x llama-server`
#                              and, if one is up, inherits -m/-ot/-t/-ub/-ct from its
#                              argv. Without this a stray production server silently
#                              rewrites the config under test.
#   ALLOW_CONCURRENT_SERVER=1  the same detection makes the sweep refuse to start
#                              (exit 2) while a server runs. Mocked arms allocate no
#                              VRAM, so there is genuinely nothing to contend for.
#
# Asserts STRUCTURE and STATUS only — never a throughput value. Numbers from a fake
# server mean nothing, and real numbers are not portable across rigs.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWEEP="$ROOT_DIR/scripts/offload-matrix.sh"
REND="$ROOT_DIR/scripts/offload-matrix-render.py"
FIX="$ROOT_DIR/scripts/tests/fixtures/offload-matrix"
PORT="${TEST_PORT:-8137}"
fail() { echo "FAIL: $1" >&2; exit 1; }

for f in "$SWEEP" "$FIX/fake-llama-server" "$FIX/fake-nvidia-smi"; do
  [[ -f "$f" ]] || fail "missing fixture or script: $f"
done

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"
install -m 0755 "$FIX/fake-nvidia-smi"  "$TMP/bin/nvidia-smi"
install -m 0755 "$FIX/fake-llama-server" "$TMP/bin/fake-llama-server"
export PATH="$TMP/bin:$PATH"
: > "$TMP/model.gguf"          # existence is all the sweep checks

NCOL=$(python3 - "$SWEEP" <<'PY'
import re,sys
src=open(sys.argv[1],encoding="utf-8").read()
print(len(re.search(r"HDR=\$'(.*?)'",src,re.S).group(1).split("\\t")))
PY
)

# Run one arm of the real sweep under a given fake-server scenario.
run_scenario() {
  # two statements deliberately: `local a="$1" b="$TMP/$a"` expands BOTH words before
  # local assigns anything, so $a is unbound under set -u
  local scen="$1"
  local out="$TMP/$scen"
  mkdir -p "$out"
  FAKE_SCENARIO="$scen" FAKE_NGPU=2 FAKE_SLOTS="742,337" FAKE_TOKENS=12 \
  MODEL="$TMP/model.gguf" LLAMA_SERVER="$TMP/bin/fake-llama-server" \
  LAYERS_TOTAL=48 FIRST_MOE_LAYER=1 \
  OUT_DIR="$out" TSV="$out/r.tsv" PORT="$PORT" HOST=127.0.0.1 \
  PRESET=smoke SWEEP_OFFLOAD=4 SWEEP_N=1 SWEEP_NMAX=0 SWEEP_CTX=4096 \
  SWEEP_CACHE=1024 SWEEP_ADMIT=1/64 SWEEP_SHAPE=copy \
  REPS=1 ROUNDS=2 GEN=8 BOOT_TIMEOUT=30 \
  INHERIT=0 ALLOW_CONCURRENT_SERVER=1 \
    bash "$SWEEP" > "$out/sweep.log" 2>&1 || true
  [[ -f "$out/r.tsv" ]] || { sed -n '1,25p' "$out/sweep.log" >&2; fail "$scen: no TSV produced"; }
  echo "$out/r.tsv"
}

status_of() { awk -F'\t' 'NR>1{print $NF}' "$1" | head -1; }

# Column count is the invariant that caught the original 32-for-33 printf defect and
# the short early-exit rows. Checked on EVERY scenario, including the failure paths --
# those are exactly the rows that used to come out ragged.
assert_shape() {
  local tsv="$1" scen="$2"
  local widths; widths=$(awk -F'\t' '{print NF}' "$tsv" | sort -u | paste -sd,)
  [[ "$widths" == "$NCOL" ]] || fail "$scen: ragged TSV — widths [$widths], header is $NCOL"
  [[ "$(wc -l < "$tsv")" -ge 2 ]] || fail "$scen: header only, no data row"
}

echo "  running mocked arms (no GPU, no model) ..."

# 1. healthy -> OK, with a real pool sum and non-zero throughput
tsv=$(run_scenario healthy); assert_shape "$tsv" healthy
[[ "$(status_of "$tsv")" == "OK" ]] || fail "healthy should be OK, got '$(status_of "$tsv")'"
pool=$(python3 - "$tsv" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1],encoding="utf-8"),delimiter="\t"))
print(r[0].get("pool_slots","-"))
PY
)
# 742 + 337 = 1079: pool_slots must SUM both pool-init lines, not tail to the last one
[[ "$pool" == "1079" ]] || fail "pool_slots should sum all pool lines (742+337=1079), got '$pool'"
echo "  ✓ healthy -> OK, pool_slots sums every pool line (1079)"

# 2. a decode that declined the cache invalidates the arm
tsv=$(run_scenario bypass); assert_shape "$tsv" bypass
[[ "$(status_of "$tsv")" == "INVALID_BYPASS" ]] || fail "bypass should be INVALID_BYPASS, got '$(status_of "$tsv")'"
echo "  ✓ bypass -> INVALID_BYPASS"

# 3. cache requested, never allocated on ANY device
tsv=$(run_scenario no_budget_all); assert_shape "$tsv" no_budget_all
[[ "$(status_of "$tsv")" == "CACHE_DISABLED" ]] || fail "no_budget_all should be CACHE_DISABLED, got '$(status_of "$tsv")'"
echo "  ✓ no_budget_all -> CACHE_DISABLED"

# 4. ⚠️ THE HOLE (issue #824). '[moe-cache] enabled:' still prints while ONE device got
#    no pool, so a presence-check scores the arm healthy and it silently measures a
#    half-cached path. Observed live 2026-07-30: CUDA0 no budget, CUDA1 a 69-slot pool.
#    Correct detection compares pool-line count against device count.
tsv=$(run_scenario partial_budget); assert_shape "$tsv" partial_budget
st=$(status_of "$tsv")
[[ "$st" == "CACHE_DISABLED" ]] || fail \
  "partial_budget must be CACHE_DISABLED, got '$st' — the detector is keying on the presence of '[moe-cache] enabled:' instead of comparing pool-line count to device count (issue #824)"
echo "  ✓ partial_budget -> CACHE_DISABLED (per-device budget failure is caught)"

# 5. zero tokens must never read OK — this shipped as agg=0.00/OK once
tsv=$(run_scenario no_tokens); assert_shape "$tsv" no_tokens
[[ "$(status_of "$tsv")" == "NO_TOKENS" ]] || fail "no_tokens should be NO_TOKENS, got '$(status_of "$tsv")'"
echo "  ✓ no_tokens -> NO_TOKENS"

# 6. HTTP failures surface as REQ_ERRORS (and outrank the cache states)
tsv=$(run_scenario req_errors); assert_shape "$tsv" req_errors
[[ "$(status_of "$tsv")" =~ ^(REQ_ERRORS|NO_TOKENS)$ ]] || fail "req_errors should be REQ_ERRORS/NO_TOKENS, got '$(status_of "$tsv")'"
echo "  ✓ req_errors -> $(status_of "$tsv")"

# 7. a server that dies before listening still writes a FULL-WIDTH row. Early-exit rows
#    used to omit `shape` and land 4 columns short, and a ctx ladder that OOMs produces
#    these routinely — so every one of them was silently shifted.
tsv=$(run_scenario boot_fail); assert_shape "$tsv" boot_fail
[[ "$(status_of "$tsv")" == "BOOT_FAIL" ]] || fail "boot_fail should be BOOT_FAIL, got '$(status_of "$tsv")'"
echo "  ✓ boot_fail -> BOOT_FAIL, row still full width"

# 8. status=OK implies real throughput. Checked across every row produced above.
python3 - "$TMP" <<'PY' || exit 1
import csv,glob,sys,os
bad=[]
for p in glob.glob(os.path.join(sys.argv[1],"*","r.tsv")):
    for r in csv.DictReader(open(p,encoding="utf-8"),delimiter="\t"):
        if r["status"]=="OK":
            try: agg=float(r["agg"])
            except Exception: agg=0.0
            if agg<=0: bad.append((os.path.basename(os.path.dirname(p)),r["agg"]))
        if r["status"]!="OK" and r.get("bypass","0") not in ("-","0"):
            pass  # bypass>0 with a non-OK status is correct
        if r.get("bypass","0") not in ("-","0") and r["status"]=="OK":
            bad.append((os.path.basename(os.path.dirname(p)),"bypass>0 but OK"))
if bad: sys.exit(f"FAIL: status=OK with no throughput / bypass: {bad}")
print("  ✓ invariant: status=OK implies agg>0 and bypass==0")
PY

# 9. the renderer must survive every view on real produced rows — one view once died
#    with KeyError and another rendered '-' forever.
cat "$TMP"/healthy/r.tsv > "$TMP/all.tsv"
for s in bypass no_budget_all partial_budget no_tokens boot_fail; do
  tail -n +2 "$TMP/$s/r.tsv" >> "$TMP/all.tsv"
done
for view in --perf --bw --cache --all; do
  python3 "$REND" "$TMP/all.tsv" "$view" > "$TMP/render$view.txt" 2>"$TMP/render$view.err" \
    || { sed -n '1,15p' "$TMP/render$view.err" >&2; fail "renderer crashed on $view"; }
  [[ -s "$TMP/render$view.txt" ]] || fail "renderer produced nothing for $view"
done
python3 "$REND" "$TMP/all.tsv" --perf --md > "$TMP/render.md" || fail "renderer crashed on --md"
grep -q '^|' "$TMP/render.md" || fail "--md produced no markdown table"
echo "  ✓ renderer handles every view (+ --md) over mixed-status rows"

# 10. non-OK arms must be called out, so a reader cannot fold a broken arm into a conclusion
grep -qE "NOT OK|⚠" "$TMP/render--perf.txt" || fail "renderer must flag non-OK arms in its summary"
echo "  ✓ renderer flags non-OK arms"

echo "test-offload-matrix-mocked: ok"
