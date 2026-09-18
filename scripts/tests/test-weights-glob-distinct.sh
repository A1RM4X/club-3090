#!/usr/bin/env bash
# Gate: two weights variants may share a (subdir, verify_glob) ONLY if they
# resolve to the same artifact.
#
# WHY. `weights_state_for()` decides PRESENT/PARTIAL/ABSENT with
# `base.glob(verify_glob)`. If two DIFFERENT artifacts live in one subdir under
# one glob, each entry matches the other's file and their download states become
# indistinguishable: delete one and the catalog still reports it PRESENT, because
# its sibling satisfies the glob. Nothing errors — the column just lies.
#
# ⚠️ FOUND IN THE WILD 2026-09-18, twice:
#   • qwen3.8-27b prism-ternary-pq2 (7.21 GB) + prism-ternary-mmproj-q8 (0.63 GB)
#     shared subdir AND "*.gguf" — the projector reported PRESENT once deleted,
#     because the language model matched its glob.
#   • qwen3.6-35b-a3b mudler-apex-compact + -quality: two different packs, one
#     glob, mutually masking.
#   • qwen3.6-27b gguf_mmproj_f16 matched the MAIN model file.
# The GLM entries were given nested globs for exactly this reason — 9 of 87
# entries already carry one. This gate stops the next one being authored.
#
# ⚠️⚠️ SHARING IS NOT AUTOMATICALLY A BUG, which is why this keys on the FILE
# SET and not on the glob alone. Several variants are deliberate ALIASES: one
# artifact registered under two names so two engines can each name their own
# (gemma-4-12b beellama-q8kxl + unsloth-q8kxl both ARE
# gemma-4-12b-it-UD-Q8_K_XL.gguf; qwen3.6-27b beellama-q8kxl-mtp + -dflash both
# ARE Qwen3.6-27B-UD-Q8_K_XL.gguf). Their download states SHOULD be identical, a
# unique glob is impossible, and "fixing" them would invent a distinction that
# does not exist on disk. A naive glob-uniqueness gate flags all four.
#
# THE RULE: within a subdir, a variant's glob may not MATCH another variant's
# declared file, unless the two are aliases (identical `files:` set).
#
# ⚠️ An earlier version grouped by IDENTICAL glob string and missed the real
# hazard: globs that OVERLAP. Reverting one entry to "*.gguf" while its
# sibling kept "*Compact.gguf" put them in different groups, so the gate
# passed on exactly the regression it exists to catch. Overlap, not equality.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

python3 - <<'PY'
import io, json, subprocess, sys, collections

# ⚠️ READ THE MODEL YAMLs, NOT `weights.py list --json`. That emitter does not
# carry `files:` — it returns None for every variant — so an alias check built on
# it flags the legitimate pairs and cannot ever pass. The YAML is the only place
# the declared artifact list exists.
import glob as _glob, pathlib, yaml

rows = []
for _dir in ("scripts/lib/profiles/models", "scripts/lib/profiles-local/models.d"):
    for _f in sorted(_glob.glob(f"{_dir}/*.yml")):        # local layer is optional
        try:
            _d = yaml.safe_load(io.open(_f, encoding="utf-8")) or {}
        except Exception as exc:
            print(f"  note: skipping unreadable {_f} ({exc})")
            continue
        for _k, _v in (_d.get("weights") or {}).items():
            if not isinstance(_v, dict):
                continue
            rows.append({
                "model": _d.get("id", pathlib.Path(_f).stem),
                "variant": _k,
                # weights.py resolves the directory as local_subdir or path
                "subdir": _v.get("local_subdir") or _v.get("path"),
                "verify_glob": _v.get("verify_glob"),
                "files": _v.get("files"),
                "hf_repo": _v.get("hf_repo"),
            })

from fnmatch import fnmatch

bysub = collections.defaultdict(list)
for r in rows:
    if r.get("subdir") and r.get("verify_glob"):
        bysub[r["subdir"]].append(r)

def catchall(r):
    """A BYO slot, not a specific artifact: nothing identifies WHICH file it is —
    no declared `files:` and no `hf_repo`. qwen3.6-27b's `gguf` variant is one
    (`size_gb: variable`), and matching anything in the directory is the POINT of
    it, so its glob overlapping a named sibling is by design, not a defect.
    A variant with an hf_repo IS a specific artifact even without `files:` — the
    prism PQ2_0 pack was exactly that, and its `*.gguf` really did mask the
    projector."""
    return not r.get("files") and not r.get("hf_repo")

def sig(r):
    """Alias signature: the declared artifact set. None = 'the whole subdir',
    which can never be an alias of a NAMED file."""
    f = r.get("files")
    return tuple(sorted(f)) if f else None

problems, aliases, pairs = [], 0, 0
for sub, members in sorted(bysub.items()):
    for a in members:
        for b in members:
            if a is b:
                continue
            bf = b.get("files")
            if not bf:
                continue                      # nothing named to be claimed
            if not any(fnmatch(f, a["verify_glob"]) for f in bf):
                continue                      # a's glob does not reach b's file
            if catchall(a):
                continue                      # BYO slot: matching anything is its job
            pairs += 1
            sa, sb = sig(a), sig(b)
            if sa is not None and sa == sb:
                aliases += 1                  # one artifact, two names — fine
                continue
            problems.append((sub, a, b))

if problems:
    print("FAIL: variants share a (subdir, verify_glob) but are DIFFERENT artifacts —")
    print("      their download states are indistinguishable, so a deleted file still")
    print("      reports PRESENT:")
    seen = set()
    for sub, a, b in problems:
        key = (sub, a["variant"], b["variant"])
        if key in seen or (sub, b["variant"], a["variant"]) in seen:
            continue
        seen.add(key)
        print(f"  ⛔ {sub}")
        print(f"       {a['variant']:26} glob={a['verify_glob']!r} files={a.get('files')}")
        print(f"       {b['variant']:26} glob={b['verify_glob']!r} files={b.get('files')}")
        print(f"       → {a['variant']}'s glob MATCHES {b['variant']}'s file")
        print("       Fix: give each variant a glob matching only ITS file")
        print("            (e.g. \"mmproj*.gguf\" / \"*Quality.gguf\"), or — if these")
        print("            really are one artifact under two names — declare the same")
        print("            `files:` list on both so they register as aliases.")
    sys.exit(1)

print(f"  ✓ no variant's glob claims another's artifact "
      f"({aliases} alias pairing(s) allowed; {pairs} overlapping pair(s) checked "
      f"across {len(bysub)} subdir(s))")
PY

echo "test-weights-glob-distinct: ok"
