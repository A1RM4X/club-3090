#!/usr/bin/env bash
# test-compose-no-rig-paths.sh — no compose may use a MAINTAINER-RIG absolute path
# as an environment DEFAULT. The repo ships to strangers; a default is what runs
# when they set nothing, so a default of `/mnt/models/huggingface` silently points
# their container at a directory only the reference rig has.
#
# Why a guard and not a convention: the convention already existed and was followed
# by most composes (the repo-relative `${MODEL_DIR:-../../../../../../models-cache}`
# is the documented shape), yet three shipped composes had drifted onto rig paths
# and a fourth was added on 2026-08-11 before this guard existed. Conventions that
# only live in reviewers' heads decay; this is the same reasoning as
# test-compose-bind-host.sh.
#
# What is CHECKED: `${VAR:-<rig-path>}` defaults and bare rig paths on the host side
# of a volume/command. What is NOT checked: comments and prose. Documenting the
# reference rig ("measured on /mnt/models/...") is legitimate and stays legal — the
# failure mode this guards is a value the compose would actually USE.
#
# If a compose genuinely needs an operator-supplied absolute path, give it a knob
# with a portable default (repo-relative, or another variable), never a rig path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Absolute prefixes that exist only on the maintainer's rig. Keep this list short
# and specific: it is about paths a stranger cannot have, not about all absolutes
# (/dev, /tmp, /usr and friends are fine and common).
RIG_PREFIXES='/opt/ai|/mnt/models|/home/[a-z][a-z0-9_-]*'

fails=0
checked=0

while IFS= read -r f; do
  checked=$((checked+1))
  # Strip comments so prose about the reference rig stays legal, then look for
  # (a) a rig path used as an env default, or (b) a bare rig path on the host side
  # of a volume entry.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    echo "FAIL: $f uses a maintainer-rig path as a value:" >&2
    echo "        $(printf '%s' "$line" | sed 's/^[[:space:]]*//')" >&2
    echo "        Use a portable default (repo-relative, or another variable)." >&2
    fails=$((fails+1))
  done < <(
    sed 's/#.*$//' "$f" \
      | grep -nE "\\\$\{[A-Za-z_][A-Za-z0-9_]*:-(${RIG_PREFIXES})|^[[:space:]]*-[[:space:]]*\"?(${RIG_PREFIXES})/" \
      || true
  )
done < <(find models -name '*.yml' -path '*/compose/*' | sort)

if [ "$fails" -gt 0 ]; then
  echo "$fails rig-path check(s) failed across $checked composes" >&2
  exit 1
fi
echo "test-compose-no-rig-paths: ok ($checked composes, no rig paths used as values)"
