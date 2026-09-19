#!/usr/bin/env bash
# SYNC the Open WebUI OpenAI connection list with what is actually serving, so a
# catalog model launched via switch.sh auto-appears in the OWUI chat picker AND
# endpoints that are no longer up stop appearing.
#
#   owui-register.sh [<port>] [owui_container]     # register <port>, prune dead
#   owui-register.sh --prune-only [owui_container] # prune dead only (teardown)
#
# ⚠️⚠️ THIS USED TO BE APPEND-ONLY, and that was the bug. It registered the
# launched port and never removed anything, so every slug ever launched with
# --owui stayed in the picker forever. Found 2026-09-18 on the reference rig:
# SEVEN registered connections (8090, 8199, 8010, 8051, 8032, 8038, 8030) and
# EVERY ONE refused the connection — the picker was 100% dead entries while the
# slug the user had just started was absent. A stale connection is not inert:
# OWUI probes it on every model-list refresh, so the picker stalls on timeouts.
#
# PRUNE SCOPE — deliberately narrow. Only `host.docker.internal:<port>` URLs
# whose port is a club-3090 registry default_port are candidates: those are the
# ones WE added. A connection the user created by hand (a cloud provider, a
# sibling box, an app on an unrelated port) is NEVER touched, even if it is down
# — it is not ours to garbage-collect, and it may be temporarily offline.
# Liveness is probed FROM INSIDE the OWUI container, which is the same network
# path OWUI itself uses — a host-side probe can succeed where OWUI would fail.
#
# CONDITIONAL: it's a no-op (exit 0) if Open WebUI isn't running — most catalog
# users drive models from the API directly (aider/opencode/agents) and never run
# OWUI, so this must never block or fail a launch.
#
# OWUI stores connections in its DB (PersistentConfig, not env), so we drive its
# admin config API with a short-lived HS256 token forged from the container's
# secret (/app/backend/.webui_secret_key) + the admin user id. Idempotent: skips
# if the endpoint is already registered. host.docker.internal:<port> is the
# container→host path (works on the bundle's OWUI; on a setup without it, add the
# host IP instead via Admin → Settings → Connections).
set -uo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
PRUNE_ONLY=0
if [[ "${1:-}" == "--prune-only" ]]; then PRUNE_ONLY=1; shift; PORT=""
else PORT="${1:?usage: owui-register.sh <port> [owui_container]  |  --prune-only [owui_container]}"; shift || true
fi
OWUI="${1:-${OWUI_CONTAINER:-open-webui}}"
log(){ echo "[owui-register] $*"; }

# Ports this repo owns — the ONLY ones eligible for pruning (see PRUNE SCOPE).
# Derived from the registry, never hand-listed: a hand-list goes stale the first
# time a slug's default_port changes, and the failure is silent (we would either
# orphan a dead route forever or delete a port we no longer recognise).
# Degrades OPEN: if the registry cannot be read we pass an empty set, which
# prunes NOTHING and leaves this a pure add — never a destructive guess.
_OWNED_PORTS="$(bash "$(dirname "$0")/registry-emit.sh" --json 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: print(""); raise SystemExit
print(",".join(sorted({str(v["port"]) for v in d.get("variants",[]) if v.get("port")})))
' 2>/dev/null || printf "")"

if ! docker ps --format '{{.Names}}' 2>/dev/null | command grep -qx "$OWUI"; then
  log "Open WebUI ('$OWUI') not running — skipping (it's optional; start OWUI + re-run with --owui to wire)."
  exit 0
fi
SECRET="$(docker exec "$OWUI" cat /app/backend/.webui_secret_key 2>/dev/null || true)"
if [[ -z "$SECRET" ]]; then log "couldn't read OWUI secret key — skipping."; exit 0; fi

docker exec -i "$OWUI" python3 - "$SECRET" "$PORT" "$_OWNED_PORTS" <<'PY'
import sys, sqlite3, hmac, hashlib, base64, json, urllib.request, urllib.error
secret = sys.argv[1].encode(); port = sys.argv[2]
owned = {p for p in (sys.argv[3] if len(sys.argv) > 3 else "").split(",") if p}
db = "/app/backend/data/webui.db"
try:
    admins = [r[0] for r in sqlite3.connect(db).execute(
        "select id from user where role='admin' order by created_at limit 1")]
except Exception as e:
    print("[owui-register] cannot read OWUI db (%s) — skipping." % e); sys.exit(0)
if not admins:
    print("[owui-register] no admin user yet — open the UI + create one, then re-run with --owui. Skipping.")
    sys.exit(0)
b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
hdr = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
pl  = b64(json.dumps({"id": admins[0]}).encode())
jwt = (hdr + b"." + pl + b"." + b64(hmac.new(secret, hdr + b"." + pl, hashlib.sha256).digest())).decode()
H = {"Authorization": "Bearer " + jwt, "Content-Type": "application/json"}
def call(path, data=None):
    req = urllib.request.Request("http://localhost:8080" + path, headers=H,
                                 data=(json.dumps(data).encode() if data is not None else None))
    return json.load(urllib.request.urlopen(req, timeout=30))
url = ("http://host.docker.internal:%s/v1" % port) if port else None
try:
    cfg = call("/openai/config")
except Exception as e:
    print("[owui-register] OWUI config API unreachable (%s) — skipping." % e); sys.exit(0)
urls = list(cfg.get("OPENAI_API_BASE_URLS") or [])
keys = list(cfg.get("OPENAI_API_KEYS") or [])

# keys is POSITIONALLY paired with urls — drop the same index from both, or every
# surviving connection silently inherits the wrong API key.
while len(keys) < len(urls): keys.append("sk-noauth")

import re
OURS = re.compile(r"^http://host\.docker\.internal:(\d+)/v1/?$")

def reachable(u):
    """Probe exactly as OWUI would, from in here. Any answer at all (including
    401/404) proves something is listening; only a transport error is dead."""
    try:
        urllib.request.urlopen(u.rstrip("/") + "/models", timeout=2); return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False

kept_u, kept_k, pruned = [], [], []
for u, k in zip(urls, keys):
    m = OURS.match(u.strip())
    if m and m.group(1) in owned and u != url and not reachable(u):
        pruned.append(u); continue
    kept_u.append(u); kept_k.append(k)

added = False
if url and url not in kept_u:
    kept_u.append(url); kept_k.append("sk-noauth"); added = True

if not added and not pruned:
    print("[owui-register] no change (%d connection(s); %s)"
          % (len(kept_u), ("already registered: %s" % url) if url else "nothing dead to prune"))
    sys.exit(0)

new = dict(cfg); new.update({"ENABLE_OPENAI_API": True,
                             "OPENAI_API_BASE_URLS": kept_u, "OPENAI_API_KEYS": kept_k})
try:
    call("/openai/config/update", new)
    for u in pruned:
        print("[owui-register] ✂ pruned dead connection: %s" % u)
    if added:
        served = [m["id"] for m in call("/openai/models").get("data", [])]
        print("[owui-register] ✓ wired %s into Open WebUI; picker now lists: %s"
              % (url, ", ".join(served[-3:]) if served else "(endpoint up)"))
    print("[owui-register] %d connection(s) registered" % len(kept_u))
except Exception as e:
    print("[owui-register] update failed (%s) — add it manually in Admin → Settings → Connections." % e)
PY
