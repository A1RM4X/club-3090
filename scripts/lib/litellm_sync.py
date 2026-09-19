#!/usr/bin/env python3
"""Render the LiteLLM config the gateway ACTUALLY serves, from what is running.

See scripts/lib/litellm-sync.sh for the full rationale. In short:

  config.yaml          tracked CATALOG view — one canonical slug per model,
                       port-pinned, registry-derived. Right for a file in git.
  config.runtime.yaml  gitignored RUNTIME view — what the container mounts,
                       rendered from endpoints that actually answered.

The catalog view cannot be what a gateway serves: its route for a model names
ONE slug's default_port, so running a sibling slug for that model advertises a
model on a port with nothing behind it, and 13 of 22 models have no route at all.
Both failures are silent — the model list looks populated either way.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import sys
import urllib.request

BEGIN = "  # === BEGIN GENERATED LOCAL BLOCK"
END = "  # === END GENERATED LOCAL BLOCK ==="


def registry_ports(root: str) -> list[int]:
    """Every port the catalog owns. These are the ONLY ports this sync probes,
    and the only ones it may prune — see PRUNE SCOPE in the wrapper."""
    try:
        out = subprocess.run(
            ["bash", os.path.join(root, "scripts/lib/registry-emit.sh"), "--json"],
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        ).stdout
        d = json.loads(out)
    except Exception:
        return []
    return sorted({v["port"] for v in d.get("variants", []) if v.get("port")})


def probe(port: int) -> list[tuple[int, str]]:
    """Ask the server what it serves. Names come from /v1/models, NOT from the
    registry's `served_name`: 72 of 138 variants do not declare one, so a
    registry-derived name would have nothing to emit for half the catalog."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            data = json.load(r).get("data") or []
        return [(port, m["id"]) for m in data if m.get("id")]
    except Exception:
        return []


def live_routes(ports: list[int]) -> list[tuple[int, str]]:
    # TEST SEAM. Probing real sockets makes a gate depend on whatever the rig
    # happens to be serving, which is neither deterministic nor reproducible on
    # a contributor's machine. `C3_LITELLM_FAKE_LIVE="8182=a,8091=b"` substitutes
    # the probe result wholesale. Deliberately env-only and undocumented in
    # --help: it is for tests, not operators.
    fake = os.environ.get("C3_LITELLM_FAKE_LIVE")
    if fake is not None:
        out: list[tuple[int, str]] = []
        for item in fake.split(","):
            if "=" in item:
                p, mid = item.split("=", 1)
                if p.strip().isdigit():
                    out.append((int(p.strip()), mid.strip()))
        return out
    found: list[tuple[int, str]] = []
    with cf.ThreadPoolExecutor(max_workers=32) as ex:
        for res in ex.map(probe, ports):
            found += res
    return found


def render_block(live: list[tuple[int, str]]) -> tuple[str, int]:
    lines = [
        BEGIN.rstrip() + " — RUNTIME VIEW, rendered by scripts/lib/litellm-sync.sh ===",
        "  # Routes for endpoints that answered /v1/models at render time. Names come",
        "  # from each server's own model list, so a slug with no registry served_name",
        "  # still gets a route. Regenerated on every switch.sh launch and teardown.",
        "  # DO NOT EDIT — edit services/litellm/config.yaml (the catalog view) instead.",
    ]
    seen: set[str] = set()
    for port, mid in sorted(live, key=lambda t: (t[1], t[0])):
        if mid in seen:
            lines.append(f"  # ⚠️ '{mid}' is also served on :{port}; keeping the first route only")
            continue
        seen.add(mid)
        lines += [
            f"  - model_name: {mid}",
            "    litellm_params:",
            f"      model: openai/{mid}",
            f"      api_base: http://host.docker.internal:{port}/v1",
            "      api_key: EMPTY",
            "",
        ]
    if not live:
        lines.append("  # (nothing serving right now — no local routes)")
    return "\n".join(lines).rstrip() + "\n" + END + "\n", len(seen)


def prune(text: str, reg_ports: set[str], live_ports: set[str]) -> str:
    """Drop OUR dead routes. A route is ours when its api_base is
    host.docker.internal on a registry-owned port. Everything else — the cloud
    block, anything on a port we do not own — passes through untouched: not ours
    to garbage-collect, and a cloud endpoint is not dead because a GPU is idle."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    for ln in text.split("\n"):
        if ln.startswith("  - model_name:") and cur:
            chunks.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks.append(cur)
    kept: list[str] = []
    for c in chunks:
        body = "\n".join(c)
        m = re.search(r"api_base:\s*http://host\.docker\.internal:(\d+)/", body)
        if m and m.group(1) in reg_ports and m.group(1) not in live_ports:
            continue
        kept.append(body)
    return "\n".join(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    def log(msg: str) -> None:
        if not a.quiet:
            print(f"[litellm-sync] {msg}")

    template = os.path.join(a.root, "services/litellm/config.yaml")
    runtime = os.path.join(a.root, "services/litellm/config.runtime.yaml")
    if not os.path.exists(template):
        print(f"[litellm-sync] no template at {template}", file=sys.stderr)
        return 1

    src = io.open(template, encoding="utf-8").read()
    b, e = src.find(BEGIN), src.find(END)
    if b < 0 or e < 0:
        print("[litellm-sync] template has no BEGIN/END markers", file=sys.stderr)
        return 1

    ports = registry_ports(a.root)
    live = live_routes(ports)
    reg_ports = {str(p) for p in ports}
    live_ports = {str(p) for p, _ in live}

    block, n_routes = render_block(live)
    out = (prune(src[:b], reg_ports, live_ports)
           + block
           + prune(src[e + len(END):], reg_ports, live_ports))

    cur = io.open(runtime, encoding="utf-8").read() if os.path.exists(runtime) else None
    if a.check:
        if cur == out:
            log("runtime config is up to date")
            return 0
        log("runtime config is STALE — run: bash scripts/lib/litellm-sync.sh")
        return 1

    if cur == out:
        log(f"no change ({n_routes} live route(s))")
        return 0

    tmp = runtime + ".tmp"                       # temp + replace: a failed encode
    io.open(tmp, "w", encoding="utf-8").write(out)   # never truncates the original
    os.replace(tmp, runtime)
    log(f"rendered {n_routes} live route(s) -> services/litellm/config.runtime.yaml")

    # LiteLLM has NO config-reload endpoint on the pinned image (POST
    # /config/reload -> 404), so a changed file needs a restart. Only on a real
    # change, or every launch would bounce the gateway for nothing.
    if not a.no_restart:
        try:
            names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                   capture_output=True, text=True, timeout=15).stdout.split()
            if "litellm" in names:
                subprocess.run(["docker", "restart", "litellm"],
                               capture_output=True, timeout=90, check=True)
                log("restarted litellm to pick it up")
        except Exception:
            log("WARN: could not restart litellm — routes apply on its next start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
