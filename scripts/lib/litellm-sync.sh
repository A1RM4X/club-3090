#!/usr/bin/env bash
#
# litellm-sync.sh — render the LiteLLM config the gateway ACTUALLY serves.
#
#   litellm-sync.sh [--check] [--quiet] [--no-restart]
#
# WHY THIS EXISTS
# ---------------
# `litellm-emit.sh` renders the CATALOG view into the tracked config.yaml: one
# canonical slug per model, port-pinned, registry-derived. That is the right
# thing for a file in git — it describes the catalog, not one rig's live state.
#
# It is not what a gateway should SERVE. A model's route names ONE slug's
# default_port, so the moment you run a sibling slug for that model — say
# `sgl/qwen38-27b-dual-max` on :8145 while the route points at the vLLM slug's
# :8091 — the gateway advertises the model and dials a port with nothing on it.
# And 13 of 22 models have no route at all, so launching them changes nothing.
# Both failures are silent: the model list looks populated either way.
#
# So config.yaml stays the tracked catalog view (litellm-emit and its gate are
# untouched), and this renders config.runtime.yaml — the file the container
# mounts — from endpoints that actually answered.
#
# ⚠️ WHY A SECOND FILE, NOT config.yaml. config.yaml is tracked AND mounted.
# Writing rig state into it would leave every checkout permanently dirty, make
# `git pull` conflict on a file nobody edited, and red `test-litellm-generate`
# (which asserts the checked-in block matches the registry). The runtime file is
# gitignored; the tracked one keeps its meaning.
#
# PRUNE SCOPE mirrors owui-register.sh: only `host.docker.internal:<port>` routes
# on a club-3090 registry default_port are ours to remove. The cloud block
# (DashScope, which backs benchlocal quality runs) and anything on a port we do
# not own pass through untouched — not ours to garbage-collect, and a cloud
# endpoint is not "dead" because a local GPU is idle.
#
# ⚠️ LiteLLM has NO config-reload endpoint on the pinned image (POST
# /config/reload → 404), so a changed file needs a container restart. Done here,
# and ONLY when the content actually changed — otherwise every launch would
# bounce the gateway for nothing.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Repo convention: UTF-8 mode before any python3, so a non-UTF-8 locale cannot
# mangle the unicode in these configs (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"

exec python3 "${ROOT_DIR}/scripts/lib/litellm_sync.py" --root "$ROOT_DIR" "$@"
