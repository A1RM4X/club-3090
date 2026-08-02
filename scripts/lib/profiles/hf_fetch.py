"""Verify-in-place for metadata-less local files + announce WHY (club-3090 #812).

A file placed by hand has no ``.cache/huggingface/download/<f>.metadata`` beside
it, so the downloader cannot tell "complete" from "half-copied 40 GB" and
re-pulls it — silently, for hours (Discord report: setup/pull re-downloading a
model that was already at the target path). The instinct is right; the silence
is the bug. So:

* **The announcement is unconditional.** Nothing here starts a transfer without
  first printing one honest line per present file saying what was found and
  what will happen to it.
* **Adoption requires a real hash.** ``VERIFY_IN_PLACE=1`` opts into hashing
  (it costs a full read of a multi-GB file); on a sha256 match the file is
  adopted and the HF metadata stub written, so ``hf download`` skips it
  forever after. Size agreement alone is NEVER called "verified" — a wrapper
  on this rig once printed "DONE (hash-verified)" over silent corruption at the
  correct size, and that is exactly the failure this module refuses to repeat.

Wording contract (asserted by ``test-pull-verify-in-place.sh``): the token
``verified`` appears in an announcement **only** when a sha256 was actually
computed and compared. A size-only observation says "size matches" and nothing
stronger.

The hub-metadata plumbing here (repo tree API, the non-redirect resolve HEAD,
the sha256 helpers) is deliberately generic — the download resilience ladder
(#804) is built on the same three primitives.

Stdlib only — this runs under the bare ``python3`` that ``scripts/pull.sh``
uses, where ``huggingface_hub`` is NOT importable (it exists on this rig only as
a ``uv tool`` exposing the ``hf`` CLI).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_HF = "https://huggingface.co"
_NET_TIMEOUT = 60
_SHA_CHUNK = 8 * 1024 * 1024


# ---------------------------------------------------------------------------
# announcement plumbing
# ---------------------------------------------------------------------------
def default_log(msg: str) -> None:
    """Announcements go to stderr so they interleave with `hf`'s own progress
    bars and never pollute a `--json` stdout contract."""
    print(msg, file=sys.stderr, flush=True)


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown size"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} TB"  # pragma: no cover - unreachable


def short_sha(s: Optional[str]) -> str:
    return f"{s[:12]}…" if s else "?"


def verify_in_place_enabled(env: Optional[dict] = None) -> bool:
    """Opt-in for hashing an already-present file instead of re-downloading it.
    The *announcement* is unconditional and does not consult this — only the
    decision to spend a full read of a multi-GB file does."""
    e = os.environ if env is None else env
    return (e.get("CLUB3090_VERIFY_IN_PLACE") or e.get("VERIFY_IN_PLACE")
            or "") == "1"


# ---------------------------------------------------------------------------
# HF metadata: repo tree + the non-redirect resolve HEAD
# ---------------------------------------------------------------------------
@dataclass
class FileMeta:
    """What the hub publishes about one file. `sha256` is the canonical LFS
    hash (`x-linked-etag` on the non-redirected resolve hop == the model API's
    `siblings[].lfs.sha256`); None means genuinely unverifiable."""
    name: str
    size: Optional[int] = None
    sha256: Optional[str] = None
    commit: Optional[str] = None
    source: str = ""          # "api" | "resolve-head" | ""


def _token(env: Optional[dict] = None) -> Optional[str]:
    """HF token from the env, else the CLI's token file. nohup/cron/non-login
    shells don't source a profile, so an unexported token silently degrades to
    anonymous -> rate-limited -> looks exactly like a stall (the rig wrapper
    learned this the slow way)."""
    e = os.environ if env is None else env
    tok = e.get("HF_TOKEN") or e.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (
        Path(e.get("HF_HOME", "")) / "token" if e.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if p is None:
            continue
        try:
            t = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if t:
            return t
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do NOT follow the 302 into the CAS bridge.

    Following it is the trap: the post-redirect CAS response carries a
    CAS-*blob* ETag (the xet hash) and NO `x-linked-etag`, so a redirect-
    following HEAD reads as "no hash published" on every Xet-backed repo. The
    sha256 lives on the FIRST hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_head(repo_id: str, filename: str, *, revision: str = "main",
                 token: Optional[str] = None,
                 timeout: int = _NET_TIMEOUT,
                 opener: Optional[Any] = None) -> FileMeta:
    """Non-redirect-following HEAD on the resolve endpoint -> size + sha256.

    Reads `x-linked-etag` (canonical LFS sha256) / `x-linked-size` /
    `x-repo-commit` off the 302 itself. `opener` is injectable so tests never
    touch the network."""
    url = f"{_HF}/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    op = opener or urllib.request.build_opener(_NoRedirect())
    hdrs = None
    try:
        with op.open(req, timeout=timeout) as resp:
            hdrs = resp.headers          # 200 (a small non-LFS file)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            hdrs = exc.headers           # the hop we actually want
        else:
            return FileMeta(name=filename)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return FileMeta(name=filename)
    if hdrs is None:                     # pragma: no cover - defensive
        return FileMeta(name=filename)

    def _h(*names: str) -> Optional[str]:
        for n in names:
            v = hdrs.get(n)
            if v:
                return v.strip().strip('"')
        return None

    sha = _h("x-linked-etag", "X-Linked-Etag")
    size = _h("x-linked-size", "X-Linked-Size") or _h("content-length")
    try:
        size_i = int(size) if size is not None else None
    except ValueError:
        size_i = None
    return FileMeta(
        name=filename, size=size_i,
        sha256=sha.lower() if sha else None,
        commit=_h("x-repo-commit", "X-Repo-Commit"),
        source="resolve-head" if sha else "",
    )


def repo_tree(repo_id: str, *, revision: str = "main",
              token: Optional[str] = None, timeout: int = _NET_TIMEOUT,
              urlopen: Optional[Callable] = None) -> list[dict]:
    """`/api/models/<repo>/tree/<rev>?recursive=1` — path + size + lfs.oid for
    every file. Needed to expand `--include` globs for rung 3 (curl fetches one
    named file at a time) and to price/verify an on-disk file. Returns [] on any
    failure; the caller degrades to "no metadata to verify against" rather than
    inventing one."""
    url = (f"{_HF}/api/models/{repo_id}/tree/{revision}"
           f"?recursive=1&expand=1")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    fn = urlopen or urllib.request.urlopen
    try:
        with fn(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def tree_meta(tree: Iterable[dict]) -> dict[str, FileMeta]:
    """`{path: FileMeta}` from a repo-tree payload. `lfs.oid`/`lfs.sha256` is
    the canonical sha256 for LFS files; a plain (non-LFS) file publishes only a
    size, and stays hash-unverifiable — which the announcement says out loud
    instead of implying otherwise."""
    out: dict[str, FileMeta] = {}
    for e in tree or []:
        if e.get("type") not in (None, "file"):
            continue
        name = e.get("path")
        if not name:
            continue
        lfs = e.get("lfs") if isinstance(e.get("lfs"), dict) else {}
        sha = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size") if lfs.get("size") is not None else e.get("size")
        try:
            size_i = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_i = None
        out[name] = FileMeta(
            name=name, size=size_i,
            sha256=str(sha).lower() if sha else None,
            source="api" if sha else "",
        )
    return out


def api_meta(api: dict) -> dict[str, FileMeta]:
    """`{path: FileMeta}` from the `/api/models/<repo>?blobs=true` payload the
    deriver already fetched and stashed at `der.profile['_hf_api']` — free, and
    redirect-immune. Preferred over a network round-trip when present."""
    out: dict[str, FileMeta] = {}
    for s in (api or {}).get("siblings", []) or []:
        if not isinstance(s, dict):
            continue
        name = s.get("rfilename")
        if not name:
            continue
        lfs = s.get("lfs") if isinstance(s.get("lfs"), dict) else {}
        sha = lfs.get("sha256")
        size = lfs.get("size") if lfs.get("size") is not None else s.get("size")
        try:
            size_i = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_i = None
        out[name] = FileMeta(
            name=name, size=size_i,
            sha256=str(sha).strip().strip('"').lower() if sha else None,
            source="api" if sha else "",
        )
    return out


def expand_includes(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """Resolve `--include` globs against a repo file list, preserving pattern
    order and de-duplicating. Exact filenames are valid globs, so the
    single-file case falls out for free."""
    all_names = list(names)
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for n in all_names:
            if n == pat or fnmatch.fnmatch(n, pat):
                if n not in seen:
                    seen.add(n)
                    out.append(n)
    return out


# ---------------------------------------------------------------------------
# hashing + the HF local-dir metadata stub
# ---------------------------------------------------------------------------
def sha256_file(path: Path, *, chunk: int = _SHA_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def metadata_path(local_dir: Path, filename: str) -> Path:
    """The path `huggingface_hub._local_folder.get_local_download_paths`
    computes, so a stub we write is the same file `hf download` reads."""
    return (Path(local_dir) / ".cache" / "huggingface" / "download"
            / f"{os.path.join(*filename.split('/'))}.metadata")


def read_metadata(local_dir: Path, filename: str) -> Optional[dict]:
    """`{'commit': …, 'etag': …, 'timestamp': float}` or None.

    Mirrors `_local_folder.read_download_metadata`, INCLUDING its staleness
    rule: metadata older than the file's mtime is ignored (the file changed
    after the record was written, so the record no longer describes it)."""
    mp = metadata_path(local_dir, filename)
    try:
        lines = mp.read_text(encoding="utf-8").splitlines()
        commit, etag, ts = lines[0].strip(), lines[1].strip(), float(lines[2])
    except (OSError, IndexError, ValueError):
        return None
    try:
        st = (Path(local_dir) / filename).stat()
    except OSError:
        return None
    if st.st_mtime - 1 > ts:
        return None                      # outdated record — do not trust it
    return {"commit": commit, "etag": etag.strip('"'), "timestamp": ts}


def write_metadata(local_dir: Path, filename: str, *, commit: str,
                   etag: str) -> Path:
    """Write the 3-line stub (`commit_hash` / `etag` / `timestamp`) that makes
    `hf download` treat the file as already-fetched.

    ONLY ever called after a full sha256 match — this is the adoption receipt
    for a byte-verified file, never a shortcut around verification."""
    mp = metadata_path(local_dir, filename)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(f"{commit or ''}\n{etag}\n{time.time()}\n", encoding="utf-8")
    return mp


# ---------------------------------------------------------------------------
# #812 — verify-in-place decisions + the unconditional announcement
# ---------------------------------------------------------------------------
@dataclass
class PlanEntry:
    name: str
    action: str        # "adopt" | "download"
    reason: str        # machine token
    message: str       # the honest human line (announced verbatim)
    local_size: Optional[int] = None
    remote_size: Optional[int] = None
    hashed: bool = False


def plan_local_file(local_dir: Path, name: str, meta: Optional[FileMeta], *,
                    do_hash: bool,
                    hasher: Callable[[Path], str] = sha256_file) -> PlanEntry:
    """Decide — and phrase — what happens to one already-present file.

    Order is deliberate: size first (a cheap, certain reject that needs no
    read), then the downloader's own metadata record, then the opt-in hash.
    ``adopt`` is reachable ONLY through a computed sha256 match or a metadata
    record the downloader itself wrote; a bare size agreement is reported as
    "size matches" and still re-downloads, because size agreement is exactly
    what silent corruption looks like."""
    fp = Path(local_dir) / name
    try:
        lsize = fp.stat().st_size
    except OSError:
        return PlanEntry(name, "download", "absent",
                         f"{name}: not present — downloading")

    rsize = meta.size if meta else None
    rsha = meta.sha256 if meta else None
    h = human_bytes(lsize)

    # 1. size mismatch -> certain reject, names BOTH sizes, no read needed.
    if rsize is not None and lsize != rsize:
        return PlanEntry(
            name, "download", "size-mismatch",
            f"found {name} ({h}, {lsize} bytes) but the repo publishes "
            f"{human_bytes(rsize)} ({rsize} bytes) — truncated or stale, "
            f"re-downloading",
            local_size=lsize, remote_size=rsize)

    # 2. the downloader's own record (what `hf download` itself consults).
    md = read_metadata(local_dir, name)
    if md and md.get("etag") and rsha and md["etag"].lower() == rsha:
        return PlanEntry(
            name, "adopt", "metadata-match",
            f"found {name} ({h}) with an HF download record for sha256 "
            f"{short_sha(rsha)} — already fetched by the downloader, skipping",
            local_size=lsize, remote_size=rsize)

    # 3. nothing published to check against -> say so, and say the cost.
    if rsha is None:
        extra = (f" and the size matches ({h})"
                 if rsize is not None and lsize == rsize else "")
        return PlanEntry(
            name, "download", "no-remote-hash",
            f"found {name} ({h}) but the repo publishes no sha256 for it, so "
            f"it cannot be verified{extra} — re-downloading ({h} over the "
            f"wire)",
            local_size=lsize, remote_size=rsize)

    # 4. verifiable, but hashing is opt-in (a full read of a multi-GB file).
    if not do_hash:
        return PlanEntry(
            name, "download", "verify-not-enabled",
            f"found {name} ({h}) but cannot verify it without hashing "
            f"(no HF metadata beside it) — re-downloading ({h} over the wire). "
            f"Set VERIFY_IN_PLACE=1 (or pull.sh --verify-in-place) to sha256 "
            f"the local copy instead and adopt it on a match",
            local_size=lsize, remote_size=rsize)

    # 5. the real check.
    try:
        actual = hasher(fp)
    except OSError as exc:
        return PlanEntry(
            name, "download", "hash-unreadable",
            f"found {name} ({h}) but it could not be read for hashing "
            f"({exc}) — re-downloading",
            local_size=lsize, remote_size=rsize)
    if actual.lower() == rsha:
        return PlanEntry(
            name, "adopt", "hash-match",
            f"verified in place: {name} ({h}) sha256 {short_sha(rsha)} "
            f"matches the repo — adopting, 0 bytes re-downloaded",
            local_size=lsize, remote_size=rsize, hashed=True)
    return PlanEntry(
        name, "download", "hash-mismatch",
        f"found {name} ({h}) but its sha256 {short_sha(actual)} does NOT "
        f"match the repo's {short_sha(rsha)} — corrupt, re-downloading",
        local_size=lsize, remote_size=rsize, hashed=True)


def plan_and_announce(local_dir: Path, names: Iterable[str],
                      meta: dict[str, FileMeta], *, do_hash: bool,
                      log: Callable[[str], None] = default_log,
                      prefix: str = "[verify]",
                      hasher: Callable[[Path], str] = sha256_file,
                      commit: str = "") -> list[PlanEntry]:
    """Announce, unconditionally, what happens to every target file — then
    write the adoption receipt for each verified one.

    This is #812's rule "no code path starts a download without printing why"
    made mechanical: the caller cannot obtain the plan without the lines being
    emitted."""
    entries: list[PlanEntry] = []
    for name in names:
        e = plan_local_file(local_dir, name, meta.get(name), do_hash=do_hash,
                            hasher=hasher)
        entries.append(e)
        if e.reason == "absent":
            continue                     # summarised by the caller, not spammed
        log(f"{prefix} {e.message}")
        if e.action == "adopt" and e.reason == "hash-match":
            m = meta.get(name)
            try:
                write_metadata(local_dir, name, commit=commit,
                               etag=(m.sha256 if m and m.sha256 else ""))
            except OSError as exc:       # pragma: no cover - defensive
                log(f"{prefix} (could not write the adoption record for "
                    f"{name}: {exc} — it will be re-checked next time)")
    return entries


