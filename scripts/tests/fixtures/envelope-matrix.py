"""#1363 matrix guard — every registry slug x every hardware profile.

Asserts three properties of resolve_variant_pin():

  A. TOTAL      it does not raise for a registered slug
  B. ALLOWLIST  every emitted key is accepted by BOTH launchers' case arms
                (a key in one and not the other still hard-fails `exit 2`)
  C. CONSUMED   every emitted key appears as ${KEY:- in that slug's own compose
                -- never emit a variable the compose does not read

A and C are expected-red today and are tracked; B must be GREEN, because an
unallowlisted key is not a silent no-op -- it is `exit 2`, an unlaunchable slug.
"""
import io, os, re, sys, glob
from pathlib import Path

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY as REG   # noqa: E402
from scripts.lib.profiles.compat import load_profiles                        # noqa: E402
from scripts.lib.profiles import launch_compat as lc                         # noqa: E402


def launcher_allowlist(path):
    """Parse the case arms of export_variant_engine_pin() -- never duplicate them."""
    txt = io.open(path, encoding="utf-8").read().split("\n")
    i = next(k for k, l in enumerate(txt) if "export_variant_engine_pin()" in l)
    keys, depth = set(), 0
    for l in txt[i:i + 90]:
        m = re.match(r"\s+([A-Z][A-Z0-9_]*)\)", l)
        if m:
            keys.add(m.group(1))
        if re.match(r"\s+esac", l) and keys:
            break
    return keys


def hardware_specs(profiles):
    """One representative 2-card gpu_spec per hardware profile."""
    out = {}
    for p in sorted(glob.glob("scripts/lib/profiles/hardware/*.yml")):
        import yaml
        d = yaml.safe_load(io.open(p, encoding="utf-8").read()) or {}
        hid = d.get("id") or Path(p).stem
        vram = int(d.get("vram_mib") or d.get("vram_gb", 24) * 1024)
        sm = d.get("sm") or d.get("min_sm") or 8.6
        name = d.get("detect_names", [d.get("display_name", hid)])
        name = name[0] if isinstance(name, list) and name else str(name)
        out[hid] = f"0|{name}|{vram}|{sm};1|{name}|{vram}|{sm}"
    return out


def compose_vars(compose_path, _cache={}):
    if compose_path not in _cache:
        try:
            t = io.open(compose_path, encoding="utf-8").read()
        except OSError:
            t = ""
        _cache[compose_path] = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)[:\-}]", t))
    return _cache[compose_path]


def main():
    profiles = load_profiles()
    allow_switch = launcher_allowlist("scripts/switch.sh")
    allow_launch = launcher_allowlist("scripts/launch.sh")
    allow = allow_switch & allow_launch
    specs = hardware_specs(profiles)

    raises, not_allowed, not_consumed = {}, {}, {}
    for slug, entry in sorted(REG.items()):
        cpath = entry.get("compose_path") or ""
        for hid, spec in specs.items():
            try:
                exports = lc.resolve_variant_pin(profiles, slug, spec)
            except Exception as exc:                       # noqa: BLE001
                raises.setdefault(type(exc).__name__, set()).add(slug)
                continue
            for k in exports:
                if k not in allow:
                    not_allowed.setdefault(k, set()).add(f"{slug}@{hid}")
                if cpath and os.path.exists(cpath) and k not in compose_vars(cpath):
                    not_consumed.setdefault(k, set()).add(f"{slug}@{hid}")

    print(f"  matrix: {len(REG)} slugs x {len(specs)} hardware profiles = {len(REG)*len(specs)} resolutions")
    print(f"  launcher allowlist (intersection of both): {len(allow)} keys")
    if allow_switch != allow_launch:
        print(f"  ⛔ LAUNCHERS DIVERGE: {allow_switch ^ allow_launch}")

    print("\n  A. TOTAL — resolve_variant_pin must not raise")
    for exc, slugs in sorted(raises.items()):
        print(f"     {exc}: {len(slugs)} slugs   (tracked: #1365)")

    print("\n  B. ALLOWLIST — emitted keys accepted by BOTH launchers")
    if not not_allowed:
        print("     ✅ none missing")
    for k, where in sorted(not_allowed.items()):
        print(f"     ⛔ {k}: emitted for {len(where)} (slug,card) — NOT allowlisted => launcher exit 2")

    print("\n  C. CONSUMED — emitted keys appear as ${KEY:- in the slug's compose")
    if not not_consumed:
        print("     ✅ none unconsumed")
    for k, where in sorted(not_consumed.items()):
        ex = sorted(where)[:2]
        print(f"     ⚠️  {k}: {len(where)} (slug,card) do not read it, e.g. {ex}   (tracked: #1365)")

    # ------------------------------------------------------------------
    # D. POST-FLIP SIMULATION. B above passes only because 73 slugs RAISE
    # before they can emit -- the raise masks the landmine. #1365 removes the
    # raise, so model that world now: call the injectors directly, skipping the
    # image pin, and see what WOULD reach the launcher.
    # ------------------------------------------------------------------
    future = {}
    for slug, entry in sorted(REG.items()):
        for hid, spec in specs.items():
            emitted = {}
            for fn in (lc._envelope_env, lc._mem_util_env):
                try:
                    emitted.update(fn(profiles, slug, spec, entry)
                                   if fn is lc._envelope_env else fn(profiles, slug, spec))
                except Exception:                          # noqa: BLE001
                    pass
            for fn in (lc._moe_cache_env, lc._deepgemm_env):
                try:
                    emitted.update(fn(profiles, slug, entry, spec))
                except Exception:                          # noqa: BLE001
                    pass
            for k in emitted:
                if k not in allow:
                    future.setdefault(k, set()).add(f"{slug}@{hid}")

    # E. STATIC: keys emittable BY CONSTRUCTION must be allowlisted even when no
    # row exists to surface them yet. Without this, adding the first sglang
    # envelope row would discover MAX_RUNNING_REQUESTS at a user's launcher.
    for key in sorted(set(lc._ENGINE_TYPE_CONCURRENCY_ENV.values())):
        if key not in allow:
            future.setdefault(key, set()).add("(static: _ENGINE_TYPE_CONCURRENCY_ENV)")

    print("\n  D. POST-FLIP (#1365) — what WOULD reach the launcher once the raise goes")
    if not future:
        print("     ✅ nothing unallowlisted")
    for k, where in sorted(future.items()):
        cards = sorted({w.split("@")[1] for w in where if "@" in w}) or ["(static)"]
        print(f"     ⛔ {k}: {len(where)} (slug,card) on {cards} — would be launcher exit 2")

    # B is the hard gate TODAY; D is the hard gate for #1365 and is asserted now
    # so the landmine cannot be discovered by a user after the flip.
    # Tracked exceptions: a key may stay unallowlisted ONLY with an issue that
    # clears it. This keeps the landmine as a failing-shaped assertion rather
    # than prose, without leaving a red test in the suite.
    TRACKED = {
        # _moe_cache_env has NEVER been booted on a >24 GB card (hardware/rtx-a6000.yml).
        # Allowlisting it without that boot would put an unvalidated reserve live the
        # moment #1365 flips the gate. Decision belongs to #1365, not to this guard.
        "MOE_RESERVE_MB": "#1365",
    }
    untracked = {k: v for k, v in future.items() if k not in TRACKED}
    for k in sorted(set(future) & set(TRACKED)):
        print(f"     (tracked by {TRACKED[k]} — not failing the build)")
    if untracked:
        print("\n  FAIL: #1365 would emit a key no launcher allows, and it is NOT tracked.")
        print("        Allowlist it in BOTH launchers, or add it to TRACKED with the issue")
        print("        that will clear it. An unallowlisted key is `exit 2`, not a no-op.")
        for k in sorted(untracked):
            print(f"          - {k}")
        return 1
    if not_allowed:
        print("\n  FAIL: an emitted key that no launcher allows makes the slug unlaunchable.")
        return 1
    print("\n  PASS: every emitted key is allowlisted in both launchers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
