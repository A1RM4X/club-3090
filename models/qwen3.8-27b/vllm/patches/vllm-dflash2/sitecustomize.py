"""club-3090 memprobe sitecustomize — auto-loads on every Python startup in the
container (sitecustomize is imported by site.py at interpreter init).

When MEMPROBE=1, it wraps vllm's model-loader `get_model` so that, right after
the (target) model is built, we install a forward() hook that logs the per-step
peak transient activation memory. That's the real headroom number we need to size
--gpu-memory-utilization.

No-op unless MEMPROBE=1, so it's safe to always ship.
"""
import os

if os.environ.get("MEMPROBE", "0") != "1":
    raise SystemExit  # don't patch anything; sitecustomize can bail quietly


def _install():
    import importlib
    try:
        import vllm.model_executor.model_loader as ML
    except Exception:
        return  # vllm not importable in this process (e.g. the API server); skip

    memprobe = importlib.import_module("memprobe")  # on PYTHONPATH via /etc/club3090/dflash2
    orig_get_model = getattr(ML, "get_model", None)
    if orig_get_model is None or getattr(ML, "_memprobe_patched", False):
        return

    def get_model(*args, **kwargs):
        model = orig_get_model(*args, **kwargs)
        try:
            memprobe.install(model)
        except Exception as e:
            print(f"[memprobe] install failed (non-fatal): {e!r}", flush=True)
        return model

    ML.get_model = get_model
    ML._memprobe_patched = True
    print("[memprobe] sitecustomize: hooked vllm model_loader.get_model", flush=True)


try:
    _install()
except SystemExit:
    pass
except Exception as e:
    # Never let a probe break model load.
    print(f"[memprobe] sitecustomize hook error (non-fatal): {e!r}", flush=True)
