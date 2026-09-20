# prism b10709 + 3 upstream PRs — build recipe for `llama-cpp-prism-mtp`

⚠️ These patches are **baked into the engine image at build time**, not mounted at
runtime. The engine profile marks them `image_baked: true` for that reason. They are
vendored here so the image is **reproducible**, and so each one's upstream status can
be tracked to its drop trigger.

| patch | upstream | diff | why |
|---|---|---|---|
| `pr205.diff` | [#205](https://github.com/PrismML-Eng/llama.cpp/pull/205) OPEN | +14/-0 | `--spec-type draft-mtp` cannot START on a Hadamard-folded target without it. The reason this engine exists. |
| `pr216.diff` | [#216](https://github.com/PrismML-Eng/llama.cpp/pull/216) OPEN | +5/-2 | GDN `cols_per_warp=4` was gated to GB10; widening to Ampere+ measured **+8.6% prefill** on sm_86. |
| `pr189.diff` | [#189](https://github.com/PrismML-Eng/llama.cpp/pull/189) OPEN | +6/-0 | Guards a CUDA illegal memory access in short-query MMA FA with quantized V (Q 3-4). Inert at the shipped `SPEC_N=4` (Q=5). |

⛔ **Drop trigger for all three:** when the PR merges AND ships in a prism release
tarball, drop the patch and re-point this engine at the stock image. If all three
drop, delete this engine profile and fold the slug back onto `llama-cpp-prism`.

## Rebuild

```bash
git clone --depth 1 --branch prism-b10709-9a9394a \
  https://github.com/PrismML-Eng/llama.cpp.git
cd llama.cpp && for p in pr205 pr216 pr189; do git apply ../$p.diff; done
```

Then build in a CUDA **12.8** devel image (see the Dockerfile stanza below) and copy
`build/bin/` over `/app/` in the stock prism image.

```dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build build-essential libcurl4-openssl-dev ca-certificates
COPY llama.cpp /src
WORKDIR /src
# ⚠️ libcuda.so.1 is the DRIVER library, absent in any build container, so the link
# of llama-server fails on cuMemMap/cuMemCreate/cuDeviceGet. Use the devel image's
# stub. The linker wants the SONAME, hence the symlink.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cmake -B build -G Ninja \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
      -DLLAMA_CURL=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
 && cmake --build build --target llama-server -j 32
FROM ghcr.io/noonghunna/llamacpp-prism@sha256:919ee810d4764f43ee6b88afa7dc23c67e61579ac59f198a1f29d691232d6a96
COPY --from=build /src/build/bin/ /app/
```

## Three build traps, each of which cost a build

1. **`-DCMAKE_CUDA_ARCHITECTURES` must be explicit.** Bare `-DGGML_CUDA=ON` fails with
   `CUDA_ARCHITECTURES is empty for target "llama-common"` (upstream
   [#182](https://github.com/PrismML-Eng/llama.cpp/issues/182)). `86;89;120` **matches
   the stock release matrix** — verify with `cuobjdump --list-elf`, do not read it off
   release notes. A first build at `86` alone was *narrower than stock* and would not
   have run on a 4090 or 5090.
2. **`-Wl,-rpath-link`, not `-L`.** `-L` resolves explicit `-lfoo`; `libcuda.so.1` is a
   transitive `DT_NEEDED` of `libggml-cuda.so` and is searched via `rpath-link`. `ld`
   says so verbatim and it still took two builds to act on.
3. ⛔ **Never `-rpath`** — that bakes the 66 KB stub into the runtime search path, so the
   container loads the stub instead of the real driver. Verify with
   `readelf -d /app/llama-server` that no `RPATH`/`RUNPATH` is present.

⚠️ The built binary reports `build 1` because the shallow clone has no git history to
count from. The **commit hash** (`9a9394a`) is the reliable identifier.
