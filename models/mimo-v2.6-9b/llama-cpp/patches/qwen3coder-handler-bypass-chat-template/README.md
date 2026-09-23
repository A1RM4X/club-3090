# MiMo-V2.6-9B chat template — Qwen3-Coder handler bypass

A **temporary** vendored chat template for `llamacpp/mimo9b-single-vision`. It exists only until the
llama.cpp pin reaches a published image containing ggml-org/llama.cpp#29257; drop it in that bump.

## Provenance

| | sha256 | bytes |
|---|---|---|
| embedded `tokenizer.chat_template` of `bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF` Q8_0 (revision `4371da10c84fb26da3592d4cf312d24aa82b7b65`) | `59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63` | 3916 |
| this file | `83f5ec365ea00579f6160ce7cae73a9b47882502c9c7589392e1faec7c335301` | 3921 |

The **only** change, line 44:

```diff
-        {{- '<tool_call><function=' ~ tool_call.name ~ '>' -}}
+        {{- '<tool_call><func' ~ 'tion=' ~ tool_call.name ~ '>' -}}
```

Both render the identical string. The rendered prompt is byte-for-byte unchanged; this was checked through
`/apply-template` against the stock template on the same build.

## Why

llama.cpp b10920 (our pin) selects its hard-coded **Qwen3-Coder** tool-call handler for any template whose
*source* contains `<tool_call>`, `<function=` and `<parameter=`, regardless of whitespace
(`common/chat.cpp`). That handler's grammar and parser (`common/parsers/qwen3-coder.cpp`) force a newline after
every opening tag and end a string argument **only** on `\n</parameter>\n`. MiMo writes its tool calls without
newlines (`<parameter=command>VALUE</parameter>`), so under that grammar a call can never close, end-of-turn is
never legal, and every tool-calling turn runs to the token cap. The boot log at high verbosity says
`Using specialized template: Qwen3-Coder`.

Splitting the `<function=` literal means the substring test no longer matches, so llama.cpp falls through to its
generic auto-parser, which infers the format from what the template actually renders — newline-free. That is
exactly the routing upstream now applies: #29257 excludes templates that render
`'<tool_call><function=' ~ tool_call.name ~ '>'` from the Qwen3-Coder handler.

## Evidence (llama.cpp b10920, MiMo Q8_0, thinking on)

| | stock template | this template |
|---|---|---|
| CLI-21 first turn | `finish=length`, 3000/3000 tokens, one tool call with 12K-char arguments | `finish=tool_calls`, 116 tokens, two well-formed calls |
| cli-40 CLI-21…CLI-26 (multi-round) | 0/6, `server_error` ~266 s each | 6/6 passed, 5–22 s |
| toolcall-15 TC-06/07/15 resampled 12× each, 4000-token cap | 14/36 ran to the cap | 0/36, max 158 tokens |
| hermesagent-20 HA-01 | 499 s | 4.8 s (a genuine partial verdict) |

## Drop when

The `llama-cpp-mainline` pin for this compose moves to the first published `ggml-org/llama.cpp` image built
from `bfd73a876` or later. In that same bump: remove the `--chat-template-file` flag and the volume from the
compose, delete this directory, and drop the `patches.yml` entry. Verify after the bump with the CLI-21
first-turn probe (it must end `finish=tool_calls`).
