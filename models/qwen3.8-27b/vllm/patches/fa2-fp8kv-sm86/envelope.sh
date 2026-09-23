#!/usr/bin/env bash
# fa2_envelope — keep vLLM's arguments inside the FA2 fp8-KV kernel's envelope (#1358).
#
# Sourced by every compose that mounts this patch, right before `exec vllm serve`:
#
#   source /etc/club3090/fa2/envelope.sh
#   fa2_envelope "$@" || exit 1
#   set -- "${FA2_ARGS[@]}"
#
# WHY. The plugin's split-KV kernel refuses a batch whose longest query exceeds
# 2048 tokens (`fp8_attn.cu`: max_q <= 2048 && max_k <= 262144, "Sequence bounds
# exceed this port's supported envelope"), and the failure is fatal to the engine.
# The plugin avoids that kernel for prefill only when the batch holds exactly ONE
# request (backend.py: bounded-prefill route needs num_reqs == 1). With two or more
# requests in a step, a prefill chunk goes to the capped kernel, and vLLM sizes that
# chunk by --long-prefill-token-threshold (or the whole --max-num-batched-tokens
# budget when the threshold is 0). So with fp8 KV and --max-num-seqs > 1, a chunk
# over 2048 kills the engine. Nothing in this depends on TP.
#
# WHAT. Only when the KV cache is fp8 (the plugin hands bf16 to stock FlashAttention,
# where none of this applies):
#   - max-num-seqs > 1 and a per-request chunk that can exceed 2048
#       -> --long-prefill-token-threshold 2048 (the batch budget is left alone, so
#          several requests can still share a step);
#   - --max-model-len > 262144 -> refuse to start (the kernel's other bound).
# Shipped single-stream defaults pass through unchanged.
#
# Output: the adjusted argument list in the array FA2_ARGS. Returns 1 on refusal.

FA2_MAX_Q=2048
FA2_MAX_K=262144

fa2_envelope() {
  FA2_ARGS=()
  local kv="" seqs="" budget="" threshold="" model_len=""
  local -a args=("$@")
  local i n=${#args[@]} a
  for ((i = 0; i < n; i++)); do
    a="${args[i]}"
    case "$a" in
      --kv-cache-dtype=*) kv="${a#*=}" ;;
      --kv-cache-dtype) kv="${args[i+1]:-}" ;;
      --max-num-seqs=*) seqs="${a#*=}" ;;
      --max-num-seqs) seqs="${args[i+1]:-}" ;;
      --max-num-batched-tokens=*) budget="${a#*=}" ;;
      --max-num-batched-tokens) budget="${args[i+1]:-}" ;;
      --long-prefill-token-threshold=*) threshold="${a#*=}" ;;
      --long-prefill-token-threshold) threshold="${args[i+1]:-}" ;;
      --max-model-len=*) model_len="${a#*=}" ;;
      --max-model-len) model_len="${args[i+1]:-}" ;;
    esac
  done

  FA2_ARGS=("${args[@]}")
  case "$kv" in
    fp8|fp8_e4m3) ;;
    *) return 0 ;;   # bf16 / auto: stock FlashAttention, no envelope
  esac

  if [[ "$model_len" =~ ^[0-9]+$ ]] && (( model_len > FA2_MAX_K )); then
    echo "[fa2] --max-model-len ${model_len} exceeds the fp8-KV kernel's max_k ${FA2_MAX_K};" \
      "it would fail at the first request that long (#1358). Lower MAX_MODEL_LEN or use bf16 KV." >&2
    return 1
  fi

  # One sequence: every prefill takes the bounded route. vLLM's own default is far
  # above 1, so a missing flag counts as "more than one".
  if [[ "$seqs" =~ ^[0-9]+$ ]] && (( seqs <= 1 )); then
    return 0
  fi

  # The largest chunk one request can put in a step: the threshold when set (> 0),
  # never more than the step budget.
  local chunk=""
  if [[ "$threshold" =~ ^[0-9]+$ ]] && (( threshold > 0 )); then
    chunk="$threshold"
  fi
  if [[ "$budget" =~ ^[0-9]+$ ]] && (( budget > 0 )); then
    if [[ -z "$chunk" ]] || (( budget < chunk )); then
      chunk="$budget"
    fi
  fi
  if [[ -n "$chunk" ]] && (( chunk <= FA2_MAX_Q )); then
    return 0
  fi

  local -a out=()
  local replaced=0
  for ((i = 0; i < n; i++)); do
    a="${args[i]}"
    case "$a" in
      --long-prefill-token-threshold=*)
        out+=("--long-prefill-token-threshold=${FA2_MAX_Q}"); replaced=1 ;;
      --long-prefill-token-threshold)
        out+=("$a" "${FA2_MAX_Q}"); replaced=1; ((i++)) ;;
      *) out+=("$a") ;;
    esac
  done
  (( replaced )) || out+=(--long-prefill-token-threshold "${FA2_MAX_Q}")
  FA2_ARGS=("${out[@]}")
  echo "[fa2] fp8 KV with max-num-seqs=${seqs:-<vLLM default>}: per-request prefill chunk" \
    "${chunk:-<unbounded>} -> ${FA2_MAX_Q} (--long-prefill-token-threshold), so a batch of" \
    "several requests stays inside the kernel's max_q <= ${FA2_MAX_Q} (#1358)" >&2
  return 0
}
