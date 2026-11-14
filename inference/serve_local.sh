#!/usr/bin/env bash
# Serve Resolv's models locally on a 6 GB GPU (WSL2 for vLLM), then measure them with inference/.
#
#   bash inference/serve_local.sh vllm               # base Qwen2.5-1.5B + GRPO LoRA "grpo", OpenAI API on :8001
#   bash inference/serve_local.sh agent <hf-model>   # a resolver with tool calling on :8001 (AGENT_MODEL=hosted_vllm/agent)
#   bash inference/serve_local.sh gguf               # build base GGUF (f16, Q8_0, Q4_K_M) + LoRA GGUF
#   bash inference/serve_local.sh llamacpp q4        # llama.cpp server on :8002 (q8 | q4 | f16)
#
# Port 8000 is left for the Resolv API (uvicorn resolv.api.main:app); 8001 is config.SELFHOSTED_BASE_URL's default.
# Extra vLLM flags for one-variable experiments go in VLLM_EXTRA, e.g.
#   VLLM_EXTRA="--no-enable-prefix-caching" bash inference/serve_local.sh agent Qwen/Qwen3-4B-AWQ
# Record every flag used for a result next to it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE="Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER="$ROOT/models/adapters/extractor/latest"
GGUF="$ROOT/models/gguf"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"   # a llama.cpp checkout with built binaries
VLLM_EXTRA="${VLLM_EXTRA:-}"

case "${1:-}" in
  vllm)
    # 1.5B in bf16 is ~3.1 GB; 0.85 of 6 GB leaves ~2 GB of KV cache. Add --enforce-eager if CUDA graph capture OOMs.
    # model=grpo in a request uses the adapter; model=Qwen/Qwen2.5-1.5B-Instruct is the untuned base.
    exec vllm serve "$BASE" \
      --port 8001 \
      --max-model-len 2048 \
      --gpu-memory-utilization 0.85 \
      --enable-lora --max-lora-rank 16 \
      --lora-modules grpo="$ADAPTER" $VLLM_EXTRA
    ;;
  agent)
    # The resolver: tool calls need the auto tool choice + hermes parser for Qwen-family models
    # (check `vllm serve --help` for the pinned version). Served as "agent", so AGENT_MODEL=hosted_vllm/agent.
    MODEL="${2:?usage: $0 agent <hf-model-id>}"
    exec vllm serve "$MODEL" \
      --served-model-name agent \
      --port 8001 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
      --enable-auto-tool-choice --tool-call-parser hermes $VLLM_EXTRA
    ;;
  gguf)
    mkdir -p "$GGUF"
    BASE_DIR="$GGUF/hf-base"
    huggingface-cli download "$BASE" --local-dir "$BASE_DIR"
    python "$LLAMA_CPP/convert_hf_to_gguf.py" "$BASE_DIR" --outtype f16 --outfile "$GGUF/base-f16.gguf"
    python "$LLAMA_CPP/convert_lora_to_gguf.py" "$ADAPTER" --base "$BASE_DIR" --outtype f16 --outfile "$GGUF/grpo-lora-f16.gguf"
    "$LLAMA_CPP/build/bin/llama-quantize" "$GGUF/base-f16.gguf" "$GGUF/base-q8_0.gguf" Q8_0
    "$LLAMA_CPP/build/bin/llama-quantize" "$GGUF/base-f16.gguf" "$GGUF/base-q4_k_m.gguf" Q4_K_M
    ;;
  llamacpp)
    case "${2:-q4}" in
      f16) MODEL="$GGUF/base-f16.gguf" ;;
      q8)  MODEL="$GGUF/base-q8_0.gguf" ;;
      q4)  MODEL="$GGUF/base-q4_k_m.gguf" ;;
      *) echo "precision must be f16, q8 or q4" >&2; exit 1 ;;
    esac
    # LoRA applied at runtime on top of the quantized base. 8 slots x 1024 ctx; drop --lora to serve the untuned base.
    exec "$LLAMA_CPP/build/bin/llama-server" -m "$MODEL" --lora "$GGUF/grpo-lora-f16.gguf" \
      -ngl 99 -c 8192 -np 8 --port 8002
    ;;
  *)
    echo "usage: $0 vllm | agent <hf-model> | gguf | llamacpp [f16|q8|q4]" >&2
    exit 1
    ;;
esac
