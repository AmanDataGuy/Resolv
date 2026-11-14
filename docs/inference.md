# Local inference

How to serve Resolv's models on a small GPU (written against an RTX 3050 6 GB laptop, Windows 11 +
WSL2) and measure them. Two scripts do the work: `inference/serve_local.sh` starts a server and
`inference/serve_bench.py` scores it.

## Setup (WSL2)
1. In PowerShell (admin): `wsl --install -d Ubuntu-24.04`, then reboot. Install the NVIDIA driver
   on Windows only, never inside WSL. `nvidia-smi` inside Ubuntu must list the GPU.
2. Keep models and venvs on the WSL filesystem (`~/`), not `/mnt/c` or `/mnt/d`.
3. Windows reaches WSL servers on `localhost`, so the app can keep running from the Windows venv.
   If vLLM gets OOM-killed, set `[wsl2] memory=12GB` in `%UserProfile%\.wslconfig` and `wsl --shutdown`.
4. Serving env, separate from the app's venv (never added to `requirements.txt`):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv ~/venvs/serve && source ~/venvs/serve/bin/activate
   uv pip install "vllm==<pinned release>" "huggingface_hub[cli]"
   ```
5. llama.cpp, built with CUDA at `$LLAMA_CPP` (default `~/llama.cpp`):
   ```bash
   git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
   cmake -B build -DGGML_CUDA=ON && cmake --build build -j --config Release
   uv pip install -r requirements.txt     # for the GGUF conversion scripts
   ```

Record the driver, CUDA, vLLM and llama.cpp versions next to any result.

## Serving
```bash
bash inference/serve_local.sh vllm               # base Qwen2.5-1.5B + GRPO LoRA "grpo" on :8001
bash inference/serve_local.sh gguf               # once: base GGUF (f16, Q8_0, Q4_K_M) + LoRA GGUF
bash inference/serve_local.sh llamacpp q4        # llama.cpp on :8002 (f16 | q8 | q4), LoRA applied at runtime
bash inference/serve_local.sh agent <hf-model>   # a resolver with tool calling on :8001, served as "agent"
```
The adapter is read from `models/adapters/extractor/latest`. Port 8000 is left for the API. Extra
vLLM flags go in `VLLM_EXTRA` (e.g. `--no-enable-prefix-caching`); if startup OOMs, add
`--enforce-eager` or lower `--gpu-memory-utilization`. Check a server with `curl localhost:8001/v1/models`.

## Measuring the extractor
The extractor backend changed from ADK to json on 2026-11-14; numbers before that date are from the ADK backend.

Every arm is scored on the 92 held-out cases (`evals.extractor.load_cases("test")`). Scorecards go to
`inference/results/extractor/serve_<label>.json`.

**The regex arm is the bar.** A model that does not beat it on both-correct, without a higher
hallucination rate, has not earned a GPU. Run it first; every later run prints a paired McNemar
comparison against it (`--compare <label>` picks another reference).
```bash
python -m inference.serve_bench --backend regex --label regex
python -m inference.serve_bench --model grpo --label vllm-grpo                       # GRPO adapter
python -m inference.serve_bench --model Qwen/Qwen2.5-1.5B-Instruct --label vllm-base # untuned base
python -m inference.serve_bench --base-url http://localhost:8002/v1 --model local --label llamacpp-q4
```
- `--format line` (default) sends the GRPO training prompt to `/v1/completions`, greedy, 40 tokens.
  `--format json` sends the production instruction as chat with the `CustomerClaim` schema enforced;
  invalid JSON counts as an error.
- `-c` sets concurrency, `--limit 5` makes a smoke run.
- The training run recorded 0.707 both-correct for the GRPO adapter; the served figure and the regex
  figure are not yet re-measured in-repo. If a served result is far off, check the prompt
  (raw vs chat template), greedy decoding and max tokens before anything else.
- Hallucination is measured on the 13 null-order held-out cases. That is low power; say so when
  comparing two rates.

## Running the resolver locally
The eval runner treats a self-hosted model as one more litellm model string sent to
`SELFHOSTED_BASE_URL` (default `http://localhost:8001/v1`):
```bash
bash inference/serve_local.sh agent Qwen/Qwen3-4B-AWQ    # a 6 GB GPU fits only a small or 4-bit model
AGENT_MODEL=hosted_vllm/agent python -m evals.runner --n 5 --tasks 40 --k 3 --out sweep_agent.jsonl
```
Change one role per sweep and keep `SIMULATOR_MODEL` and `JUDGE_MODEL` pinned across the comparison.
`unauthorized_rate` must be exactly 0.0; anything else is a harness bug. Until a sweep has run, its
numbers are "not yet measured".
