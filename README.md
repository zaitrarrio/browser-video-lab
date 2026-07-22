# Browser Video Lab

Four related experiments in one project:

1. **SD-Turbo WebGPU** — browser-side text encoding, denoising and VAE decoding using ONNX Runtime Web.
2. **LongLive WebGPU** — a streaming, causal browser runtime with persistent KV cache for an externally exported LongLive graph.
3. **MemFlow WebGPU** — prompt-conditioned adaptive retrieval over a bounded bank of historical video memories.
4. **Wan 2.1 + Pruna Smash** — native CUDA compression, persistence and benchmarking for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

The browser runtimes do not download multi-gigabyte weights with the repository. Put exported ONNX models under `public/models`, then copy each `manifest.example.json` to `manifest.json`. Large ONNX files should be hosted with byte-range support in production.

## Browser app

Requirements: Node 20+, Chrome or Edge with WebGPU, and a desktop GPU.

```bash
npm install
npm run dev
```

The SD graph contract is:

- text encoder: `input_ids -> last_hidden_state`
- UNet: `sample, timestep, encoder_hidden_states -> out_sample`
- VAE decoder: `latent_sample -> sample`

The included denoising loop is deliberately scheduler-neutral and useful for integration testing. For production-quality SD-Turbo output, export the scheduler math into the UNet graph or replace the Euler update in `src/runtime/sd-turbo.ts` with the exact scheduler used during export.

## LongLive experimental graph contract

LongLive's official implementation depends on CUDA/Triton and does not currently publish a browser ONNX graph. The runtime here is browser-native, but requires an export/distillation step outside the browser. Its generator accepts:

- `noise`, `prompt_embeds`, `chunk_index`
- zero or more `past.*` KV tensors

It returns a latent `sample` or `latents` plus matching `present.*` tensors. The runtime renames `present.*` to `past.*` for the next causal chunk. Export a fixed-resolution, fixed-window graph first; dynamic KV shapes are not consistently supported across WebGPU implementations.

This is an honest integration boundary: the browser runner is implemented, but the upstream LongLive CUDA checkpoint cannot simply be renamed to ONNX. LongLive 2.0's NVFP4 kernels are NVIDIA-native and are not WebGPU operators.

## MemFlow adaptive-memory contract

This implementation targets Kling's streaming-video MemFlow, not the unrelated CVPR 2024 optical-flow project with the same name. It adds a bounded browser-side memory bank to the causal generator:

1. Mean-pool the current prompt embedding into a retrieval query.
2. Rank historical memory keys by cosine similarity.
3. Feed only the top-K memory values into the next generation chunk.
4. Store the generator's new `memory_key` and `memory_value` outputs.
5. Evict the oldest entry when the configured capacity is reached.

The generator graph accepts `noise`, `prompt_embeds`, `chunk_index`, `memory_values`, and `memory_mask`. It returns `latents` (or `sample`), `memory_key`, and `memory_value`. Export an empty-memory sentinel path so that a zero `memory_mask` is valid on the first chunk. The example manifest defaults to 16 stored entries and top-4 retrieval.

The official MemFlow release is tested on 80 GB NVIDIA GPUs and uses native PyTorch/CUDA dependencies. The browser runtime is complete, but producing its ONNX graph requires distillation/export of a fixed-shape generator; the original checkpoint does not execute directly in ONNX Runtime Web.

## Smash Wan 2.1

Use a Linux CUDA system with enough VRAM. Pruna authentication/package access may be required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements-wan.txt
python python/smash_wan21.py --smoke-test
python python/benchmark_wan21.py artifacts/wan21-t2v-1.3b-smashed
```

Pruna API releases have used both `token=` and `api_key=`; the script supports both call signatures. The default compiler follows Pruna's Wan tutorial. Add an available kernel explicitly, for example `--kernel flash_attn3`, only when the installed Pruna build and GPU support it.

## Distill LongLive for the browser

The production recipe targets an approximately 390M-parameter causal student at four latent frames and 256×384-equivalent latent resolution. It combines output/noise matching, temporal-difference matching, and width-independent hidden-relation matching. Training uses precomputed `.pt` shards in `data/longlive-latents`, each containing `latents` and `prompt_embeds` tensors.

LongLive's upstream repository officially uses an AR teacher-forcing stage followed by DMD distillation. This project adds cross-architecture teacher–student compression on top: the 1.3B teacher remains frozen while the smaller browser architecture learns its denoising response and temporal structure. Implement the small adapter described in `python/longlive_distill/ADAPTER.md` against the exact upstream revision being used.

```bash
task setup:python   # base env (numpy/PyYAML/onnx) — installs on any platform
task setup:torch    # installs PyTorch; requires a supported GPU platform (see below)
task distill:smoke
task distill DISTILL_CONFIG=python/longlive_distill/configs/browser-384m.yaml
task distill:export CHECKPOINT=artifacts/longlive-browser-384m/student.pt
```

`torch` is intentionally split out of the base environment (`python/requirements-torch.txt`,
installed by `task setup:torch`) so that local browser development is never blocked by it.
The distill tasks (`distill`, `distill:smoke`, `distill:export`) depend on `setup:torch`.

### Platform support for distillation

Distillation is GPU/CUDA-oriented and cannot run on macOS:

- **macOS is not an option.** NVIDIA/CUDA support for macOS ended after CUDA 10.2 (2019),
  and PyTorch ships no macOS x86_64 wheels after 2.2.2. Intel Macs can install no compatible
  `torch` at all; Apple Silicon can use the MPS backend but never CUDA. Docker on macOS runs a
  Linux VM with no GPU passthrough, so it is CPU-only (fine for `distill:smoke`, impractical for
  real training).
- **Use Python ≤3.13.** PyTorch has no wheels for Python 3.14 on any platform yet.

Supported targets:

- **Linux + NVIDIA** — the intended platform.
- **Windows + NVIDIA** — see below.

#### Windows / WSL2

WSL2 is the recommended path on Windows (closest to the Linux target, with GPU passthrough):

```bash
# In an Ubuntu WSL2 shell with NVIDIA drivers installed on the Windows host
# (WSL2 CUDA passthrough needs no separate driver inside the distro).
sudo apt-get update && sudo apt-get install -y python3.12-venv
python3.12 -m venv .venv
task setup:python
# Install the CUDA build of torch (adjust cuXXX to your CUDA version):
.venv/bin/pip install torch>=2.4 --index-url https://download.pytorch.org/whl/cu124
task distill:smoke
```

Native Windows (PowerShell) also works with CUDA wheels:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r python\requirements-distill.txt
.venv\Scripts\python -m pip install torch>=2.4 --index-url https://download.pytorch.org/whl/cu124
$env:PYTHONPATH="python"; .venv\Scripts\python -m longlive_distill.train --config python\longlive_distill\configs\smoke.yaml
```

Verify the GPU is visible before a real run: `python -c "import torch; print(torch.cuda.is_available())"`.

The initial export is a timestep-conditioned `denoiser.onnx`; it is intentionally not copied over the browser's causal `generator.onnx`. Recommended progression is 390M BF16 teacher matching, then few-step/DMD tuning, then packaging the distilled sampler behind the browser generator contract, INT8 validation, and only then INT4 weight quantization for WebGPU. The smoke configuration uses synthetic tensors solely to validate the optimizer and export plumbing; production training requires real VAE latents and text embeddings.

## Task automation

Install [Task](https://taskfile.dev/), then run `task --list`. The Taskfile covers dependency setup, browser development, all checks, distillation smoke/production/export, Wan Smash and benchmarking, packaging, and reproducible-output cleanup.

## Validate

```bash
npm run typecheck
npm test
npm run build
node scripts/check-models.mjs public/models/*/manifest.example.json
```

## Production cautions

- Configure COOP/COEP headers if using threaded WASM fallbacks.
- Cache weights with a service worker or Origin Private File System.
- Validate model licenses before redistributing weights.
- Browser GPU memory is not the same as installed VRAM; handle device loss and out-of-memory errors.
- Keep native Pruna artifacts server-side. Smash compiler outputs are not WebGPU models.
