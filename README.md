# Browser Video Lab

Three related experiments in one project:

1. **SD-Turbo WebGPU** — browser-side text encoding, denoising and VAE decoding using ONNX Runtime Web.
2. **LongLive WebGPU** — a streaming, causal browser runtime with persistent KV cache for an externally exported LongLive graph.
3. **Wan 2.1 + Pruna Smash** — native CUDA compression, persistence and benchmarking for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.

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
