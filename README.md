# ComfyUI-Fibo-1.5

Custom ComfyUI nodes for running **BRIA Fibo 1.5 GGUF** models with City96's ComfyUI-GGUF backend.

This project adds a Fibo-specific runtime to ComfyUI while keeping the original BRIA/Diffusers-style transformer tensor names. It is intended for running quantized Fibo 1.5 transformer models without requiring a destructive tensor-key remap.

> **Status:** Experimental / early release. Text-to-image generation is working, including Fibo's 48-channel latent path, SmolLM3 conditioning, GGUF transformer loading, and Wan VAE decoding.

## Features

- Fibo 1.5 GGUF transformer loader
- City96 ComfyUI-GGUF mmap/GGML operations
- SmolLM3 text encoder loader
- Fibo-specific text conditioning
- 48-channel Fibo latent node
- Wan/Fibo VAE loader and decoder
- Portable config/tokenizer layout inside the custom node
- Q2/Q3/Q4/Q5/Q6/Q8 GGUF support when the model file is compatible
- No hardcoded developer machine paths

## Requirements

- ComfyUI
- [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
- PyTorch / CUDA environment supported by your ComfyUI installation
- `diffusers` version containing `AutoencoderKLWan`
- `transformers`
- `safetensors`
- BRIA Fibo 1.5 model components:
  - Fibo transformer GGUF
  - SmolLM3 text encoder weights
  - Wan/Fibo VAE weights
  - Fibo text encoder/tokenizer/VAE configuration files

The Fibo model itself is **not included** in this repository.

## Installation

Clone or extract this repository into:

```text
ComfyUI/
└── custom_nodes/
    └── ComfyUI-Fibo-GGUF/
```

Install City96's ComfyUI-GGUF separately under `custom_nodes`.

Restart ComfyUI after installation.

## Repository layout

```text
ComfyUI-Fibo-GGUF/
├── __init__.py
├── nodes.py
├── fibo_model.py
├── fibo_runtime.py
└── config/
    ├── vae/
    │   └── config.json
    ├── text_encoder/
    │   └── config.json
    └── tokenizer/
        └── ...
```

The runtime expects its configuration/tokenizer assets inside the custom-node directory. It does not depend on a developer-specific absolute path.

## Model placement

### Transformer GGUF

Place Fibo GGUF files in either:

```text
ComfyUI/models/diffusion_models/
```

or:

```text
ComfyUI/models/unet/
```

The **Fibo GGUF Loader** scans both locations.

The GGUF must identify itself with:

```text
general.architecture = fibo
```

### Text encoder

Place the merged SmolLM3/Fibo text encoder in:

```text
ComfyUI/models/text_encoders/
```

Example:

```text
Fibo-1.5-text-encoder-BF16.safetensors
```

### VAE

Place the Fibo/Wan VAE weights in:

```text
ComfyUI/models/vae/
```

## Included nodes

### Fibo GGUF Loader

Loads a Fibo transformer GGUF and constructs the Fibo transformer using City96's GGML-aware operations.

### Fibo Text Encoder Loader

Loads the Fibo 1.5 SmolLM3 text encoder from ComfyUI's standard `models/text_encoders` directory.

Available dtypes:

- BF16
- FP16
- FP32

### Fibo Text Encode

Creates Fibo conditioning from the SmolLM3 hidden states.

The current runtime uses:

- concatenated final two SmolLM3 hidden states for the main 4096-dimensional context
- per-layer hidden states for Fibo's per-block caption projections
- the text attention mask

Fibo responds much better to its intended **structured JSON-style prompts** than to short free-form prompts.

### Fibo Empty Latent (48ch)

Creates the native Fibo latent:

```text
48 channels
spatial downscale: 16x
```

Do not substitute a normal 4-channel or 16-channel SD/Flux latent.

### Fibo VAE Loader

Loads the Wan-based Fibo VAE and decodes the 48-channel latent.

Available dtypes:

- FP32
- BF16
- FP16

The decoder releases the diffusion model from VRAM before moving the VAE to CUDA, which is useful on lower-VRAM GPUs.

## Recommended generation settings

Fibo 1.5 is a distilled model. A good starting point is:

```text
Steps: 4-6
CFG: 1.0
Sampler: Euler
```

**CFG 1.0 is important.** Fibo 1.5 is intended to run without normal classifier-free guidance amplification.

For testing quant quality, keep the prompt, seed, resolution, steps, sampler and CFG identical between GGUF tiers.

## Prompting

Fibo 1.5 performs best with detailed structured prompts rather than short prose.

Example:

```json
{
  "short_description": "A cinematic photograph of a tabby cat sitting beside a sunlit window.",
  "objects": [
    {
      "object_id": "cat",
      "type": "animal",
      "description": "A realistic brown tabby cat with detailed striped fur and amber eyes.",
      "position": "Center frame, sitting upright on a wooden windowsill."
    }
  ],
  "lighting": {
    "conditions": "Warm natural late-afternoon window light."
  },
  "style_medium": "Photography",
  "artistic_style": "Photorealistic naturalism."
}
```

The JSON is still ultimately text conditioning; the structure helps present the scene in the distribution Fibo expects.

## Basic workflow

```text
Fibo GGUF Loader ────────────────┐
                                 │
Fibo Text Encoder Loader         │
          ↓                      │
Fibo Text Encode ────────────────┤
                                 ↓
Fibo Empty Latent (48ch) → KSampler
                                 ↓
                         Fibo VAE Loader
                                 ↓
                            Save Image
```

Use CFG `1.0` and start at 4-6 steps.

## Quantized models

The loader is designed for Fibo GGUF files produced from the same transformer architecture. Typical llama.cpp tiers include:

```text
Q2_K
Q3_K_M
Q4_K_S
Q4_K_M
Q5_K_M
Q6_K
Q8_0
```

Higher quantization tiers generally preserve more model fidelity at the cost of larger files and more memory.

Quantization quality should be evaluated with controlled A/B tests rather than different random seeds.

## Troubleshooting

### `Expected GGUF arch fibo`

The model metadata is not marked as the Fibo architecture. The runtime intentionally rejects unrelated GGUF architectures.

### `City96 ComfyUI-GGUF was not found`

Install City96's ComfyUI-GGUF in your ComfyUI `custom_nodes` directory and restart ComfyUI.

### Text encoder does not appear

Make sure the `.safetensors` file is inside:

```text
ComfyUI/models/text_encoders/
```

Then restart or refresh ComfyUI's model list.

### VAE config not found

Verify:

```text
ComfyUI-Fibo-GGUF/config/vae/config.json
```

exists.

### Tokenizer/config not found

Verify the repository contains:

```text
config/text_encoder/config.json
config/tokenizer/
```

### Decode appears to hang

Wan VAE decoding can be memory intensive. The current decoder unloads the diffusion model before moving the VAE to CUDA and prints decode timing information to the ComfyUI console.

Look for:

```text
[Fibo VAE] decode start
[Fibo VAE] freed diffusion-model VRAM
[Fibo VAE] VAE moved to ...
[Fibo VAE] core decode finished
[Fibo VAE] decode complete
```

Those lines show which stage is taking the time.

## Current limitations

- Experimental runtime; not yet a native upstream ComfyUI Fibo architecture.
- Requires City96 ComfyUI-GGUF.
- Fibo's structured prompting is not automatically generated by these nodes; supply the structured prompt yourself.
- Exact text rendering remains generative and is not guaranteed.
- VAE decoding can be expensive on low-VRAM GPUs.
- Image-edit pipeline support is not currently the focus of this implementation.
- W4A8 support is separate from the GGUF loader and is not currently exposed through these nodes.

## Credits

- **BRIA AI** — Fibo / Fibo 1.5 architecture and model
- **Hugging Face Diffusers** — Fibo and Wan reference components
- **City96** — ComfyUI-GGUF and GGML-backed ComfyUI operations
- **ComfyUI** — node/runtime ecosystem

## License and model terms

This repository contains custom integration/runtime code. The upstream Fibo model has its own license and usage terms.

**Downloading or using Fibo weights does not automatically grant the same rights as this custom-node code.** Review and comply with BRIA's current model license before distributing or using the model weights.

This project is unofficial and is not affiliated with or endorsed by BRIA AI, Hugging Face, City96, or ComfyUI.
