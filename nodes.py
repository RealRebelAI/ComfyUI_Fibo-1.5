import logging
from pathlib import Path
import torch
import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import comfy.ops

from .fibo_runtime import load_city96
from .fibo_model import FiboModelConfig, FiboBaseModel

NODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = NODE_DIR / "config"

class FiboTextEncoderHandle:
    def __init__(self, text_encoder_path, dtype="bfloat16"):
        self.text_encoder_path = str(text_encoder_path)
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def _find_fibo_repo(self):
        # Portable GitHub layout only. Never depend on the developer's machine.
        root = CONFIG_DIR
        te_cfg = root / "text_encoder" / "config.json"
        tok_dir = root / "tokenizer"
        if te_cfg.is_file() and tok_dir.is_dir():
            return root

        raise RuntimeError(
            "Fibo text config/tokenizer not found inside the custom node. Expected: "
            f"{te_cfg} and tokenizer files under {tok_dir}"
        )

    def load(self):
        if self.model is not None:
            return self

        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        from safetensors.torch import load_file

        repo = self._find_fibo_repo()
        te_cfg = repo / "text_encoder" if (repo / "text_encoder" / "config.json").is_file() else repo
        tok_dir = repo / "tokenizer"

        dt = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tok_dir), local_files_only=True, trust_remote_code=True
        )

        config = AutoConfig.from_pretrained(
            str(te_cfg), local_files_only=True, trust_remote_code=True
        )

        # Build on CPU and load the user's single merged BF16 safetensors.
        self.model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, dtype=dt
        )

        sd = load_file(self.text_encoder_path, device="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        # SmolLM3 may omit lm_head.weight because it is tied to the token embeddings.
        allowed_missing = {"lm_head.weight"}
        real_missing = [k for k in missing if k not in allowed_missing]

        if "lm_head.weight" in missing:
            self.model.tie_weights()
            print("[Fibo] SmolLM3 lm_head.weight omitted; tied to token embeddings.")

        if real_missing:
            raise RuntimeError(
                "Fibo SmolLM3 merged encoder is missing keys. First missing keys: "
                + ", ".join(real_missing[:20])
            )
        if unexpected:
            raise RuntimeError(
                "Fibo SmolLM3 merged encoder has unexpected keys. First unexpected keys: "
                + ", ".join(unexpected[:20])
            )

        self.model.eval()

        # Keep the 3B encoder on CPU by default. Hidden states are moved only
        # while encoding, avoiding permanent VRAM use beside the Q4 transformer.
        self.model.to("cpu")
        return self

def infer_config(sd):
    def shp(k):
        if k not in sd: raise RuntimeError(f"Fibo GGUF missing {k}")
        return tuple(int(x) for x in sd[k].shape)
    dual=0
    while f"transformer_blocks.{dual}.norm1.linear.weight" in sd: dual+=1
    single=0
    while f"single_transformer_blocks.{single}.norm.linear.weight" in sd: single+=1
    if not dual or not single: raise RuntimeError(f"Could not infer Fibo blocks: dual={dual}, single={single}")
    cs=shp("context_embedder.weight"); xs=shp("x_embedder.weight"); cps=shp("caption_projection.0.linear.weight")
    inner=cs[0]
    return dict(in_channels=xs[1],num_layers=dual,num_single_layers=single,attention_head_dim=128,
                num_attention_heads=inner//128,joint_attention_dim=cs[1],text_encoder_dim=cps[1],
                axes_dims_rope=(16,56,56),rope_theta=10000,time_theta=10000)

class FiboGGUFLoader:
    _path_map = {}

    @classmethod
    def _find_ggufs(cls):
        found = {}
        models_dir = Path(folder_paths.models_dir)

        # First include anything Comfy itself has registered.
        for key in ("diffusion_models", "unet"):
            try:
                for name in folder_paths.get_filename_list(key):
                    if not name.lower().endswith(".gguf"):
                        continue
                    try:
                        p = folder_paths.get_full_path(key, name)
                    except Exception:
                        p = None
                    if p:
                        # Keep the visible value filename-like, not a raw absolute path.
                        label = name.replace("\\", "/")
                        if label in found and found[label] != str(p):
                            label = f"{key}/{label}"
                        found[label] = str(p)
            except Exception:
                pass

        # Direct physical scan. This is important for Comfy builds where
        # models/unet exists but is not registered as the "unet" category.
        physical_roots = (
            ("unet", models_dir / "unet"),
            ("diffusion_models", models_dir / "diffusion_models"),
        )

        for category, root in physical_roots:
            if not root.is_dir():
                continue
            for p in root.rglob("*.gguf"):
                rel = p.relative_to(root).as_posix()
                label = rel
                if label in found and found[label] != str(p):
                    label = f"{category}/{rel}"
                found[label] = str(p)

        items = sorted(
            found.items(),
            key=lambda kv: (0 if "fibo" in kv[0].lower() else 1, kv[0].lower())
        )
        cls._path_map = dict(items)
        return [name for name, _ in items]

    @classmethod
    def INPUT_TYPES(cls):
        names = cls._find_ggufs()
        if not names:
            names = ["<no GGUF models found>"]
        return {"required": {"gguf_name": (names,)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Fibo/GGUF"
    TITLE = "Fibo GGUF Loader"

    def load(self, gguf_name):
        loader, ops, city_nodes = load_city96()

        # Refresh so newly-added files can resolve after restart/refresh.
        self._find_ggufs()
        path = self._path_map.get(gguf_name)

        if not path:
            raise RuntimeError(
                f"Could not resolve GGUF model: {gguf_name}\n"
                f"Scanned:\n"
                f"  {Path(folder_paths.models_dir) / 'unet'}\n"
                f"  {Path(folder_paths.models_dir) / 'diffusion_models'}"
            )

        sd, extra = loader.gguf_sd_loader(path)
        if extra.get("arch_str") != "fibo":
            raise RuntimeError(
                f"Expected GGUF arch fibo, got {extra.get('arch_str')!r}"
            )

        uc = infer_config(sd)
        cfg = FiboModelConfig(uc)

        # Critical for GGUF: construct every Fibo Linear with City96's
        # GGML-aware ops so quantized byte-backed tensors are dequantized/
        # cast on demand instead of reaching plain torch F.linear as uint8.
        cfg.custom_operations = ops.GGMLOps

        dev = comfy.model_management.get_torch_device()
        wd = getattr(next(iter(sd.values())), "tensor_type", None)

        params = comfy.utils.calculate_parameters(sd)
        try:
            dt = comfy.model_management.unet_dtype(
                model_params=params,
                supported_dtypes=cfg.supported_inference_dtypes,
                weight_dtype=wd,
            )
        except TypeError:
            dt = comfy.model_management.unet_dtype(
                model_params=params,
                supported_dtypes=cfg.supported_inference_dtypes,
            )

        manual = comfy.model_management.unet_manual_cast(
            dt, dev, cfg.supported_inference_dtypes
        )
        print("[Fibo] constructing transformer with City96 GGMLOps")
        model = FiboBaseModel(cfg, device=dev)
        model.diffusion_model.to(dtype=dt)

        missing, unexpected = model.diffusion_model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(
                "Unexpected Fibo GGUF keys: " + ", ".join(unexpected[:30])
            )
        if missing:
            raise RuntimeError(
                "Missing Fibo GGUF keys: " + ", ".join(missing[:30])
            )

        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=dev,
            offload_device=comfy.model_management.unet_offload_device(),
        )

        # Re-wrap with City96's GGUF patcher so quantized tensors remain mmap/GGML-backed.
        patcher.__class__ = city_nodes.GGUFModelPatcher
        patcher.patch_on_device = False

        return (patcher,)


class FiboINT8DiffusionModelLoader:
    """Load stock-ComfyUI native int8_tensorwise Fibo safetensors."""

    _path_map = {}

    @classmethod
    def _find_int8_models(cls):
        found = {}
        models_dir = Path(folder_paths.models_dir)

        for key in ("diffusion_models", "unet"):
            try:
                for name in folder_paths.get_filename_list(key):
                    if not name.lower().endswith((".safetensors", ".sft")):
                        continue
                    try:
                        p = folder_paths.get_full_path(key, name)
                    except Exception:
                        p = None
                    if p:
                        label = name.replace("\\", "/")
                        if label in found and found[label] != str(p):
                            label = f"{key}/{label}"
                        found[label] = str(p)
            except Exception:
                pass

        for category, root in (
            ("unet", models_dir / "unet"),
            ("diffusion_models", models_dir / "diffusion_models"),
        ):
            if not root.is_dir():
                continue
            for ext in ("*.safetensors", "*.sft"):
                for p in root.rglob(ext):
                    rel = p.relative_to(root).as_posix()
                    label = rel
                    if label in found and found[label] != str(p):
                        label = f"{category}/{rel}"
                    found[label] = str(p)

        items = sorted(
            found.items(),
            key=lambda kv: (
                0 if ("fibo" in kv[0].lower() and "int8" in kv[0].lower()) else
                1 if "fibo" in kv[0].lower() else 2,
                kv[0].lower(),
            ),
        )
        cls._path_map = dict(items)
        return [name for name, _ in items]

    @classmethod
    def INPUT_TYPES(cls):
        names = cls._find_int8_models()
        if not names:
            names = ["<no safetensors diffusion models found>"]
        return {
            "required": {
                "model_name": (names,),
                "compute_dtype": (
                    ["bfloat16", "float16"],
                    {"default": "bfloat16"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Fibo/INT8"
    TITLE = "Fibo INT8 Diffusion Model Loader"

    def load(self, model_name, compute_dtype):
        self._find_int8_models()
        path = self._path_map.get(model_name)
        if not path:
            raise RuntimeError(f"Could not resolve INT8 diffusion model: {model_name}")

        sd, metadata = comfy.utils.load_torch_file(
            path, safe_load=True, return_metadata=True
        )

        # Require native comfy_quant INT8 markers so a random BF16/FP16
        # safetensors cannot accidentally be loaded through this node.
        quant_keys = [k for k in sd if k.endswith(".comfy_quant")]
        if not quant_keys:
            raise RuntimeError(
                "Selected file contains no native ComfyUI .comfy_quant markers. "
                "Expected a Fibo int8_tensorwise safetensors model."
            )

        # Validate at least one marker before constructing the 8B model.
        import json
        marker_ok = False
        marker_error = None
        for key in quant_keys[:8]:
            try:
                raw = bytes(sd[key].detach().cpu().to(torch.uint8).tolist())
                cfg_blob = json.loads(raw.decode("utf-8").strip())
                if cfg_blob.get("format") == "int8_tensorwise":
                    marker_ok = True
                    break
            except Exception as e:
                marker_error = e

        if not marker_ok:
            raise RuntimeError(
                "Selected model does not advertise int8_tensorwise quantization "
                f"in its .comfy_quant markers. Last marker error: {marker_error}"
            )

        uc = infer_config(sd)
        cfg = FiboModelConfig(uc)

        dt = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[compute_dtype]

        # Native Comfy quantized checkpoints must be constructed with
        # mixed_precision_ops. load_state_dict consumes each module's
        # .comfy_quant / weight_scale side tensors and wraps weight as a
        # comfy-kitchen QuantizedTensor.
        cfg.custom_operations = comfy.ops.mixed_precision_ops(
            {}, compute_dtype=dt
        )

        dev = comfy.model_management.get_torch_device()
        print(
            f"[Fibo INT8] loading native int8_tensorwise model "
            f"with compute dtype {dt}: {path}"
        )

        model = FiboBaseModel(cfg, device=dev)
        missing, unexpected = model.diffusion_model.load_state_dict(sd, strict=False)

        # Quant side tensors are consumed by MixedPrecisionOps during load.
        # Anything left over here is useful evidence of a format mismatch.
        bad_missing = [
            k for k in missing
            if not k.endswith((".weight_scale", ".comfy_quant"))
        ]
        bad_unexpected = [
            k for k in unexpected
            if not k.endswith((".weight_scale", ".comfy_quant"))
        ]

        if bad_missing:
            raise RuntimeError(
                "Fibo INT8 model is missing keys: " + ", ".join(bad_missing[:30])
            )
        if bad_unexpected:
            raise RuntimeError(
                "Fibo INT8 model has unexpected keys: "
                + ", ".join(bad_unexpected[:30])
            )

        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=dev,
            offload_device=comfy.model_management.unet_offload_device(),
        )
        return (patcher,)

class FiboTextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Native Comfy combo widget. Values are filenames from models/text_encoders.
        names = folder_paths.get_filename_list("text_encoders")
        return {
            "required": {
                "text_encoder_name": (names,),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("FIBO_TEXT_ENCODER",)
    FUNCTION = "load"
    CATEGORY = "Fibo/Text"
    TITLE = "Fibo Text Encoder Loader"

    def load(self, text_encoder_name, dtype):
        # Resolve the selected filename only after the user chooses it.
        path = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)

        if not str(path).lower().endswith(".safetensors"):
            raise RuntimeError(
                "Fibo Text Encoder Loader currently expects a .safetensors text encoder."
            )

        h = FiboTextEncoderHandle(path, dtype).load()
        return (h,)


class FiboWanVAE:
    def __init__(self, weights_path, dtype="float32", tiling="On"):
        self.weights_path = str(weights_path)
        self.dtype_name = dtype
        self.tiling = tiling
        self.model = None

    def _dtype(self):
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype_name]

    def _find_config_dir(self):
        # Portable GitHub layout only. Never depend on the developer's machine.
        config_dir = CONFIG_DIR / "vae"
        config_file = config_dir / "config.json"
        if config_file.is_file():
            return config_dir

        raise RuntimeError(
            "Fibo VAE config.json not found inside the custom node. Expected: "
            f"{config_file}"
        )

    def _load(self):
        if self.model is not None:
            return

        try:
            from diffusers import AutoencoderKLWan
        except Exception as e:
            raise RuntimeError(
                "Fibo VAE requires a Diffusers build with AutoencoderKLWan."
            ) from e

        from safetensors.torch import load_file

        config_dir = self._find_config_dir()
        print("[Fibo] loading Wan VAE config from:", config_dir)
        print("[Fibo] loading Wan VAE weights from:", self.weights_path)

        self.model = AutoencoderKLWan.from_config(str(config_dir))
        sd = load_file(self.weights_path, device="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        if missing:
            raise RuntimeError("Fibo VAE missing keys: " + ", ".join(missing[:30]))
        if unexpected:
            raise RuntimeError("Fibo VAE unexpected keys: " + ", ".join(unexpected[:30]))

        self.model.eval()
        self.model.to(device="cpu", dtype=self._dtype())

    def _configure_tiling(self, model, samples):
        mode = str(self.tiling).lower()
        enable = mode == "on"
        if mode == "auto":
            # Fibo latents are 16x spatially downsampled. Enable tiling for
            # roughly 768px+ outputs where the Wan VAE becomes expensive.
            h = int(samples.shape[-2]) * 16
            w = int(samples.shape[-1]) * 16
            enable = max(h, w) >= 768

        if enable:
            fn = getattr(model, "enable_tiling", None)
            if fn is None:
                raise RuntimeError(
                    "This Diffusers AutoencoderKLWan build does not expose enable_tiling(). "
                    "Update Diffusers or set VAE tiling to Off."
                )
            fn()
            print(f"[Fibo VAE] tiled decode ENABLED ({self.tiling})")
        else:
            fn = getattr(model, "disable_tiling", None)
            if fn is not None:
                fn()
            print(f"[Fibo VAE] tiled decode disabled ({self.tiling})")

    @torch.inference_mode()
    def decode(self, samples):
        import time

        self._load()

        device = comfy.model_management.get_torch_device()
        dtype = self._dtype()

        print(
            f"[Fibo VAE] decode start: latent={tuple(samples.shape)} "
            f"dtype={dtype} device={device}"
        )

        # Our VAE is not a normal Comfy ModelPatcher, so Comfy does not
        # automatically clear the diffusion model before this manual CUDA move.
        # On low-VRAM cards that can cause severe paging/thrashing during decode.
        if getattr(device, "type", str(device)) == "cuda":
            t0 = time.perf_counter()
            try:
                comfy.model_management.unload_all_models()
            except Exception as e:
                print(f"[Fibo VAE] unload_all_models warning: {e}")
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass
            print(f"[Fibo VAE] freed diffusion-model VRAM in {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        model = self.model.to(device=device, dtype=dtype)
        self._configure_tiling(model, samples)
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        print(f"[Fibo VAE] VAE moved to {device} in {time.perf_counter()-t0:.2f}s")

        # Comfy image latent BCHW -> Wan/Fibo BCFHW with one frame.
        latents = samples.to(device=device, dtype=dtype).unsqueeze(2)

        cfg = model.config
        if getattr(cfg, "latents_mean", None) is not None and getattr(cfg, "latents_std", None) is not None:
            zdim = latents.shape[1]
            mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, zdim, 1, 1, 1)
            std = torch.tensor(cfg.latents_std, device=device, dtype=dtype).view(1, zdim, 1, 1, 1)
            latents = latents * std + mean

        t0 = time.perf_counter()
        decoded = model.decode(latents, return_dict=False)[0]
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"[Fibo VAE] core decode finished in {time.perf_counter()-t0:.2f}s; "
            f"decoded={tuple(decoded.shape)}"
        )

        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]

        image = ((decoded.float() / 2.0) + 0.5).clamp(0.0, 1.0)
        image = image.permute(0, 2, 3, 1).cpu()

        model.to("cpu")
        try:
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass

        print(f"[Fibo VAE] decode complete: image={tuple(image.shape)}")
        return image


class FiboVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("vae")
        return {
            "required": {
                "vae_name": (names,),
                "dtype": (["float32", "bfloat16", "float16"], {"default": "float32"}),
                "tiling": (["On", "Auto", "Off"], {"default": "On"}),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "Fibo/VAE"
    TITLE = "Fibo VAE Loader"

    def load(self, vae_name, dtype, tiling):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return (FiboWanVAE(path, dtype, tiling),)

class FiboTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"text_encoder":("FIBO_TEXT_ENCODER",),
                            "prompt":("STRING",{"multiline":True,"default":""}),
                            "max_tokens":("INT",{"default":2048,"min":32,"max":3000,"step":32}),
                            "transformer_layers":("INT",{"default":46,"min":1,"max":128})}}
    RETURN_TYPES=("CONDITIONING",); FUNCTION="encode"; CATEGORY="Fibo/Text"; TITLE="Fibo Text Encode"
    @torch.inference_mode()
    def encode(self,text_encoder,prompt,max_tokens,transformer_layers):
        h=text_encoder.load(); tok=h.tokenizer; model=h.model
        dev = comfy.model_management.get_torch_device()
        model.to(dev)
        if prompt=="":
            ids=torch.full((1,1),128000,dtype=torch.long,device=dev); mask=torch.ones_like(ids)
        else:
            t=tok([prompt],padding="longest",max_length=max_tokens,truncation=True,add_special_tokens=True,return_tensors="pt")
            ids=t.input_ids.to(dev); mask=t.attention_mask.to(dev)
        out=model(ids,attention_mask=mask,output_hidden_states=True,use_cache=False)
        hs=list(out.hidden_states); embeds=torch.cat([hs[-1],hs[-2]],-1)
        layers=hs[-transformer_layers:] if len(hs)>=transformer_layers else hs+[hs[-1]]*(transformer_layers-len(hs))
        layer_tensor=torch.stack(layers,1)
        result = [[embeds.detach().cpu(),{"fibo_text_layers":layer_tensor.detach().cpu(),"fibo_attention_mask":mask.detach().cpu()}]]
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (result,)

class FiboEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"width":("INT",{"default":1024,"min":256,"max":4096,"step":16}),
                            "height":("INT",{"default":1024,"min":256,"max":4096,"step":16}),
                            "batch_size":("INT",{"default":1,"min":1,"max":8})}}
    RETURN_TYPES=("LATENT",); FUNCTION="make"; CATEGORY="Fibo/Latent"; TITLE="Fibo Empty Latent (48ch)"
    def make(self,width,height,batch_size):
        return ({"samples":torch.zeros((batch_size,48,height//16,width//16),dtype=torch.float32)},)

NODE_CLASS_MAPPINGS={
    "FiboGGUFLoader": FiboGGUFLoader,
    "FiboINT8DiffusionModelLoader": FiboINT8DiffusionModelLoader,
    "FiboTextEncoderLoader": FiboTextEncoderLoader,
    "FiboTextEncode": FiboTextEncode,
    "FiboEmptyLatent": FiboEmptyLatent,
    "FiboVAELoader": FiboVAELoader,
    "FiboVAELoaderV2": FiboVAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS={
    "FiboGGUFLoader": "Fibo GGUF Loader",
    "FiboINT8DiffusionModelLoader": "Fibo INT8 Diffusion Model Loader",
    "FiboTextEncoderLoader": "Fibo Text Encoder Loader",
    "FiboTextEncode": "Fibo Text Encode",
    "FiboEmptyLatent": "Fibo Empty Latent (48ch)",
    "FiboVAELoader": "Fibo VAE Loader",
    "FiboVAELoaderV2": "Fibo VAE Loader (v2)",
}
print("[Fibo nodes v10-vae-tiling] registered:", ", ".join(NODE_CLASS_MAPPINGS.keys()))
