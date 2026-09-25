import json
import torch
import torch.nn.functional as F
import folder_paths
import comfy.model_management

from . import nodes as base


FIBO_EDIT_PREFERRED_1024 = [
    (832, 1248), (880, 1184), (912, 1136), (1024, 1024),
    (1136, 912), (1184, 880), (1216, 848), (1248, 832),
    (1264, 816), (1296, 800), (1360, 768),
]


def _fibo_edit_safe_dims(width, height, base_resolution=1024, multiple=16):
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid reference size: {width}x{height}")
    scale = min(1.0, ((base_resolution * base_resolution) / float(width * height)) ** 0.5)
    out_w = max(multiple, int(round(width * scale / multiple)) * multiple)
    out_h = max(multiple, int(round(height * scale / multiple)) * multiple)
    return out_w, out_h


def _fibo_edit_output_dims(width, height, auto_resize=True):
    safe_w, safe_h = _fibo_edit_safe_dims(width, height)
    if not auto_resize:
        return safe_w, safe_h
    ratio = safe_w / safe_h
    return min(FIBO_EDIT_PREFERRED_1024, key=lambda s: abs((s[0] / s[1]) - ratio))


class FiboEditWanVAE(base.FiboWanVAE):
    @torch.inference_mode()
    def encode_edit_references(self, images, mask=None):
        """Encode 1-4 Comfy IMAGE tensors into Fibo Edit reference-token context."""
        import time

        if not images or len(images) > 4:
            raise RuntimeError("Fibo Edit requires between 1 and 4 reference images.")

        self._load()
        device = comfy.model_management.get_torch_device()
        dtype = self._dtype()

        if getattr(device, "type", str(device)) == "cuda":
            try:
                comfy.model_management.unload_all_models()
            except Exception as e:
                print(f"[Fibo Edit VAE] unload_all_models warning: {e}")
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass

        t0 = time.perf_counter()
        model = self.model.to(device=device, dtype=dtype)
        print(f"[Fibo Edit VAE] VAE moved to {device} in {time.perf_counter()-t0:.2f}s")

        packed_refs = []
        ids_refs = []
        first_size = None

        try:
            for ref_index, image in enumerate(images, start=1):
                if not isinstance(image, torch.Tensor) or image.ndim != 4:
                    raise RuntimeError(
                        f"Reference {ref_index} must be a Comfy IMAGE tensor [B,H,W,C], got {type(image)}"
                    )
                if image.shape[0] != 1:
                    raise RuntimeError(
                        "Fibo Edit treats image inputs as references, not a batch. "
                        f"Reference {ref_index} has batch={image.shape[0]}; use one image per input."
                    )
                if image.shape[-1] < 3:
                    raise RuntimeError(f"Reference {ref_index} has fewer than 3 channels: {tuple(image.shape)}")

                orig_h, orig_w = int(image.shape[1]), int(image.shape[2])
                if first_size is None:
                    first_size = (orig_w, orig_h)

                x = image[..., :3].permute(0, 3, 1, 2).float().clamp(0.0, 1.0)

                if ref_index == 1 and mask is not None:
                    m = mask
                    if not isinstance(m, torch.Tensor):
                        raise RuntimeError("Fibo Edit mask must be a Comfy MASK tensor.")
                    if m.ndim == 2:
                        m = m.unsqueeze(0).unsqueeze(0)
                    elif m.ndim == 3:
                        m = m.unsqueeze(1)
                    elif m.ndim == 4 and m.shape[1] == 1:
                        pass
                    else:
                        raise RuntimeError(f"Unsupported mask shape: {tuple(m.shape)}")
                    if m.shape[0] != 1:
                        raise RuntimeError("Fibo Edit mask batch must be 1.")
                    if tuple(m.shape[-2:]) != (orig_h, orig_w):
                        raise RuntimeError(
                            f"Mask size {tuple(m.shape[-2:])} must match first reference {(orig_h, orig_w)}."
                        )
                    m = m.float().clamp(0.0, 1.0)
                    x = x * (1.0 - m) + 0.5 * m

                ref_w, ref_h = _fibo_edit_safe_dims(orig_w, orig_h)
                if (ref_h, ref_w) != (orig_h, orig_w):
                    x = F.interpolate(
                        x, size=(ref_h, ref_w), mode="bicubic", align_corners=False, antialias=True
                    ).clamp(0.0, 1.0)

                x = (x * 2.0 - 1.0).to(device=device, dtype=dtype)

                encoded = model.encode(x.unsqueeze(2))
                latent = encoded.latent_dist.mean
                if latent.ndim != 5 or latent.shape[2] < 1:
                    raise RuntimeError(f"Unexpected Fibo VAE encode shape: {tuple(latent.shape)}")
                latent = latent[:, :, 0, :, :]

                cfg = model.config
                means = getattr(cfg, "latents_mean", None)
                stds = getattr(cfg, "latents_std", None)
                if means is None or stds is None:
                    raise RuntimeError("Fibo Edit requires VAE config latents_mean and latents_std.")
                zdim = latent.shape[1]
                mean = torch.tensor(means, device=device, dtype=latent.dtype).view(1, zdim, 1, 1)
                std = torch.tensor(stds, device=device, dtype=latent.dtype).view(1, zdim, 1, 1)
                latent = (latent - mean) / std

                b, c, lh, lw = latent.shape
                if c != 48:
                    raise RuntimeError(f"Fibo Edit expected 48 VAE latent channels, got {c}.")

                packed = latent.permute(0, 2, 3, 1).reshape(b, lh * lw, c)
                image_ids = torch.zeros((lh, lw, 3), device=device, dtype=latent.dtype)
                image_ids[..., 0] = float(ref_index)
                image_ids[..., 1] = torch.arange(lh, device=device, dtype=latent.dtype)[:, None]
                image_ids[..., 2] = torch.arange(lw, device=device, dtype=latent.dtype)[None, :]
                image_ids = image_ids.reshape(1, lh * lw, 3)

                packed_refs.append(packed.detach().cpu())
                ids_refs.append(image_ids.detach().cpu())
                print(
                    f"[Fibo Edit] reference {ref_index}: {orig_w}x{orig_h} -> "
                    f"{ref_w}x{ref_h}, latent={lh}x{lw}, tokens={lh*lw}"
                )
        finally:
            model.to("cpu")
            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                pass

        return {
            "latents": torch.cat(packed_refs, dim=1),
            "ids": torch.cat(ids_refs, dim=1),
            "reference_count": len(packed_refs),
            "first_size": first_size,
        }


class FiboEditVAELoader(base.FiboVAELoader):
    def load(self, vae_name, dtype, tiling):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return (FiboEditWanVAE(path, dtype, tiling),)


class FiboEditTextEncode(base.FiboTextEncode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("FIBO_TEXT_ENCODER",),
                "prompt": ("STRING", {"multiline": True, "default": '{"edit_instruction":""}'}),
                "max_tokens": ("INT", {"default": 3000, "min": 32, "max": 3000, "step": 32}),
                "transformer_layers": ("INT", {"default": 46, "min": 1, "max": 128}),
            }
        }
    CATEGORY = "Fibo/Edit"
    TITLE = "Fibo Edit Text Encode"

    def encode(self, text_encoder, prompt, max_tokens, transformer_layers):
        try:
            obj = json.loads(prompt)
        except Exception as e:
            raise RuntimeError("Fibo Edit prompt must be valid JSON.") from e
        if not isinstance(obj, dict) or "edit_instruction" not in obj:
            raise RuntimeError('Fibo Edit JSON must contain an "edit_instruction" key.')
        return super().encode(text_encoder, prompt, max_tokens, transformer_layers)


class FiboEditReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image_1": ("IMAGE",),
                "auto_resize": (["On", "Off"], {"default": "On"}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ("FIBO_EDIT_REFERENCE", "INT", "INT")
    RETURN_NAMES = ("references", "width", "height")
    FUNCTION = "encode"
    CATEGORY = "Fibo/Edit"
    TITLE = "Fibo Edit Reference Encode"

    def encode(self, vae, image_1, auto_resize, image_2=None, image_3=None, image_4=None, mask=None):
        if not hasattr(vae, "encode_edit_references"):
            raise RuntimeError("Use the Fibo VAE Loader with Fibo Edit Reference Encode.")
        images = [image_1] + [x for x in (image_2, image_3, image_4) if x is not None]
        if mask is not None and len(images) != 1:
            raise RuntimeError("Fibo Edit masks are supported only with exactly one reference image.")
        refs = vae.encode_edit_references(images, mask=mask)
        first_w, first_h = refs["first_size"]
        width, height = _fibo_edit_output_dims(first_w, first_h, auto_resize == "On")
        print(f"[Fibo Edit] output latent resolution: {width}x{height}")
        return (refs, int(width), int(height))


class FiboEditApplyReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "references": ("FIBO_EDIT_REFERENCE",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "Fibo/Edit"
    TITLE = "Fibo Edit Apply References"

    @staticmethod
    def _apply_one(conditioning, references):
        out = []
        for item in conditioning:
            meta = item[1].copy()
            meta["fibo_edit_latents"] = references["latents"]
            meta["fibo_edit_ids"] = references["ids"]
            out.append([item[0], meta])
        return out

    def apply(self, positive, negative, references):
        return (self._apply_one(positive, references), self._apply_one(negative, references))


EDIT_NODE_CLASS_MAPPINGS = {
    "FiboVAELoader": FiboEditVAELoader,
    "FiboVAELoaderV2": FiboEditVAELoader,
    "FiboEditTextEncode": FiboEditTextEncode,
    "FiboEditReferenceEncode": FiboEditReferenceEncode,
    "FiboEditApplyReferences": FiboEditApplyReferences,
}

EDIT_NODE_DISPLAY_NAME_MAPPINGS = {
    "FiboVAELoader": "Fibo VAE Loader",
    "FiboVAELoaderV2": "Fibo VAE Loader (v2)",
    "FiboEditTextEncode": "Fibo Edit Text Encode",
    "FiboEditReferenceEncode": "Fibo Edit Reference Encode",
    "FiboEditApplyReferences": "Fibo Edit Apply References",
}
