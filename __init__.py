"""ComfyUI custom node pack entrypoint for advanced Euler sampler extensions."""

from __future__ import annotations

from pathlib import Path
import k_diffusion.sampling as k_sampling
import comfy.k_diffusion.sampling as comfy_k_sampling
import comfy.samplers

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

SCRIPT_FILES = [
    "euler-negative-sampler.py",
    "euler-dy-sampler.py",
    "euler-dy-negative-sampler.py",
    "euler-max-sampler.py",
    "euler-smea-sampler.py",
    "euler-smea-dy-sampler.py",
    "kohaku-lonyu-yog-sampler.py",
    "pyramid-noise-euler-sampler.py",
    "pyramid-noise-heun-sampler.py",
    "pyramid-noise-dpmpp-2s-sampler.py",
]


FALLBACK_ALIAS_TO_FUNC = {
    "euler_negative": "sample_euler_negative",
    "euler_dy": "sample_euler_dy",
    "euler_dy_negative": "sample_euler_dy_negative",
    "euler_max": "sample_euler_max",
    "euler_smea": "sample_euler_smea",
    "euler_smea_dy": "sample_euler_smea_dy",
    "kohaku_lonyu_yog": "sample_Kohaku_LoNyu_Yog",
    "k_euler_pyramid": "sample_euler_pyramid",
    "k_heun_pyramid": "sample_heun_pyramid",
    "k_dpmpp_2s_pyramid": "sample_dpmpp_2s_pyramid",
}


def _strip_a1111_bits(source: str) -> str:
    cutoff_tokens = [
        "\n# add sampler",
        "\nif not NAME in [x.name for x in sd_samplers.all_samplers]:",
    ]

    cut_positions = [source.find(token) for token in cutoff_tokens if source.find(token) != -1]
    if cut_positions:
        source = source[: min(cut_positions)]

    lines = []
    for line in source.splitlines():
        if "from modules import" in line:
            continue
        lines.append(line)
    return "\n".join(lines)




def _patch_rescaler(ns: dict) -> None:
    cls = ns.get("_Rescaler")
    if cls is None:
        return

    class SafeRescaler(cls):
        def __init__(self, model, x, mode, **extra_args):
            self.model = model
            self.x = x
            self.mode = mode
            self.extra_args = extra_args
            self._had_init_latent = hasattr(model, "init_latent")
            self._had_mask = hasattr(model, "mask")
            self._had_nmask = hasattr(model, "nmask")
            self.init_latent = getattr(model, "init_latent", None)
            self.mask = getattr(model, "mask", None)
            self.nmask = getattr(model, "nmask", None)

        def __enter__(self):
            if self.init_latent is not None and self._had_init_latent:
                self.model.init_latent = __import__("torch").nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4], mode=self.mode)
            if self.mask is not None and self._had_mask:
                self.model.mask = __import__("torch").nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
            if self.nmask is not None and self._had_nmask:
                self.model.nmask = __import__("torch").nn.functional.interpolate(input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._had_init_latent:
                self.model.init_latent = self.init_latent
            if self._had_mask:
                self.model.mask = self.mask
            if self._had_nmask:
                self.model.nmask = self.nmask

    ns["_Rescaler"] = SafeRescaler

def _load_sampler_namespace(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    source = _strip_a1111_bits(source)
    namespace = {
        "__name__": f"comfy_dynamic_{path.stem.replace('-', '_')}",
        "__file__": str(path),
    }
    exec(source, namespace, namespace)
    _patch_rescaler(namespace)
    return namespace


def _register_samplers() -> list[str]:
    registered: list[str] = []

    for script_name in SCRIPT_FILES:
        ns = _load_sampler_namespace(SCRIPTS_DIR / script_name)
        name = ns.get("NAME")
        alias = ns.get("ALIAS")
        sample_fn = next((v for k, v in ns.items() if k.startswith("sample_") and callable(v)), None)

        if not name or not alias or sample_fn is None:
            continue

        fn_name = sample_fn.__name__
        alias_fn_name = f"sample_{alias}"
        # ComfyUI resolves samplers via comfy.k_diffusion.sampling.sample_<sampler_name>
        setattr(k_sampling, fn_name, sample_fn)
        setattr(comfy_k_sampling, fn_name, sample_fn)
        # Also expose alias-based name for non-standard function naming (e.g. Kohaku_LoNyu_Yog)
        setattr(k_sampling, alias_fn_name, sample_fn)
        setattr(comfy_k_sampling, alias_fn_name, sample_fn)

        if alias not in comfy.samplers.KSampler.SAMPLERS:
            comfy.samplers.KSampler.SAMPLERS.append(alias)
        registered.append(alias)

    # Ensure alias-based symbols always exist (including non-matching function names).
    for alias, fn_name in FALLBACK_ALIAS_TO_FUNC.items():
        target = getattr(comfy_k_sampling, fn_name, None) or getattr(k_sampling, fn_name, None)
        if callable(target):
            setattr(k_sampling, f"sample_{alias}", target)
            setattr(comfy_k_sampling, f"sample_{alias}", target)
            if alias not in comfy.samplers.KSampler.SAMPLERS:
                comfy.samplers.KSampler.SAMPLERS.append(alias)
            if alias not in registered:
                registered.append(alias)

    return registered


REGISTERED_SAMPLERS = _register_samplers()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REGISTERED_SAMPLERS",
]
