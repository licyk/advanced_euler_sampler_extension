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


def _load_sampler_namespace(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    source = _strip_a1111_bits(source)
    namespace = {
        "__name__": f"comfy_dynamic_{path.stem.replace('-', '_')}",
        "__file__": str(path),
    }
    exec(source, namespace, namespace)
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
        # ComfyUI resolves samplers via comfy.k_diffusion.sampling.sample_<sampler_name>
        setattr(k_sampling, fn_name, sample_fn)
        setattr(comfy_k_sampling, fn_name, sample_fn)

        if alias not in comfy.samplers.KSampler.SAMPLERS:
            comfy.samplers.KSampler.SAMPLERS.append(alias)
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
