"""ComfyUI custom node pack entrypoint for advanced Euler sampler extensions."""

from __future__ import annotations

from pathlib import Path
import types
import k_diffusion.sampling as k_sampling
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


def _load_sampler_namespace(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    lines: list[str] = []
    skip_tail = False
    for line in source.splitlines():
        if line.strip().startswith("# add sampler"):
            skip_tail = True
        if skip_tail:
            continue
        if "from modules import" in line:
            continue
        lines.append(line)

    namespace = {
        "__name__": f"comfy_dynamic_{path.stem.replace('-', '_')}",
        "__file__": str(path),
    }
    exec("\n".join(lines), namespace, namespace)
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

        setattr(k_sampling, alias, sample_fn)

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
