# ComfyUI Advanced Euler Sampler Pack

This repository is now structured as a **ComfyUI custom node pack** that registers additional samplers at ComfyUI startup.

## Included samplers

- `euler_negative`
- `euler_dy`
- `euler_dy_negative`
- `euler_max`
- `euler_smea`
- `euler_smea_dy`
- `kohaku_lonyu_yog`
- `k_euler_pyramid`
- `k_heun_pyramid`
- `k_dpmpp_2s_pyramid`

## Installation (ComfyUI)

1. Go to your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:

```bash
git clone https://github.com/licyk/Comfyui_advanced_euler_sampler_extension
```

3. Restart ComfyUI.

## Usage

After restart, these samplers appear in sampler selections (e.g. KSampler sampler_name list), and can be used like built-in samplers.

## Notes

- The original algorithm implementations remain under `scripts/`.
- `__init__.py` dynamically loads sampler functions and registers them with ComfyUI and `k_diffusion.sampling`.
