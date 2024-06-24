import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange, tqdm
from k_diffusion import utils
from k_diffusion.sampling import to_d, default_noise_sampler, get_ancestral_step
import math
from importlib import import_module

sampling = import_module("k_diffusion.sampling")
NAME = 'Euler_A_Test'
ALIAS = 'euler_a_test'


# 仅用作测试
# sampler

@torch.no_grad()
def sample_euler_ancestral_test(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x



# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    euler_max_samplers = [(NAME, sample_euler_ancestral_test, [ALIAS], {})]
    samplers_data_euler_max_samplers = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in euler_max_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += samplers_data_euler_max_samplers
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
