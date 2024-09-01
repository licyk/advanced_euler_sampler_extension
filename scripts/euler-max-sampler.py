import torch
import tqdm
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange, tqdm
from k_diffusion import utils
import math


NAME = 'Euler_Max'
ALIAS = 'euler_max'


@torch.no_grad()
def sample_euler_max(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                   s_tmax=float('inf'), s_noise=1.):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = max(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = k_diffusion.sampling.to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + (math.cos(i + 1)/(i + 1) + 1) * d * dt
    return x


if not NAME in [x.name for x in sd_samplers.all_samplers]:
    new_samplers = [(NAME, sample_euler_max, [ALIAS], {})]
    new_samplers_data = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in new_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += new_samplers_data
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
