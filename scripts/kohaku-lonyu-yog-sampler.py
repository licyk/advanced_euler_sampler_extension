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
NAME = 'Kohaku_LoNyu_Yog'
ALIAS = 'kohaku_lonyu_yog'


# sampler

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_Kohaku_LoNyu_Yog(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,
                    s_tmax=float('inf'), s_noise=1., noise_sampler=None, eta=1.):
    """Kohaku_LoNyu_Yog"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigma_down - sigmas[i]

        if i <= (len(sigmas) - 1) / 2:
            x2 = - x
            denoised2 = model(x2, sigma_hat * s_in, **extra_args)
            d2 = to_d(x2, sigma_hat, denoised2)

            x3 = x + ((d + d2) / 2) * dt
            denoised3 = model(x3, sigma_hat * s_in, **extra_args)
            d3 = to_d(x3, sigma_hat, denoised3)

            real_d = (d + d3) / 2
            x = x + real_d * dt

            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        else:
            x = x + d * dt
    return x



# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    new_samplers = [(NAME, sample_Kohaku_LoNyu_Yog, [ALIAS], {})]
    new_samplers_data = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in new_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers.all_samplers += new_samplers_data
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
