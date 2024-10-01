import torch
import k_diffusion.sampling
import random
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from tqdm.auto import trange
from k_diffusion.sampling import default_noise_sampler, get_ancestral_step
from importlib import import_module

sampling = import_module("k_diffusion.sampling")
NAME = 'Euler pyramid'
ALIAS = 'k_euler_pyramid'



def pyramid_noise_like2(noise, iterations=5, discount=0.4):
    # iterations * discount less than 2, for example, 4 * 0.3, 8 * 0.15,
    b, c, w, h = noise.shape
    u = torch.nn.Upsample(size=(w, h), mode="bilinear").cuda()
    for i in range(iterations):
        r = random.random() * 2 + 2
        wn, hn = max(1, int(w / (r ** i))), max(1, int(h / (r ** i)))
        temp_noise = torch.randn(b, c, wn, hn).cuda()
        noise += u(temp_noise) * discount ** i
        if wn == 1 or hn == 1:
            break
    return noise / noise.std()


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
def sample_euler_pyramid(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1.,
                         noise_sampler=None):
    """using pyramid noise for euler a"""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    # addition noise to original noise
    addition_noise = torch.randn_like(x)
    x = x + pyramid_noise_like2(x)
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = sampling.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            # get pyramid noise
            noise_up = pyramid_noise_like2(noise_sampler(sigmas[i], sigmas[i + 1]),
                                           iterations=8,
                                           discount=0.25)
            x = x + noise_up * s_noise * sigma_up
    return x



# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    new_samplers = [(NAME, sample_euler_pyramid, [ALIAS], {})]
    new_samplers_data = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in new_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers_kdiffusion.sampler_extra_params["sample_euler_pyramid"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
    sd_samplers.all_samplers += new_samplers_data
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
