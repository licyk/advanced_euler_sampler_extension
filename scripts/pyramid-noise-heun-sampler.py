import torch
import tqdm
import random
import k_diffusion.sampling
from modules import sd_samplers_common, sd_samplers_kdiffusion, sd_samplers
from importlib import import_module

sampling = import_module("k_diffusion.sampling")
NAME = 'Heun pyramid'
ALIAS = 'k_heun_pyramid'



def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


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
def sample_heun_pyramid(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1., restart_list=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    step_id = 0

    addition_noise = torch.randn_like(x)
    x = x + pyramid_noise_like2(x)

    def heun_step(x, old_sigma, new_sigma, second_order=True):
        nonlocal step_id
        denoised = model(x, old_sigma * s_in, **extra_args)
        d = sampling.to_d(x, old_sigma, denoised)
        if callback is not None:
            callback({'x': x, 'i': step_id, 'sigma': new_sigma, 'sigma_hat': old_sigma, 'denoised': denoised})
        dt = new_sigma - old_sigma
        if new_sigma == 0 or not second_order:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = model(x_2, new_sigma * s_in, **extra_args)
            d_2 = sampling.to_d(x_2, new_sigma, denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        step_id += 1
        return x

    steps = sigmas.shape[0] - 1
    if restart_list is None:
        if steps >= 10:
            restart_steps = 3
            restart_times = 1
            if steps >= 20:
                restart_steps = steps // 4
                restart_times = 2
            sigmas = get_sigmas_karras(steps - restart_steps * restart_times, sigmas[-2].item(), sigmas[0].item(),
                                       device=sigmas.device)
            restart_list = {0.1: [restart_steps + 1, restart_times, 2]}
        else:
            restart_list = {}

    restart_list = {int(torch.argmin(abs(sigmas - key), dim=0)): value for key, value in restart_list.items()}

    step_list = []
    for i in range(len(sigmas) - 1):
        step_list.append((sigmas[i], sigmas[i + 1]))
        if i + 1 in restart_list:
            restart_steps, restart_times, restart_max = restart_list[i + 1]
            min_idx = i + 1
            max_idx = int(torch.argmin(abs(sigmas - restart_max), dim=0))
            if max_idx < min_idx:
                sigma_restart = get_sigmas_karras(restart_steps, sigmas[min_idx].item(), sigmas[max_idx].item(),
                                                  device=sigmas.device)[:-1]
                while restart_times > 0:
                    restart_times -= 1
                    step_list.extend(zip(sigma_restart[:-1], sigma_restart[1:]))

    last_sigma = None
    for old_sigma, new_sigma in tqdm.tqdm(step_list, disable=disable):
        if last_sigma is None:
            last_sigma = old_sigma
        elif last_sigma < old_sigma:
            # print(f"add noise here,sigma is{sigmas}")
            noise_up = pyramid_noise_like2(torch.randn_like(x),
                                           iterations=4,
                                           discount=0.125)
            x = x + noise_up * s_noise * (old_sigma ** 2 - last_sigma ** 2) ** 0.5
        x = heun_step(x, old_sigma, new_sigma)
        # print(f"now old_sigma is {old_sigma},and new_sigma is {new_sigma}")
        last_sigma = new_sigma

    return x



# add sampler
if not NAME in [x.name for x in sd_samplers.all_samplers]:
    new_samplers = [(NAME, sample_heun_pyramid, [ALIAS], {})]
    new_samplers_data = [
        sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: sd_samplers_kdiffusion.KDiffusionSampler(funcname, model), aliases, options)
        for label, funcname, aliases, options in new_samplers
        if callable(funcname) or hasattr(k_diffusion.sampling, funcname)
    ]
    sd_samplers_kdiffusion.sampler_extra_params["sample_heun_pyramid"] = ["s_churn", "s_tmin", "s_tmax", "s_noise"]
    sd_samplers.all_samplers += new_samplers_data
    sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    sd_samplers.set_samplers()
