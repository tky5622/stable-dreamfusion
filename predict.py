import os
import math
import fire
import numpy as np
import time

import torch
from contextlib import nullcontext
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange

import sys
import os
import importlib

# カレントディレクトリをsys.pathに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.util import  instantiate_from_config
from lovely_numpy import lo
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from transformers import AutoFeatureExtractor
from torch import autocast
from torchvision import transforms


_GPU_INDEX = 0

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# def instantiate_from_config(config):
#     if not "target" in config:
#         if config == "__is_first_stage__":
#             return None
#         elif config == "__is_unconditional__":
#             return None
#         raise KeyError("Expected key `target` to instantiate.")
#     return get_obj_from_str(config["target"])(**config.get("params", dict()))



def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w,
                 ddim_steps, n_samples, scale, ddim_eta,
                 elevation, azimuth, radius):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([elevation,
                              math.sin(azimuth), math.cos(azimuth),
                              radius])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1).float()
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            # print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    print('old input_im:', input_im.size)
    start_time = time.time()

    if preprocess:
        # input_im = load_and_preprocess(models['carvekit'], input_im)
        # input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].
        stt = 1 
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].

        # old method: thresholding background, very important
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print(f'Infer foreground mask (preprocess_image) took {time.time() - start_time:.3f}s.')
    print('new input_im:', lo(input_im))

    return input_im


def main_run(raw_im,
             models, device,
             elevation=0.0, azimuth=0.0, radius=0.0,
             preprocess=True,
             scale=3.0, n_samples=4, ddim_steps=50, ddim_eta=1.0,
             precision='fp32', h=256, w=256):
    '''
    :param raw_im (PIL Image).
    '''
    
    raw_im.thumbnail([1536, 1536], Image.Resampling.LANCZOS)
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (image, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        to_return = [None] * 10
        description = ('###  <span style="color:red"> Unfortunately, '
                       'potential NSFW content was detected, '
                       'which is not supported by our model. '
                       'Please try again with a different image. </span>')
        to_return[0] = description
        return to_return
    else:
        print('Safety check passed.')

    input_im = preprocess_image(models, raw_im, preprocess)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(models['turncam'])
    # used_x = -x  # NOTE: Polar makes more sense in Basile's opinion this way!
    used_elevation = elevation  # NOTE: Set this way for consistency.
    x_samples_ddim = sample_model(input_im, models['turncam'], sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta,
                                  used_elevation, azimuth, radius)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))

    return output_ims

from PIL import Image
import numpy as np

def predict(device_idx: int =_GPU_INDEX,
            ckpt: str ="./105000.ckpt",
            config: str ="configs/sd-objaverse-finetune-c_concat-256.yaml",
            cond_image=None,
            elevation_in_degree: float = 0.0,
            azimuth_in_degree: float = 0.0,
            radius: float = 0.0,
            output_image_path: str = "output.png"):
    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)

    assert os.path.exists(ckpt)
    # assert os.path.exists(cond_image_path)

    # Instantiate all models beforehand for efficiency.
    models = dict()
    print('Instantiating LatentDiffusion...')
    models['turncam'] = load_model_from_config(config, ckpt, device=device)
    print('Instantiating Carvekit HiInterface...')
    # models['carvekit'] = create_carvekit_interface()
    print('Instantiating StableDiffusionSafetyChecker...')
    models['nsfw'] = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker').to(device)
    print('Instantiating AutoFeatureExtractor...')
    models['clip_fe'] = AutoFeatureExtractor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker')

    # cond_image = Image.open(cond_image_path)
    image = Image.fromarray(cond_image)
    image = image.convert("RGB")


    preds_images = main_run(raw_im=image,
                            models=models, device=device,
                            elevation=np.deg2rad(elevation_in_degree),
                            azimuth=np.deg2rad(azimuth_in_degree),
                            radius=radius)

    pred_image = preds_images[-1]
    return pred_image


if __name__ == '__main__':
    '''
    python predict.py --ckpt "path_to_ckpt" \
        --cond_image_path "path_to_cond_image" \
        --elevation_in_degree 30.0 \
        --azimuth_in_degree 0.0 \
        --radius 1.0 \
        --output_image_path "path_to_output_image"            
    '''
    fire.Fire(predict)
