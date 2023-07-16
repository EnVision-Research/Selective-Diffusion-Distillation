from typing_extensions import Self
import torch
from torch import nn
from models.mapper import latent_mappers
from models.stylegan2.model import *
from utils.file_util import *


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class StyleMapper(nn.Module):
    def __init__(self, mapper:dict, ckpt):
        super(StyleMapper, self).__init__()
        self.mapper = instantiate_from_config(mapper)
        # self.decoder = StyleGANv2Generator(1024)
        self.decoder = StyleGANv2Generator(512)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.decoder.load_state_dict(torch.load(ckpt, map_location='cpu')['g_ema'], strict=False)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.mapper(x)
        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)
        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images
