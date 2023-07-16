from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, logging
import time

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class SDLoss(torch.nn.Module):
    def __init__(
            self,
            text,
            max_iter,
            skip=True,
            max_rate=0.98,
            min_rate=0.02,
            t_policy='random',
            num_search=20,
            loss_weight=1.0,
            guidance_scale=100,
            clip_grad=False,
            clip_grad_topk=2048,
            norm=False, # only used when input image is in [0,1]
            noise_type='random',
    ):
        super(SDLoss, self).__init__()
        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(
                f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = 'cuda'
        self.max_iter = max_iter
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * min_rate)
        self.max_step = int(self.num_train_timesteps * max_rate)

        print(f'[INFO] loading stable diffusion...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                                 use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                                         use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.skip = skip
        self.t_policy = t_policy
        self.loss_weight = loss_weight
        self.clip_grad = clip_grad
        self.clip_grad_topk = clip_grad_topk
        self.norm = norm
        self.noise_type = noise_type
        self.num_search = num_search
        self.text = text
        self.guidance_scale = guidance_scale

        print(f'[INFO] loaded stable diffusion!')

    def _encode_text(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # # Do the same for unconditional embeddings
        # uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        # with torch.no_grad():
        #     uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # # Cat for final embeddings
        # text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def _encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        # all imgs from style gan are in range(0, 1)
        if self.norm:
            imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def forward_normal(self, image, iter, backward=True, return_x0=False):
        # text_embeddings = self._encode_text([text]).repeat(image.shape[0], 1, 1)

        cond_txt_embed = self._encode_text([self.text]).repeat(image.shape[0], 1, 1)
        uncond_txt_embed = self._encode_text(['']).repeat(image.shape[0], 1, 1)

        txt_embed = torch.cat([uncond_txt_embed, cond_txt_embed])

        _, _, img_h, img_w = image.shape
        pred_rgb_512 = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False)
        pred_rgb_512.retain_grad()

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self._encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if self.t_policy == 'random':
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        elif self.t_policy.startswith('fixed'):
            t = int(self.t_policy.split('_')[-1])
            t = torch.ones([1], dtype=torch.long, device=self.device) * t
        elif self.t_policy == 'recurrent_increase':
            t = min(max(iter % self.num_train_timesteps, self.min_step), self.max_step)
            t = torch.ones([1], dtype=torch.long, device=self.device) * t
        elif self.t_policy == 'recurrent_decrease':
            t = self.num_train_timesteps - min(max(iter % self.num_train_timesteps, self.min_step), self.max_step)
            t = torch.ones([1], dtype=torch.long, device=self.device) * t
        elif self.t_policy == 'increase':
            t = min(max(int(iter / self.max_iter * self.num_train_timesteps), self.min_step), self.max_step)
            t = torch.ones([1], dtype=torch.long, device=self.device) * t
        elif self.t_policy == 'decrease':
            t = self.num_train_timesteps - min(max(int(iter / self.max_iter * self.num_train_timesteps), self.min_step),
                                               self.max_step)
            t = torch.ones([1], dtype=torch.long, device=self.device) * t
        elif self.t_policy.startswith('predetermined_'):
            t = eval(self.t_policy.split('_')[-1])
            if isinstance(t, list):
                t = torch.ones([1], dtype=torch.long, device=self.device) * random.choice(t)
            else:
                t = torch.ones([1], dtype=torch.long, device=self.device) * t
        else:
            raise NotImplementedError

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            if self.noise_type == 'random':
                noise = torch.randn_like(latents)
            elif self.noise_type == 'zero':
                noise = torch.zeros_like(latents)
            else:
                raise NotImplementedError
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=txt_embed).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        x0_pred = self.scheduler._get_prev_sample(latents_noisy, t, 0, noise_pred)
        x0_pred = x0_pred / 0.18215
        with torch.no_grad():
            x0_pred = self.vae.decode(x0_pred).sample
        x0_pred = F.interpolate(x0_pred, (img_h, img_w), mode='bilinear', align_corners=False)

        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        if self.skip:
            grad = w * (noise_pred - noise) * self.loss_weight

            # clip grad for stable training?
            # grad = grad.clamp(-1, 1)
            if self.clip_grad:
                grad = grad.view(-1)
                grad[torch.where(torch.abs(grad) < torch.topk(torch.abs(grad), self.clip_grad_topk).values[-1])] = 0
                grad = grad.view(-1, 4, 64, 64)

            # manually backward, since we omitted an item in grad and cannot simply autodiff.
            # _t = time.time()
            if backward:
                latents.backward(gradient=grad, retain_graph=True)
            # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
            if return_x0:
                return self.loss_weight * nn.functional.mse_loss(noise, noise_pred), x0_pred, t
            else:
                return self.loss_weight * nn.functional.mse_loss(noise, noise_pred)
        else:
            # _t = time.time()
            loss = nn.functional.mse_loss(noise, noise_pred) * self.loss_weight
            if backward:
                loss.backward(retain_graph=True)

            # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')
            if return_x0:
                return loss, x0_pred, t
            else:
                return loss

    def forward(self, image, iter, backward=True, return_x0=True):
        return self.forward_normal(image, iter, backward=backward, return_x0=return_x0)