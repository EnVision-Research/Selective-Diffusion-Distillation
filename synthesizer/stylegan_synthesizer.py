import torch

from .base_synthesizer import BaseSynthesizer
from utils.file_util import *
import math


STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if
                                    i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]


class StyleganSynthesizer(BaseSynthesizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator.load_state_dict(torch.load(self.kwargs.get('generator_ckpt'), map_location='cpu')['g_ema'], strict=False)


class StyleganMapperSynthesizer(StyleganSynthesizer):
    def __init__(self, **kwargs):
        super(StyleganMapperSynthesizer, self).__init__(**kwargs)
        self.mapper = instantiate_from_config(self.kwargs.get('mapper')).cuda()
        self.input_space = self.dataset.space
        self.work_space = self.kwargs.get('work_space')
        self.trunction = self.kwargs.get('trunction')
        self.latent_weight = self.kwargs.get('latent_weight')
        self.saving_input_latents = self.kwargs.get('saving_input_latents', False)
        self.latents_pool = None
        # self.mean_latent = self.get_mean_latent()
        self.ori_image = None

    def synthesize_image(self, input=None):
        # add retrain_grad to image
        if input is None:
            input = next(self.data)
            input = input.cuda()

        input = self.convert_latent_space(input, self.input_space, self.work_space)
        if self.saving_input_latents:
            if self.latents_pool is None:
                self.latents_pool = input.detach().clone().cpu()
            else:
                self.latents_pool = torch.cat([self.latents_pool, input.detach().clone().cpu()], dim=0)


        if self.work_space == 'style':

            input = self.convert_s_tensor_to_list(input)
            input = [c.cuda() for c in input]

            with torch.no_grad():
                img_ori, _ = self.generator([input],
                                             input_is_latent=True,
                                                   randomize_noise=False,
                                            #        truncation=self.trunction,
                                            # truncation_latent=self.mean_latent,
                                            input_is_stylespace=True
                                                )
                self.before_image = img_ori

            delta = self.mapper(input)
            latent_hat = [c + self.latent_weight * delta_c for (c, delta_c) in zip(input, delta)]
            self.current_latent = latent_hat
            img_gen, _ = self.generator([latent_hat],
                                               input_is_latent=True,
                                               randomize_noise=False,
                                        # truncation=self.trunction,
                                        # truncation_latent=self.mean_latent,
                                        input_is_stylespace=True
                                               )

        elif self.work_space == 'w+':
            with torch.no_grad():
                img_ori, _ = self.generator([input],
                                             input_is_latent=True,
                                                   randomize_noise=False,
                                                #    truncation=self.trunction,
                                            # truncation_latent=self.mean_latent
                                                )
                self.before_image = img_ori
            latent_hat = input + self.latent_weight * self.mapper(input)
            self.current_latent = latent_hat
            img_gen, _ = self.generator([latent_hat],
                                               input_is_latent=True,
                                               randomize_noise=False,
                                        # truncation=self.trunction,
                                        # truncation_latent=self.mean_latent
                                               )
        else:
            raise NotImplementedError

        self.before_latent = input
        img_gen.retain_grad()
        self.current_image = img_gen
        return self.current_image

    def get_image_gradient(self):
        return torch.abs(self.current_image.grad)

    def get_input(self):
        return next(self.data)

    def log_images(self, img_name, dir):
        self.mapper.eval()
        torchvision.utils.save_image(
            torch.cat([self.before_image.detach().cpu(), self.current_image.detach().cpu()], dim=0),
            f'{dir}sample_{img_name}.png',
            normalize=True, range=(-1, 1), nrow=self.batch_size
        )
        self.mapper.train()

    def log_grads(self, img_name, dir, grad=None):
        if grad is None:
            grad = self.get_image_gradient()
        torchvision.utils.save_image(
            grad,
            f'{dir}grad_{img_name}.png',
            normalize=True, range=(0, 1), nrow=self.batch_size
        )

    def get_optimized_target(self):
        return self.mapper.parameters(), 'mapper'

    def get_saved_state_dict(self):
        return self.mapper.state_dict(), 'mapper'

    def convert_s_tensor_to_list(self, batch):
        STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

        s_list = []
        for i in range(len(STYLESPACE_DIMENSIONS)):
            s_list.append(batch[:, :, 512 * i: 512 * i + STYLESPACE_DIMENSIONS[i]])
        return s_list

    def convert_latent_space(self, input, space_in, space_out):
        if space_in == space_out:
            return input
        if space_in == 'normal':
            if space_out == 'w+':
                latent = self.generator.get_latent(input)
                latent = latent.view(self.batch_size, 1, -1)
                latent = latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return latent

    def get_mean_latent(self, time=4096):
        mean_latent = self.generator.mean_latent(time)
        mean_latent = mean_latent.view(1, 1, -1)
        mean_latent = mean_latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1)
        return mean_latent

    def get_reg_loss_before(self, space):
        if space == 'w+' or space =='style':
            return self.before_latent
        elif space == 'image':
            return self.before_image
        else:
            raise NotImplementedError

    def get_reg_loss_after(self, space):
        if space == 'w+' or space == 'style':
            return self.current_latent
        elif space == 'image':
            return self.current_image
        else:
            raise NotImplementedError

    def save_input_latents(self, path, filename=''):
        if self.saving_input_latents:
            torch.save(self.latents_pool, '{}/{}_{}.pt'.format(path, self.work_space, filename))
            torch.save(self.mean_latent, '{}/{}_mean.pt'.format(path, self.work_space))

    def load(self, ckpt):
        weight = torch.load(ckpt, map_location=torch.device('cpu'))
        if 'name' in weight:
            weight = weight['name']
        elif 'mapper' in weight:
            weight = weight['mapper']
        elif 'model':
            weight = weight['model']
        self.mapper.load_state_dict(weight)


class StyleganLatentSynthesizer(StyleganSynthesizer):
    def __init__(self, **kwargs):
        super(StyleganLatentSynthesizer, self).__init__(**kwargs)
        self.input_space = self.dataset.space
        self.work_space = self.kwargs.get('work_space')
        self.trunction = self.kwargs.get('trunction')
        self.mean_latent = self.get_mean_latent()

        # prepare the optimized_latent
        self.optimized_latent = next(self.data).cuda()
        self.optimized_latent = self.convert_latent_space(self.optimized_latent, self.input_space, self.work_space).detach().clone()
        # req grad
        if self.work_space == 'w+':
            self.optimized_latent.requires_grad = True
        elif self.work_space == 'style':
            for c, s in enumerate(self.optimized_latent):
                if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                    s.requires_grad = True

        with torch.no_grad():
            self.ori_image, _ = self.generator([self.optimized_latent],
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        truncation=self.trunction,
                                        truncation_latent=self.mean_latent
                                        )
        self.before_latent = self.optimized_latent.clone()

    def synthesize_image(self, input=None):
        if input is not None:
            raise NotImplementedError

        if self.work_space == 'style':
            img_gen, _ = self.generator(
                [self.optimized_latent],
                input_is_latent=True,
                randomize_noise=False,
                input_is_stylespace=True
            )
        elif self.work_space == 'w+':
            img_gen, _ = self.generator([self.optimized_latent],
                                         input_is_latent=True,
                                         randomize_noise=False,
                                        #  truncation=self.trunction,
                                        #  truncation_latent=self.mean_latent
                                        )
        else:
            raise NotImplementedError
        img_gen.retain_grad()
        self.current_image = img_gen
        self.current_latent = self.optimized_latent.detach().clone()
        return self.current_image

    def get_image_gradient(self):
        return torch.abs(self.current_image.grad)

    def get_input(self):
        return self.optimized_latent

    def log_images(self, img_name, dir):
        torchvision.utils.save_image(
            torch.cat([self.ori_image, self.current_image], dim=0),
            f'{dir}sample_{img_name}.png',
            normalize=True, range=(-1, 1), nrow=self.batch_size
        )

    def log_grads(self, img_name, dir, grad=None):
        if grad is None:
            grad = self.get_image_gradient()
        torchvision.utils.save_image(
            grad,
            f'{dir}grad_{img_name}.png',
            normalize=True, range=(0, 1), nrow=self.batch_size
        )

    def get_optimized_target(self):
        # TODO: need support save ckpt
        return [self.optimized_latent], 'latent'

    def get_saved_state_dict(self):
        return self.optimized_latent, 'latent'

    def convert_s_tensor_to_list(self, batch):
        s_list = []
        for i in range(len(STYLESPACE_DIMENSIONS)):
            s_list.append(batch[:, :, 512 * i: 512 * i + STYLESPACE_DIMENSIONS[i]])
        return s_list

    def convert_latent_space(self, input, space_in, space_out):
        if space_in == space_out:
            return input
        if space_in == 'normal':
            if space_out == 'w+':
                latent = self.generator.get_latent(input)
                latent = latent.view(self.batch_size, 1, -1)
                latent = latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1)
            elif space_out == 'style':
                latent = self.generator.get_latent(input)
                with torch.no_grad():
                    _, _, latent_code_init = self.generator([latent], input_is_latent=True, return_latents=True)

                latent = [s.detach().clone() for s in latent_code_init]
            else:
                raise NotImplementedError
        elif space_in == 'w+':
            raise NotImplementedError
        else:
            raise NotImplementedError
        return latent

    def get_mean_latent(self, time=4096, space='w+'):
        mean_latent = self.generator.mean_latent(time)
        if space == 'w+':
            mean_latent = mean_latent.view(1, 1, -1)
            mean_latent = mean_latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1).detach().clone()
        elif space == 'style':
            with torch.no_grad():
                _, _, latent_code_init = self.generator([mean_latent], input_is_latent=True, return_latents=True)

            mean_latent = [s.detach().clone() for s in latent_code_init]
        else:
            raise NotImplementedError
        return mean_latent

    def get_reg_loss_before(self, space):
        if space == 'image':
            return self.ori_image
        else:
            if space == self.work_space:
                return self.ori_latent
            else:
                raise NotImplementedError

    def get_reg_loss_after(self, space):
        if space == 'image':
            return self.current_image
        else:
            if space == self.work_space:
                return self.optimized_latent
            else:
                raise NotImplementedError


class StyleganImageSynthesizer(StyleganSynthesizer):
    def __init__(self, **kwargs):
        super(StyleganImageSynthesizer, self).__init__(**kwargs)
        self.input_space = self.dataset.space
        self.work_space = 'image'
        self.trunction = self.kwargs.get('trunction', 1)
        self.mean_latent = self.get_mean_latent()
        self.saving_input_latents = self.kwargs.get('saving_input_latents', False)
        # prepare the optimized_latent
        optimized_latent = next(self.data).cuda()
        self.optimized_image = self.convert_latent_space(optimized_latent, self.input_space, self.work_space).detach().clone()
        # req grad
        self.optimized_image.requires_grad = True
        self.before_image = self.optimized_image.clone()


    def synthesize_image(self, input=None):
        if input is not None:
            raise NotImplementedError

        self.current_image = self.optimized_image
        return self.current_image

    def get_image_gradient(self):
        return torch.abs(self.current_image.grad)

    def get_input(self):
        return self.optimized_image

    def log_images(self, img_name, dir):
        import os
        os.makedirs(os.path.join(dir, 'original'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'edited'), exist_ok=True)
        torchvision.utils.save_image(
            self.before_image,
            f'{dir}/original/sample_{img_name}.png',
            normalize=True, range=(-1, 1), nrow=self.batch_size
        )
        torchvision.utils.save_image(
            self.current_image,
            f'{dir}/edited/sample_{img_name}.png',
            normalize=True, range=(-1, 1), nrow=self.batch_size
        )

    def log_grads(self, img_name, dir, grad=None):
        if grad is None:
            grad = self.get_image_gradient()
        torchvision.utils.save_image(
            grad,
            f'{dir}grad_{img_name}.png',
            normalize=True, range=(0, 1), nrow=self.batch_size
        )

    def get_optimized_target(self):
        # TODO: need support save ckpt
        return [self.optimized_image], 'image'

    def get_saved_state_dict(self):
        return self.optimized_image, 'image'

    def convert_s_tensor_to_list(self, batch):
        s_list = []
        for i in range(len(STYLESPACE_DIMENSIONS)):
            s_list.append(batch[:, :, 512 * i: 512 * i + STYLESPACE_DIMENSIONS[i]])
        return s_list

    def convert_latent_space(self, input, space_in, space_out):
        if space_in == space_out:
            return input
        if space_in == 'normal':
            if space_out == 'w+':
                latent = self.generator.get_latent(input)
                latent = latent.view(self.batch_size, 1, -1)
                latent = latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1)
            elif space_out == 'style':
                latent = self.generator.get_latent(input)
                with torch.no_grad():
                    _, _, latent_code_init = self.generator([latent], input_is_latent=True, return_latents=True)

                latent = [s.detach().clone() for s in latent_code_init]
            else:
                raise NotImplementedError
        elif space_in == 'w+':
            if space_out == 'image':
                with torch.no_grad():
                    image, _ = self.generator([input],
                                               input_is_latent=True,
                                               randomize_noise=False,
                                               truncation=self.trunction,
                                               truncation_latent=self.mean_latent
                                                )
                    return image
            raise NotImplementedError
        else:
            raise NotImplementedError
        return latent

    def get_mean_latent(self, time=4096, space='w+'):
        mean_latent = self.generator.mean_latent(time)
        if space == 'w+':
            mean_latent = mean_latent.view(1, 1, -1)
            mean_latent = mean_latent.repeat(1, (int(math.log(self.resolution, 2)) - 1) * 2, 1).detach().clone()
        elif space == 'style':
            with torch.no_grad():
                _, _, latent_code_init = self.generator([mean_latent], input_is_latent=True, return_latents=True)

            mean_latent = [s.detach().clone() for s in latent_code_init]
        else:
            raise NotImplementedError
        return mean_latent

    def get_reg_loss_before(self, space):
        if space == 'image':
            return self.before_image
        else:
            raise NotImplementedError

    def get_reg_loss_after(self, space):
        if space == 'image':
            return self.current_image
        else:
            raise NotImplementedError
