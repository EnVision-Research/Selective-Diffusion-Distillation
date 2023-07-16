from utils.file_util import *
import torch


class BaseSynthesizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generator = instantiate_from_config(self.kwargs.get('generator')).cuda()
        self.dataset = instantiate_from_config(self.kwargs.get('data'))
        self.resolution = self.kwargs.get('resolution')
        self.batch_size = self.kwargs.get('data')['bs_per_gpu']
        sampler = data_sampler(
            self.dataset, shuffle=False, distributed=False
        )
        self.dataloader = torch.utils.data.dataloader.DataLoader(
            self.dataset,
            batch_size=self.kwargs.get('data')['bs_per_gpu'],
            sampler=sampler,
            drop_last=True,
            num_workers=self.kwargs.get('data')['num_workers']
        )
        self.data = sample_data(self.dataloader)

        self.before_latent = None
        self.before_image = None
        self.current_latent = None
        self.current_image = None

    def synthesize_image(self, input=None):
        # add retrain_grad to image
        raise NotImplementedError

    def get_image_gradient(self):
        return torch.abs(self.current_image.grad)

    def get_input(self):
        return next(self.data)

    def log_images(self, img_name, dir):
        raise NotImplementedError

    def log_grads(self, img_name, dir, grad=None):
        raise NotImplementedError

    def get_optimized_target(self):
        raise NotImplementedError

    def get_saved_state_dict(self):
        raise NotImplementedError

    def get_reg_loss_before(self, space):
        raise NotImplementedError

    def get_reg_loss_after(self, space):
        raise NotImplementedError