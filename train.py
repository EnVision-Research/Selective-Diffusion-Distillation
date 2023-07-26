import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from matplotlib import pyplot as plt

from utils.file_util import *
from utils.dist_util import *

import argparse
import time


class Trainer:

    def __init__(
            self,
            synthesizer,
            device,
            optim,
            guidance_loss,
            regularization_losses,
            work_dir,
            iterations,
            log_image_interval,
            save_ckpt_interval,
            search_cfg,
            max_images,
            vis_grad=False,
            vis_x0=False,
    ):

        self.device = device
        self.work_dir = work_dir
        self.make_work_dir()
        if get_rank() == 0:
            self.writer = SummaryWriter(self.work_dir)
        else:
            self.writer = None

        self.iterations = iterations
        self.iter = 0
        pbar = range(int(iterations) + 1)

        if get_rank() == 0:
            self.pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
        else:
            self.pbar = pbar

        self.optim = optim

        self.synthesizer = synthesizer

        self.guidance_loss = instantiate_from_config(guidance_loss)
        if self.guidance_loss.t_policy == 'predetermined':
            self.guidance_loss.t_policy = 'predetermined_{}'.format(str(self.search(**search_cfg)))
            self.synthesizer.reset_data()
        self.regularization_losses = []
        for cfg in regularization_losses:
            self.regularization_losses.append(instantiate_from_config(cfg))

        self.max_images = max_images
        self.log_image_interval = log_image_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.vis_grad = vis_grad
        self.vis_x0 = vis_x0

    def train(self):
        for idx in self.pbar:
            self.iter = idx
            if self.iter > self.iterations:
                print("Done!")
                break

            batch = self.synthesizer.synthesize_image()

            self.optim.zero_grad()
            loss_dict = dict()
            # guidance loss backward
            gl = self.guidance_loss(batch, self.iter, backward=True)
            loss_dict['guidance_loss'] = gl
            grad = self.synthesizer.get_image_gradient().clone()
            # regularization loss backward
            for regularization_loss in self.regularization_losses:
                if regularization_loss.space == 'w+' or regularization_loss.space =='style':
                    rl = regularization_loss(self.synthesizer.before_latent, self.synthesizer.current_latent)
                elif regularization_loss.space == 'image':
                    rl = regularization_loss(self.synthesizer.before_image, self.synthesizer.before_image)

                rl.backward()
                loss_dict[type(regularization_loss).__name__] = rl

            self.optim.step()

            # after train
            self.log_metric(loss_dict)

            if (self.iter) % self.log_image_interval == 0:
                self.synthesizer.log_images(str(self.iter).zfill(6), self.sample_dir)
                if self.vis_grad:
                    self.synthesizer.log_grads(str(self.iter).zfill(6), self.sample_dir, grad)

            if (self.iter) % self.save_ckpt_interval == 0 or self.iter == self.iterations:
                if self.synthesizer.saving_input_latents:
                    self.synthesizer.save_input_latents(self.work_dir)
                self.save_ckpt()

    def search(self,
               metrics_type,
               select_by,
               min_step,
               max_step,
               search_step,
               vis_step_interval,
               num_sample=100):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        all_metrics_dict = dict()
        for image_idx in range(num_sample):
            batch = self.synthesizer.synthesize_image()
            batch = torch.unsqueeze(batch[0], dim=0)
            metrics_list = np.zeros(int((max_step-min_step)/search_step+1))
            all_t = []
            grad_image_seq = None
            # compute grad from t = 900 to t = 100
            search_dir = os.path.join(self.search_dir, 'img{}'.format(image_idx))
            os.makedirs(search_dir, exist_ok=True)
            for idx, t in enumerate(range(min_step, max_step+1, search_step)):
                self.t = t
                all_t.append(t)
                self.guidance_loss.t_policy = 'fixed_{}'.format(t)
                gl = self.guidance_loss(batch, self.iter, backward=True, return_x0=False)
                grad = self.synthesizer.get_image_gradient().clone()
                self.synthesizer.current_image.grad.data.zero_()
                metrics = self.compute_t_metrics(metrics_type, grad)
                metrics_list[idx] += metrics
                with open('{}/{}'.format(search_dir, 'metrics.txt'), 'a') as f:
                    f.write("metrics-{}_t{}: {}\n".format(metrics_type, t, metrics))
                print("metrics-{}_t{}: {}".format(metrics_type, t, metrics))

                if t % vis_step_interval == 0:
                    grad_resized = F.interpolate(grad, (256, 256), mode='bilinear', align_corners=False)
                    if grad_image_seq is None:
                        grad_image_seq = grad_resized
                    else:
                        grad_image_seq = torch.cat([grad_image_seq, grad_resized])
                    torchvision.utils.save_image(grad, f"{search_dir}/grad_{str(t).zfill(6)}.jpg",
                                                 normalize=True,
                                                 range=(0, 1))

            torchvision.utils.save_image(grad_image_seq, f"{search_dir}/grad_seq.jpg", nrow=grad_image_seq.shape[0])
            torchvision.utils.save_image(self.synthesizer.before_image, f"{search_dir}/img_ori.jpg", normalize=True, range=(-1, 1))
            all_metrics_dict[image_idx] = metrics_list
            # plot
            ax1.set_title(metrics_type)
            ax1.plot(all_t, metrics_list)
            ax2.plot(all_t, metrics_list)
            plt.savefig('{}/{}'.format(search_dir, 't_fig.png'))
            ax2.clear()

        f2, ax = plt.subplots(1, 1)
        sum_metrics = [sum([all_metrics_dict[idx][i] for idx in all_metrics_dict])/len(all_metrics_dict) for i in range(len(all_t))]
        print(all_t)
        ax1.set_title(metrics_type)
        ax.plot(all_t, sum_metrics)
        plt.savefig('{}/{}'.format(self.search_dir, 'data_avg_metrics.png'))

        all_t = np.array(all_t, dtype=int)
        sum_metrics = np.array(sum_metrics, dtype=float)
        if select_by == 'min':
            ret = all_t[sum_metrics.argmin()]
        elif select_by == 'max':
            ret = all_t[sum_metrics.argmax()]
        elif select_by.startswith('below_'):
            threshold = float(select_by.split('_')[-1])
            ret = all_t[np.where(sum_metrics < threshold)].tolist()
        elif select_by.startswith('above_'):
            threshold = float(select_by.split('_')[-1])
            ret = all_t[np.where(sum_metrics > threshold)].tolist()
        else:
            raise NotImplementedError
        print(ret)
        with open('{}/t_result.txt'.format(self.search_dir), 'w') as f:
            f.write(str(ret))
        return ret

    def inference(self, ckpt, data):
        self.synthesizer.load(ckpt)
        dataset = instantiate_from_config(data)
        sampler = data_sampler(
            dataset, shuffle=False, distributed=False
        )
        dataloader = torch.utils.data.dataloader.DataLoader(
            dataset,
            batch_size=data['bs_per_gpu'],
            sampler=sampler,
            drop_last=False,
            num_workers=data['num_workers'],
        )

        self.inference_dir = os.path.join(self.work_dir, 'inference/')
        os.makedirs(os.path.dirname(os.path.join(self.inference_dir, 'original/')), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(self.inference_dir, 'edited/')), exist_ok=True)


        for idx, batch in enumerate(dataloader):
            print('{}/{}'.format(idx, len(dataloader)))
            batch = batch.cuda()
            t = time.time()
            edited_image = self.synthesizer.synthesize_image(batch)
            print('cost: {}'.format(time.time()- t))
            torchvision.utils.save_image(self.synthesizer.before_image,
                                         '{}/{}/{}.jpg'.format(self.inference_dir, 'original', idx), normalize=True,
                                         range=(-1, 1))
            torchvision.utils.save_image(edited_image,
                                         '{}/{}/{}.jpg'.format(self.inference_dir, 'edited', idx), normalize=True,
                                         range=(-1, 1))

        self.synthesizer.save_input_latents(self.work_dir)

    def compute_t_metrics(self, metrics_type, img_grad):
        def entropy(p):
            return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum().item()

        if metrics_type == 'mean':
            return torch.mean(img_grad).item()
        if metrics_type == 'entropy':
            b, c, h, w = img_grad.shape
            img_grad = torch.mean(img_grad, dim=1)
            img_grad = img_grad.view(b, 1, -1)
            img_grad = torch.nn.functional.softmax(img_grad, dim=-1)
            img_grad = img_grad.view(b, 1, h, w)
            return entropy(img_grad)
        raise NotImplementedError

    def log_metric(self, dict):
        if get_rank() == 0:
            # self.pbar.set_description(
            #     (
            #         ' '.join([f"{k}: {v.mean().item():.4f}" for k, v in dict.items()])
            #     )
            # )
            for k, v in dict.items():
                if isinstance(v, float):
                    self.writer.add_scalar(f'train/{k}', v, self.iter)
                else:
                    self.writer.add_scalar(f'train/{k}', (v).mean(), self.iter)

    def save_ckpt(self):
        if get_rank() == 0:
            saved_state_dict, name = self.synthesizer.get_saved_state_dict()
            torch.save(
                {
                    name: saved_state_dict,
                    "optimizer": self.optim.state_dict(),
                },
                f"{self.checkpoint_dir}{name}_{str(self.iter).zfill(6)}.pt"
            )

    def make_work_dir(self):

        self.sample_dir = os.path.join(self.work_dir, 'sample/')
        self.search_dir = os.path.join(self.work_dir, 'search/')
        self.checkpoint_dir = os.path.join(self.work_dir, 'checkpoint/')

        if get_rank() == 0:
            os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.search_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)


def main():
    device = "cuda"
    # parse necessary information
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--work_dir", type=str, default='')
    args = parser.parse_args()

    # read config
    f = open(args.config, 'r', encoding='utf-8')
    d = yaml.safe_load(f)

    # dump config
    os.makedirs(os.path.dirname(args.work_dir), exist_ok=True)
    config_path = os.path.join(args.work_dir, 'config_dump.yml')
    save_dict_to_yaml(d, config_path)

    # set seed
    if 'seed' in d:
        torch.manual_seed(d['seed'])
    else:
        torch.manual_seed(1010)

    # prepare synthesizer
    synthesizer = instantiate_from_config(d['synthesizer'])

    optimized_target, _ = synthesizer.get_optimized_target()
    optimizer = optim.AdamW(
        optimized_target, lr=d['optimizer']['params']['lr'],
        weight_decay=d['optimizer']['params']['weight_decay']
    )

    # start training
    trainer = Trainer(
        synthesizer=synthesizer,
        optim=optimizer,
        device=device,
        work_dir=args.work_dir,
        guidance_loss=d['guidance_loss'],
        regularization_losses=d['regularization_losses'],
        **d['train'],
        search_cfg=d['search']
    )
    trainer.train()


if __name__ == "__main__":
    main()
