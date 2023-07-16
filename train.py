import numpy as np
from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from matplotlib import pyplot as plt

from utils.file_util import *
from utils.dist_util import *


class Trainer:

    def __init__(
            self,
            synthesizer,
            device,
            work_dir,
            iterations,
            log_image_interval,
            save_ckpt_interval,
            search_cfg,
            max_images,
            guidance_loss=None,
            regularization_losses=None,
            optim=None,
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
        self.optim = optim

        if optim:
            if get_rank() == 0:
                self.pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            else:
                self.pbar = pbar

        self.synthesizer = synthesizer

        if guidance_loss:
            self.guidance_loss = instantiate_from_config(guidance_loss)
            if self.guidance_loss.t_policy == 'predetermined':
                self.guidance_loss.t_policy = 'predetermined_{}'.format(str(self.search(**search_cfg)))
        if regularization_losses:
            self.regularization_losses = []
            for cfg in regularization_losses:
                self.regularization_losses.append(instantiate_from_config(cfg))

        self.max_images = max_images
        self.log_image_interval = log_image_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.vis_grad = vis_grad
        self.vis_x0 = vis_x0

    def search(self,
               metrics_type,
               select_by,
               min_step,
               max_step,
               search_step,
               vis_step_interval,
               num_sample=100):
        all_mean_dict = dict()
        all_ent_dict = dict()
        for image_idx in range(num_sample):
            batch = self.synthesizer.synthesize_image()
            # easily vis
            batch = torch.unsqueeze(batch[0], dim=0)
            mean_list = np.zeros(int((max_step-min_step)/search_step+1))
            ent_list = np.zeros(int((max_step-min_step)/search_step+1))
            all_t = []
            grad_image_seq = None
            search_dir = os.path.join(self.search_dir, 'img{}'.format(image_idx))
            os.makedirs(search_dir, exist_ok=True)
            # compute grad from t = min_step to t = max_step
            for idx, t in enumerate(range(min_step, max_step+1, search_step)):
                all_t.append(t)
                self.guidance_loss.t_policy = 'fixed_{}'.format(t)
                gl, _, __ = self.guidance_loss(batch, self.iter, backward=True, return_x0=True)
                grad = self.synthesizer.get_image_gradient().clone()
                self.synthesizer.current_image.grad.data.zero_()
                mean, ent = self.compute_t_metrics(metrics_type, grad)
                mean_list[idx] += mean
                ent_list[idx] += ent
                with open('{}/{}'.format(search_dir, 'mean.txt'), 'a') as f:
                    f.write("metrics-{}_t{}: {}\n".format('mean', t, mean))
                print("metrics-{}_t{}: {}".format('mean', t, mean))
                with open('{}/{}'.format(search_dir, 'ent.txt'), 'a') as f:
                    f.write("metrics-{}_t{}: {}\n".format('ent', t, ent))
                print("metrics-{}_t{}: {}".format('ent', t, ent))

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
            all_mean_dict[image_idx] = mean_list
            all_ent_dict[image_idx] = ent_list


        f2, ax = plt.subplots(1, 1)
        sum_mean = [sum([all_mean_dict[idx][i] for idx in all_mean_dict])/len(all_mean_dict) for i in range(len(all_t))]
        sum_ent = [sum([all_ent_dict[idx][i] for idx in all_ent_dict])/len(all_ent_dict) for i in range(len(all_t))]
        print(all_t)
        # normalize
        nagative_entropy = [-((i - min(sum_ent)) / (max(sum_ent) - min(sum_ent))) + 1 for i in sum_ent]
        mean = [(i - min(sum_mean)) / (max(sum_mean) - min(sum_mean)) for i in sum_mean]
        HQS = [(mean[i] + nagative_entropy[i]) / 2 for i in range(len(mean))]
        ax.plot(all_t, HQS)
        plt.savefig('{}/{}'.format(self.search_dir, 'HQS.png'))

        all_t = np.array(all_t, dtype=int)
        HQS = np.array(HQS, dtype=float)
        if select_by == 'min':
            ret = all_t[HQS.argmin()]
        elif select_by == 'max':
            ret = all_t[HQS.argmax()]
        elif select_by.startswith('below_'):
            threshold = float(select_by.split('_')[-1])
            ret = all_t[np.where(HQS < threshold)].tolist()
        elif select_by.startswith('above_'):
            threshold = float(select_by.split('_')[-1])
            ret = all_t[np.where(HQS > threshold)].tolist()
        else:
            raise NotImplementedError
        print(ret)
        with open('{}/t_result.txt'.format(self.search_dir), 'w') as f:
            f.write(str(ret))
        return ret

    def compute_t_metrics(self, metrics_type, img_grad):
        def entropy(p):
            return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum().item()

        if metrics_type == 'HQS':
            mean = torch.mean(img_grad).item()
            b, c, h, w = img_grad.shape
            img_grad = torch.mean(img_grad, dim=1)
            img_grad = img_grad.view(b, 1, -1)
            img_grad = torch.nn.functional.softmax(img_grad, dim=-1)
            img_grad = img_grad.view(b, 1, h, w)
            ent = entropy(img_grad)
            return mean, ent
        raise NotImplementedError

    def train(self):
        # Our training implementation will be released upon acceptance
        raise NotImplementedError

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

        print('inferene on {} images'.format(len(dataloader)))
        for idx, batch in enumerate(dataloader):
            print('{}/{}'.format(idx, len(dataloader)))
            batch = batch.cuda()
            # t = time.time()
            edited_image = self.synthesizer.synthesize_image(batch)
            # print('cost: {}'.format(time.time()- t))
            torchvision.utils.save_image(self.synthesizer.before_image,
                                         '{}/{}/{}.jpg'.format(self.inference_dir, 'original', idx), normalize=True,
                                         range=(-1, 1))
            torchvision.utils.save_image(edited_image,
                                         '{}/{}/{}.jpg'.format(self.inference_dir, 'edited', idx), normalize=True,
                                         range=(-1, 1))

        print('inferene done!')
        self.synthesizer.save_input_latents(self.work_dir)

    def make_work_dir(self):

        self.sample_dir = os.path.join(self.work_dir, 'sample/')
        self.search_dir = os.path.join(self.work_dir, 'search/')
        self.checkpoint_dir = os.path.join(self.work_dir, 'checkpoint/')

        if get_rank() == 0:
            os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.search_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)