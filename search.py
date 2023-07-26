from torch.utils.tensorboard import SummaryWriter

import os
from tqdm import tqdm

from matplotlib import pyplot as plt

from utils.file_util import *
from utils.dist_util import *

import argparse
from train import Trainer

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
    trainer.search(**d['search'])


if __name__ == "__main__":
    main()
