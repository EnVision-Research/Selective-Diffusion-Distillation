import argparse
import os

from train import Trainer
from utils.dist_util import *
from utils.file_util import *


def main():
    device = "cuda"
    # parse necessary information
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/home/luozhouwang/projects/style-dreamfusion/configs/train.yml')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--work_dir", type=str, default='work_dir/mapper/glasses/')
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

    # start training
    trainer = Trainer(
        synthesizer=synthesizer,
        device=device,
        work_dir=args.work_dir,
        **d['train'],
        search_cfg=d['search']
    )
    trainer.inference(**d['inference'])


if __name__ == "__main__":
    main()
