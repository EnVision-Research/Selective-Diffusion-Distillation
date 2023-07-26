# Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation (ICCV 2023)

This is the official implementation of ***SDD*** (ICCV 2023). 

For more details, please refer to:

**Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation [[Paper](https://arxiv.org/abs/2307.08448)]** <br />
Luozhou Wang*, Shuai Yang*, Shu Liu, Yingcong Chen

## Installation
1. Create an environment with python==3.8.0 `conda create -n sdd python==3.8.0`
2. Activate it `conda activate sdd`
3. Install basic requirements `pip install -r requirements.txt`

## Getting Started
### Preparation
1. Prepare data and pretrain checkpoints.

    Data: [CelebA latent code (train)](), [CelebA latent code (test)]()

    Pretrain stylegan2: [stylegan2-ffhq]()
    
    facenet for IDLoss: [facenet]()

2. Prepare your token from [Huggingface](https://huggingface.co). Please place your token at `./TOKEN`.


### Infer with pretrain SDD checkpoint (white hair)
1. Download pretrain SDD checkpoint. Please place it at `./pretrain/white_hair.pt`.
   [white hair](https://drive.google.com/file/d/12_IleMZ9fddKcPaTXy7cbS22JPrY-kKD/view?usp=share_link)

2. Run inference. `python inference.py --config ./configs/white_hair.yml --work_dir work_dirs/white_hair/`

### Train your own SDD
1. Prepare your yaml file.
2. Train SDD. 
`python train.py --config [YOUR YAML] --work_dir [YOUR WORK DIR]`

### Search with HQS
1. Prepare your yaml file.
2. Search with HQS
`python search.py --config [YOUR YAML] --work_dir [YOUR WORK DIR]`


## Citation 
If you find this project useful in your research, please consider citing:

```
@misc{wang2023steps,
      title={Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation}, 
      author={Luozhou Wang and Shuai Yang and Shu Liu and Ying-cong Chen},
      year={2023},
      eprint={2307.08448},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
-  This work is built upon the [Diffusers](https://github.com/huggingface/diffusers) and [stylegan2](https://github.com/NVlabs/stylegan2).
