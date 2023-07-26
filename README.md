# Not All Steps are Created Equal: Selective Diffusion Distillation for Image Manipulation (ICCV 2023)

This is the official implementation of ***SDD*** (ICCV 2023).

Conventional diffusion editing pipeline faces a trade-off problem: adding too much noise affects the fidelity of the image while adding too little affects its editability.
In this paper, we propose a novel framework, Selective Diffusion Distillation (SDD), that ensures both the fidelity and editability of images.
Instead of directly editing images with a diffusion model, we train a feedforward image manipulation network under the guidance of the diffusion model.
Besides, we propose an effective indicator to select the semantic-related timestep to obtain the correct semantic guidance from the diffusion model.
This approach successfully avoids the dilemma caused by the diffusion process.

<p align="center"> <img src="docs/method_SDD.pdf" width="100%"> </p>

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

    Data: [CelebA latent code (train)](https://drive.google.com/file/d/1po13cDdoPp0UK1tfB3TLtP7iLO89kaR-/view?usp=share_link), [CelebA latent code (test)](https://drive.google.com/file/d/1po13cDdoPp0UK1tfB3TLtP7iLO89kaR-/view?usp=share_link)

    Pretrain stylegan2: [stylegan2-ffhq](https://drive.google.com/file/d/1po13cDdoPp0UK1tfB3TLtP7iLO89kaR-/view?usp=share_link)
    
    Facenet for IDLoss: [facenet](https://drive.google.com/file/d/1po13cDdoPp0UK1tfB3TLtP7iLO89kaR-/view?usp=share_link)

2. Prepare your token from [Huggingface](https://huggingface.co). Please place your token at `./TOKEN`.


### Infer with pretrain SDD checkpoint (white hair)
1. Download pretrain SDD checkpoint [white hair](https://drive.google.com/file/d/12_IleMZ9fddKcPaTXy7cbS22JPrY-kKD/view?usp=share_link). 
Please place it at `./pretrain/white_hair.pt`.

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
