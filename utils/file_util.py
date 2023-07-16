import importlib
from functools import partial

import random
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import numpy as np
from einops import rearrange, repeat
import torchvision
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def save_dict_to_yaml(dict_value: dict, save_path: str):
    with open(save_path,"w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True,sort_keys=False))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_rows_from_list(samples):
    n_imgs_per_row = len(samples)
    denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
    denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    denoise_grid = torchvision.utils.make_grid(denoise_grid, nrow=n_imgs_per_row)
    return denoise_grid


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)

    return patches



def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # this is from ImageNet dataset
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image
def load_image2(img_path, img_height=None,img_width =None):
    
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image

def im_convert2(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    image = image.transpose(1,2,0)
       # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image
def get_features(image, model, layers=None):

    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features



def rand_bbox(size, res):
    W = size
    H = size
    cut_w = res
    cut_h = res
    tx = np.random.randint(0,W-cut_w)
    ty = np.random.randint(0,H-cut_h)
    bbx1 = tx
    bby1 = ty
    return bbx1, bby1


def rand_sampling(args,content_image):
    bbxl=[]
    bbyl=[]
    bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
    crop_img = content_image[:,:,bby1:bby1+args.crop_size,bbx1:bbx1+args.crop_size]
    return crop_img

def rand_sampling_all(args):
    bbxl=[]
    bbyl=[]
    out = []
    for cc in range(50):
        bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
        bbxl.append(bbx1)
        bbyl.append(bby1)
    return bbxl,bbyl