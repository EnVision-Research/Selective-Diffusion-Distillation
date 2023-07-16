import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.modules import EqualLinear, PixelNorm

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]


class Mapper(Module):

    def __init__(self, latent_dim=512):
        super(Mapper, self).__init__()

        layers = [PixelNorm()]

        for i in range(4):
            layers.append(
                EqualLinear(
                    latent_dim, latent_dim, lr_mul=0.01, activation='fused_lrelu'
                )
            )

        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class SingleMapper(Module):

    def __init__(self):
        super(SingleMapper, self).__init__()

        self.mapping = Mapper()

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):

    def __init__(self, coarse, medium, fine, coarse_medium_division=4, medium_fine_division=8,):
        super(LevelsMapper, self).__init__()

        self.coarse = coarse
        self.medium = medium
        self.fine = fine

        self.coarse_medium_division = coarse_medium_division
        self.medium_fine_division = medium_fine_division


        if coarse:
            self.course_mapping = Mapper()
        if medium:
            self.medium_mapping = Mapper()
        if fine:
            self.fine_mapping = Mapper()

    def forward(self, x):
        x_coarse = x[:, :self.coarse_medium_division, :]
        x_medium = x[:, self.coarse_medium_division:self.medium_fine_division, :]
        x_fine = x[:, self.medium_fine_division:, :]

        if self.coarse:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)

        if self.medium:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)

        if self.fine:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out


class FullStyleSpaceMapper(Module):

    def __init__(self):
        super(FullStyleSpaceMapper, self).__init__()


        for c, c_dim in enumerate(STYLESPACE_DIMENSIONS):
            setattr(self, f"mapper_{c}", Mapper(latent_dim=c_dim))

    def forward(self, x):
        out = []
        for c, x_c in enumerate(x):
            curr_mapper = getattr(self, f"mapper_{c}")
            x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            out.append(x_c_res)

        return out


class WithoutToRGBStyleSpaceMapper(Module):

    def __init__(self):
        super(WithoutToRGBStyleSpaceMapper, self).__init__()


        indices_without_torgb = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
        self.STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in indices_without_torgb]

        for c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
            setattr(self, f"mapper_{c}", Mapper(latent_dim=STYLESPACE_DIMENSIONS[c]))

    def forward(self, x):
        out = []
        for c in range(len(STYLESPACE_DIMENSIONS)):
            x_c = x[c]
            if c in self.STYLESPACE_INDICES_WITHOUT_TORGB:
                curr_mapper = getattr(self, f"mapper_{c}")
                x_c_res = curr_mapper(x_c.view(x_c.shape[0], -1)).view(x_c.shape)
            else:
                x_c_res = torch.zeros_like(x_c)
            out.append(x_c_res)

        return out