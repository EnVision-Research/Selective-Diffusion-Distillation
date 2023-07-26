import torch
from torch import nn
import torch.nn.functional as F


class L2Loss(nn.Module):
    def __init__(self, space, loss_weight):
        super(L2Loss, self).__init__()
        self.space = space
        self.loss_weight = loss_weight

    def forward(self, x, x_hat):
        if self.space == 'style':
            loss = 0
            for c_hat, c in zip(x_hat, x):
                loss += F.mse_loss(c_hat, c)
                
        elif self.space == 'w+' or self.space == 'image':
            loss = F.mse_loss(x, x_hat)
        else:
            raise NotImplementedError

        return loss * self.loss_weight
