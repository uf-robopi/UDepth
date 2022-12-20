"""
Implementation of SILog loss and domian projection loss
"""
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth


class SILogLoss(nn.Module):  
    """Scale-Invariant Log Loss"""
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)


class USLoss(nn.Module): 
    """Domain Projection Loss"""
    def __init__(self):
        super(USLoss, self).__init__()
        self.name = 'USLoss'
        # coefficient of R term
        self.u1 = 0.49598983 
        # coefficient of M term
        self.u2 = -0.38875134 
        # coefficient of constant term
        self.u3 = 0.46353632 

    def forward(self, image, label):
        image = nn.functional.interpolate(image,(240,320))

        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        gb_max = torch.max(g,b)

        logit = self.u3 + self.u1 * r + self.u2 * gb_max

        logit_re = torch.reshape(logit, (logit.shape[0], 1, 240, 320))

        return F.mse_loss(logit_re, label)


