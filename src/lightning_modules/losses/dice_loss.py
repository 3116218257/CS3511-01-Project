import torch
from torch import Tensor
import numpy as np
from torch import Tensor, einsum
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.mul(input.reshape(-1), target.reshape(-1)).sum()
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    total_positive = target.sum()
    for channel in range(input.shape[1]):
        w = 1. - (target[:, channel, ...].sum() / total_positive)
        w = w.item()
        if np.isnan(w):
            w = 0.
        
        #print(f"CLASS {channel} : {w}")
        dice += w * dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def calc_dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class GeneralizedDice(nn.Module):
    def __init__(self, **kwargs):
        super(GeneralizedDice, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [0, 1, 2] #kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(probs)
        batch_size = probs.size(0)

        pc = torch.reshape(probs, (batch_size, 3, 1024 * 1024)) #.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
        tc = torch.reshape(target, (batch_size, 3, 1024 * 1024))#.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
        #torch.sum(tc.view(batch_size, -1), dim=1)
        w = 1. / (torch.sum(tc, dim=-1) + 1e-3)
        #w = w.unsqueeze(-1)
        
        intersection = w * torch.sum(pc * tc, dim=2)
        
        union = w * (torch.sum(pc, dim=2) + torch.sum(tc, dim=2))
       
        dice = 1. - 2. * intersection / (union + 1e-3)
        #print("DICE", dice)
        #dice = dice.view(-1)
        mask = torch.sum(tc, dim=2) > 0
        #mask = mask.unsqueeze(-1)
        #mask = mask.to(torch.float32)

        loss = dice * mask
        loss = loss.sum() / (mask.sum() + 1e-3)

        return loss 
