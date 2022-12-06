import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    
    

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, threshold = 0.6):
    SMOOTH = 1e-6

    # outputs = torch.tensor(outputs,  dtype=torch.int8)

    outputs = torch.sigmoid(outputs)
    outputs = outputs[0].squeeze()
    outputs = (outputs > 0.5)
    outputs = outputs*255
    outputs = outputs.to(dtype=torch.int8)

    # labels = torch.tensor(labels,  dtype=torch.int8)
    labels = labels[0].squeeze()
    labels = labels*255
    labels = labels.to(dtype=torch.int8)

    intersection = torch.count_nonzero(outputs & labels)
    union = torch.count_nonzero(outputs | labels)
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
#     print(iou)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded, iou 
        