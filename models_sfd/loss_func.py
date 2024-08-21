import numpy as np
import torch
from monai.losses import (DiceCELoss,
                          DiceFocalLoss,
                          GeneralizedDiceFocalLoss,
                          GeneralizedWassersteinDiceLoss)


def build_dicefocal_seg_loss(dataset_name):
    if dataset_name == "seafog":
        focal_weight = torch.tensor([10.0])
    elif dataset_name == "ybsf":
        focal_weight = torch.tensor([1.0, 10.0, 1.0])
    else:
        raise NotImplementedError
    loss_func = DiceFocalLoss(
        include_background=False,  # ignore background class, else include_background=True
        to_onehot_y=False,  # no need to convert the input `y` into the one-hot format
        softmax=True,
        gamma=2.0,
        jaccard=True,
        focal_weight=focal_weight,  # one can use this parameter to assign different weights to sea fog, but expriementally, it is not useful
    )
    return loss_func
