import torch
import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = torch.tensor(1.0)
        else:
            target_tensor = torch.tensor(0.0)
        target_tensor = target_tensor.expand_as(prediction).to(prediction.device)
        loss = self.loss(prediction, target_tensor)
        return loss
