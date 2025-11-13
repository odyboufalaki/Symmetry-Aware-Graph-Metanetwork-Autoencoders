import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

from nfn.common import WeightSpaceFeatures
import math
from src.data.base_datasets import Batch
from ..scalegmn.inr import INR, make_coordinates


def select_criterion(criterion: str, criterion_args: dict) -> nn.Module:
    _map = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(**criterion_args),
        'MSE': nn.MSELoss(),
        'BCE': nn.BCELoss(),
        'ReconstructionLoss': weighted_mse_loss,  # Allows for weighted loss
        'KnowledgeDistillation': KnowledgeDistillationLoss(**criterion_args),
        'MSELoss': nn.MSELoss(**criterion_args, reduction='mean'),
    }
    if criterion not in _map.keys():
        raise NotImplementedError
    else:
        return _map[criterion]


def L2_distance(x, x_hat, batch_size=1):
    """
    Compute L2 Loss between the inputs.
    """

    if isinstance(x, torch.Tensor) and isinstance(x_hat, torch.Tensor):
        loss = torch.square(x - x_hat).sum() / batch_size

    elif isinstance(x, dict) and isinstance(x_hat, dict):
        loss = 0
        for key in x:
            loss += torch.square(x[key] - x_hat[key]).sum()

        loss = loss / batch_size

    elif isinstance(x, WeightSpaceFeatures):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, Batch):
        diff_weights = sum([torch.sum(torch.square(w1 - w2)) for w1, w2 in zip(x_hat.weights, x.weights)])
        diff_biases = sum([torch.sum(torch.square(b1 - b2)) for b1, b2 in zip(x_hat.biases, x.biases)])
        loss = (diff_weights + diff_biases) / batch_size

    elif isinstance(x, tuple):
        # problem here w1 and w2 are 3D not 4D
        #diff_weights = sum([torch.sum(torch.square(w1 - w2), (1,2,3)) for w1, w2 in zip(x_hat[0], x[0])])
        #diff_biases = sum([torch.sum(torch.square(b1 - b2), (1,2)) for b1, b2 in zip(x_hat[1], x[1])])
        diff_weights = sum([torch.sum(torch.square(w1 - w2), (1,2)) for w1, w2 in zip(x_hat[0], x[0])])
        diff_biases = sum([torch.sum(torch.square(b1 - b2), (1)) for b1, b2 in zip(x_hat[1], x[1])])
        loss = (diff_weights + diff_biases) / batch_size
    else:
        raise NotImplemented

    return loss


def weighted_mse_loss(input, target, weight=None):
    """
    Compute a weighted mean squared error loss.
    """
    if weight is not None:
        return torch.mean(weight * torch.square(input - target))
    else:
        return mse_loss(input, target)


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss, calculating the KL Divergence between the original cnn 
    probs and the reconstructed cnn probs. We only implement the KL Divergence part for 
    matching two sets of logits (it is the same as using Cross Entropy Loss for training).
    """
    def __init__(self, temperature=0.6, reduction='batchmean', scale_loss=True):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.scale_loss = scale_loss

    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: Raw logits from the student model.
            teacher_logits: Raw logits from the teacher model.
        """
        # Calculate soft targets from the teacher
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Calculate student's log probabilities
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Calculate KL divergence loss
        loss = F.kl_div(
            input=student_log_probs,
            target=teacher_probs,
            reduction=self.reduction
        )
        
        # Scale the loss
        if self.scale_loss:
            loss = loss * (self.temperature * self.temperature)
            
        return loss