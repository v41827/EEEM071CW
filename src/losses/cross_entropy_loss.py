# Copyright (c) EEEM071, University of Surrey

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
    - num_classes (int): number of classes
    - epsilon (float): weight
    - use_gpu (bool): whether to use gpu devices
    - label_smooth (bool): whether to apply label smoothing
    - device (torch.device): torch device (cpu, cuda, mps)
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True, device=None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.device = device or torch.device("cpu")
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (batch_size)
        """
        log_probs = self.logsoftmax(inputs)
        targets_onehot = torch.zeros(log_probs.size(), device=self.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
