"""
loss functions
Quan Yuan
2018-05-15
"""

import torch
from torch import nn

"""
Shorthands for loss:
- SetLoss: loss with weighted average of all features
"""
__all__ = ['WeightedAverageLoss']


class WeightedAverageLoss(nn.Module):
    """Weighted avearge loss.
    assume the last element of the feature is a weight vector
    """

    def __init__(self, seq_size=4):
        super(WeightedAverageLoss, self).__init__()
        self.seq_size = seq_size


    def forward(self, x, pids):
        feature = x[:,:,:-1]
        weight = x[:,:, -1] #.unsqueeze(2)
        # weighted feature
        weight_size = list(weight.size())
        weight_expand = weight.unsqueeze(2).expand(weight_size[0],weight_size[1],list(feature.size())[2])
        weighted_features = feature*weight_expand
