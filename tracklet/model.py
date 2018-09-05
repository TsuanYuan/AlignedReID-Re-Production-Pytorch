"""
model definition for tracklet matching
Quan Yuan
2018-09-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy
from torch.autograd import Variable

class Bottleneck1D(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck1D, self).__init__()
    self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=9, bias=False)
    self.bn1 = nn.BatchNorm1d(planes)
    self.conv2 = nn.Conv1d(planes, planes, kernel_size=9, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm1d(planes)
    self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=9, bias=False)
    self.bn3 = nn.BatchNorm1d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride


class Track1DModel(nn.Module):
    def __init__(self, planes, inplanes=5, stride=1):
        super(Track1DModel, self).__init__()

        self.base = nn.Sequential(nn.Conv1d(inplanes, planes, kernel_size=9, bias=False),
                                  nn.BatchNorm1d(planes),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(planes, planes, kernel_size=9, stride=stride,
                               padding=1, bias=False),
                                  nn.BatchNorm1d(planes),
                                  nn.ReLU(inplace=True),
                                  nn.Conv1d(planes, planes * 4, kernel_size=9, bias=False),
                                  nn.BatchNorm1d(planes * 4),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        forward_outs = self.base(x)
        forward_feature = F.avg_pool1d(forward_outs, forward_outs.size()[2:])
        r = x.flip(2)
        backward_outs = self.base(r)
        backward_feature = F.avg_pool1d(backward_outs, backward_outs.size()[2:])
        return forward_feature, backward_feature