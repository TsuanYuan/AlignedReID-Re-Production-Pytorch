"""
Model definitions
Quan Yuan
2018-05-11
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .BackBones import resnet50, resnet18, resnet34
from torchvision.models import inception_v3
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16_bn, vgg11_bn, vgg13_bn

class Model(nn.Module):
    def __init__(self, local_conv_out_channels=255, base_model='resnet18'):
        super(Model, self).__init__()
        if base_model == 'resnet50':
          self.base = resnet50(pretrained=True)
          planes = 2048
        elif  base_model == 'resnet34':
          self.base = resnet34(pretrained=True)
          planes = 512
        elif base_model == 'resnet18':
          self.base = resnet18(pretrained=True)
          planes = 512
        elif base_model == 'inception_v3':
          self.base = inception_v3(pretrained=True)
          planes = 1024  # not correct
        elif base_model == 'squeezenet':
          self.base = squeezenet1_0(pretrained=True)
          planes = 1000  # not correct
        elif base_model == 'vgg16':
          vgg = vgg16_bn(pretrained=True)
          self.base = vgg.features
          planes = 512
        elif base_model == 'vgg11':
          vgg = vgg11_bn(pretrained=True)
          self.base = vgg.features
          planes = 512
        else:
          raise RuntimeError("unknown base model!")

        self.fc = nn.Linear(planes, local_conv_out_channels)
        self.quality_weight_fc = nn.Linear(planes, 1)

    def forward(self, x):
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        condensed_feat = self.fc(global_feat)
        quality_weight_fc = self.quality_weight_fc(self.global_feat) # quality measure of the feature
        condensed_feat_with_quality = torch.cat((condensed_feat, quality_weight_fc),0)
        return condensed_feat_with_quality
