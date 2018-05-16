"""
Model definitions
Quan Yuan
2018-05-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from BackBones import resnet50, resnet18, resnet34
from torchvision.models import inception_v3
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16_bn, vgg11_bn, vgg13_bn

class WeightedReIDFeatureModel(nn.Module):
    def __init__(self, local_conv_out_channels=255, base_model='resnet18'):
        super(WeightedReIDFeatureModel, self).__init__()
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
        global_feat = torch.squeeze(F.avg_pool2d(feat, feat.size()[2:]))
        if len(list(global_feat.size())) == 1:
            global_feat = global_feat.unsqueeze(0)
        condensed_feat = self.fc(global_feat)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        quality_weight_fc = torch.exp(self.quality_weight_fc(global_feat))  # quality measure of the feature
        condensed_feat_with_quality = torch.cat((feat, quality_weight_fc),1)
        return condensed_feat_with_quality
