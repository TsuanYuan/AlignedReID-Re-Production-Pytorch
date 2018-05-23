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
from torchvision.models import vgg16_bn, vgg11_bn

class WeightedReIDSeqFeatureModel(nn.Module):
    def __init__(self, local_conv_out_channels=255, base_model='resnet18', device_id=-1):
        super(WeightedReIDSeqFeatureModel, self).__init__()
        if base_model == 'resnet50':
          self.base = resnet50(pretrained=True, device=device_id)
          planes = 2048
        elif  base_model == 'resnet34':
          self.base = resnet34(pretrained=True, device=device_id)
          planes = 512
        elif base_model == 'resnet18':
          self.base = resnet18(pretrained=True,device=device_id)
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
        if device_id >= 0:
            self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1).cuda(device_id)
            self.quality_weight_fc = nn.Linear(planes, 1).cuda(device_id)
        else:
            self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
            self.quality_weight_fc = nn.Linear(planes, 1)

    def forward(self, x):
        base_conv = self.base(x)
        pre_feat = torch.squeeze(F.avg_pool2d(base_conv, base_conv.size()[2:])) # pre_feat for quality weight
        final_conv_feat = self.final_conv(base_conv)
        condensed_feat = torch.squeeze(F.avg_pool2d(final_conv_feat,final_conv_feat.size()[2:])) # descriptor were fist conv into shorter channels and then average
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
            pre_feat = pre_feat.unsqueeze(0)

        feat = F.normalize(condensed_feat, p=2, dim=1)
        quality_weight_fc = 1/(1+torch.exp(-4*self.quality_weight_fc(pre_feat)))  # quality measure of the feature
        condensed_feat_with_quality = torch.cat((feat, quality_weight_fc),1)
        return condensed_feat_with_quality



class WeightedReIDFeatureModel(nn.Module):
    def __init__(self, local_conv_out_channels=255, base_model='resnet18', device_id=-1):
        super(WeightedReIDFeatureModel, self).__init__()
        if base_model == 'resnet50':
          self.base = resnet50(pretrained=True, device=device_id)
          planes = 2048
        elif  base_model == 'resnet34':
          self.base = resnet34(pretrained=True, device=device_id)
          planes = 512
        elif base_model == 'resnet18':
          self.base = resnet18(pretrained=True,device=device_id)
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
        if device_id >= 0:
            self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1).cuda(device_id)

        else:
            self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)

    def forward(self, x):
        base_conv = self.base(x)
        final_conv_feat = self.final_conv(base_conv)
        condensed_feat = torch.squeeze(F.avg_pool2d(final_conv_feat,final_conv_feat.size()[2:])) # descriptor were fist conv into shorter channels and then average
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        return feat