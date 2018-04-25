import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50, resnet18, resnet34
from torchvision.models import inception_v3
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16_bn, vgg11_bn, vgg13_bn

class Model(nn.Module):
  def __init__(self, local_conv_out_channels=128, num_classes=None, base_model='resnet50'):
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

    self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
    self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
    self.local_relu = nn.ReLU(inplace=True)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal(self.fc.weight, std=0.001)
      init.constant(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)
    # shape [N, C, H, 1]
    local_feat = torch.mean(feat, -1, keepdim=True)
    local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
    # shape [N, H, c]
    local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

    if hasattr(self, 'fc'):
      logits = self.fc(global_feat)
      return global_feat, local_feat, logits

    return global_feat, local_feat, None
