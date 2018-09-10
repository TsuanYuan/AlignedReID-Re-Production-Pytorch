"""
Model definitions
Quan Yuan
2018-05-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy
from torch.autograd import Variable
from BackBones import resnet50, resnet18, resnet34, resnet50_with_layers
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


class WeightedReIDFeatureROIModel(nn.Module):
    def __init__(self, local_conv_out_channels=256, base_model='resnet18', device_id=-1):
        super(WeightedReIDFeatureROIModel, self).__init__()
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
            self.final_bn = nn.BatchNorm2d(local_conv_out_channels).cuda(device_id)
            self.final_relu = nn.ReLU(inplace=True).cuda(device_id)
        else:
            self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
            self.final_bn = nn.BatchNorm2d(local_conv_out_channels)
            self.final_relu = nn.ReLU(inplace=True)

    def compute_roi(self, map_w, map_h, w_h_ratio):
        map_ratio = float(map_w)/map_h
        if type(w_h_ratio) is float:
            rv = w_h_ratio
        else:
            rv = w_h_ratio.data.cpu().numpy()
        if rv < map_ratio:   # thin box
            delta_w = min(map_w/2-1,int(round((map_w - map_h * w_h_ratio)/2))) # smaller than w/2, otherwise zero width in the roi
            xs, xe = delta_w, map_w-delta_w
            ys, ye = 0, map_h
        else:   # fat box
            delta_h = min(map_h/2-1, int(round((map_h - map_w / w_h_ratio)/2)))
            ys, ye = delta_h, map_h-delta_h
            xs, xe = 0, map_w

        return xs, xe, ys, ye

    def compute_roi_masks(self, w_h_ratios, feat_size):
        n = feat_size[0]
        if torch.has_cudnn:
            roi_masks = torch.zeros(feat_size).cuda(device=w_h_ratios.get_device())
            roi_areas = torch.zeros(n).cuda(device=w_h_ratios.get_device())
        else:
            roi_masks = torch.zeros(feat_size)
            roi_areas = torch.zeros(n)

        for i in range(n):
            xs, xe, ys, ye = self.compute_roi(feat_size[3], feat_size[2], w_h_ratios[i])
            roi_masks[i, :, ys:ye, xs:xe] = 1
            roi_areas[i] = (ye-ys)*(xe-xs)
        return roi_masks, roi_areas

    def forward(self, x, w_h_ratios):
        base_conv = self.base(x)
        final_conv_feat = base_conv #self.final_relu(self.final_bn(self.final_conv(base_conv)))
        w_h_ratios = w_h_ratios.view(-1)
        # roi pooling, crop the feature map
        conv_feat_size = list(final_conv_feat.size())
        roi_masks, roi_areas = self.compute_roi_masks(w_h_ratios, conv_feat_size)
        masked_feat = final_conv_feat*Variable(roi_masks)
        feature_sum = torch.sum(torch.sum(masked_feat, 3),2)
        roi_areas = roi_areas.unsqueeze(1)
        feature_average = feature_sum/Variable(roi_areas.expand((conv_feat_size[0], conv_feat_size[1])))
        condensed_feat = torch.squeeze(feature_average) # descriptor were fist conv into shorter channels and then average
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        return feat


class WeightedReIDFeatureModel(nn.Module):
    def __init__(self, local_conv_out_channels=256, base_model='resnet18', device_id=-1, num_classes=None):
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
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

    def forward(self, x):
        base_conv = self.base(x)
        final_conv_feat = base_conv #self.final_conv(base_conv)
        condensed_feat = torch.squeeze(F.avg_pool2d(final_conv_feat, final_conv_feat.size()[2:]))  # descriptor were fist conv into shorter channels and then average
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        if hasattr(self, 'fc'):
            logits = self.fc(feat)
            return feat, logits
        return feat, None


class BinaryModel(nn.Module):
    def __init__(self, local_conv_out_channels=256, base_model='resnet18', device_id=-1, num_classes=None):
        super(BinaryModel, self).__init__()
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

        self.fc = nn.Linear(planes, 2)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        base_conv = self.base(x)
        #final_conv_feat = base_conv
        condensed_feat = torch.squeeze(F.max_pool2d(base_conv, base_conv.size()[2:]))
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        #feat = condensed_feat
        logits = self.fc(feat)
        return feat, logits



class MGNModel(nn.Module):
    def __init__(self,
                 num_classes=(1,), base_model='resnet50', local_conv_out_channels=128, parts_model=False):
        super(MGNModel, self).__init__()
        if base_model == 'resnet50':
            self.base = resnet50_with_layers(pretrained=True)
            planes = 2048
        else:
            raise RuntimeError("unknown base model!")

        # self.parts_model = parts_model
        # if num_classes is not None:
        #     self.fc_list = nn.ModuleList()
        #     for i, num_class in enumerate(num_classes):
        #         fc = nn.Linear(local_conv_out_channels, num_class)
        #         init.normal_(fc.weight, std=0.001)
        #         init.constant_(fc.bias, 0)
        #         self.fc_list.append(fc)

        self.global_final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.global_final_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_final_relu = nn.ReLU(inplace=True)

        self.level2_strips = 6
        self.level2_planes = 512
        self.level2_conv_list = nn.ModuleList()
        for k in range(self.level2_strips):
            self.level2_conv_list.append(nn.Sequential(
                nn.Conv2d(self.level2_planes, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.level3_strips = 4
        self.level3_planes = 1024
        self.level3_conv_list = nn.ModuleList()
        for k in range(self.level3_strips):
            self.level3_conv_list.append(nn.Sequential(
                nn.Conv2d(self.level3_planes, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))


        self.fc_list = nn.ModuleList()
        for _ in range(self.level2_strips+self.level3_strips):
            fc = nn.Linear(local_conv_out_channels, 2)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)
        fc = nn.Linear(local_conv_out_channels, 2)
        init.normal_(fc.weight, std=0.001)
        init.constant_(fc.bias, 0)
        self.fc_list.append(fc)

    def forward(self, x):
        """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
        # shape [N, C, H, W]
        feat_final, feat_l3, feat_l2 = self.base(x)
        feat_shorten = self.global_final_relu(self.global_final_bn(self.global_final_conv((feat_final))))
        global_feat = F.max_pool2d(feat_shorten, feat_shorten.size()[2:])
        global_feat = torch.squeeze(global_feat)
        local_feat_list = []
        logits_list = []
        stripe_s2 = float(feat_l2.size(2)) / self.level2_strips
        # stripe_h = int(np.ceil(stripe_s))
        for i in range(self.level2_strips):
            # shape [N, C, 1, 1]
            stripe_start = int(round(stripe_s2 * i))
            stripe_end = int(min(numpy.ceil(stripe_s2 * (i + 1)), feat_l2.size(2)))
            sh = stripe_end - stripe_start
            local_feat = F.max_pool2d(
                feat_l2[:, :, stripe_start: stripe_end, :], (sh, feat_l2.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.level2_conv_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'parts_fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        stripe_s3 = float(feat_l3.size(2)) / self.level3_strips
        # stripe_h = int(np.ceil(stripe_s))
        for i in range(self.level3_strips):
            # shape [N, C, 1, 1]
            stripe_start = int(round(stripe_s3 * i))
            stripe_end = int(min(numpy.ceil(stripe_s3 * (i + 1)), feat_l3.size(2)))
            sh = stripe_end - stripe_start
            local_feat = F.max_pool2d(
                feat_l3[:, :, stripe_start: stripe_end, :], (sh, feat_l3.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.level3_conv_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'parts_fc_list'):
                logits_list.append(self.parts_fc_list[i+self.level2_strips](local_feat))
        if len(global_feat.size()) == 1:
            global_feat = global_feat.unsqueeze(0)
        local_feat_list.append(global_feat)
        logits_list.append(self.fc_list[-1](torch.squeeze(global_feat)))
        # sum up logits and concatinate features
        logits = None
        for i, logit_rows in enumerate(logits_list):
            if i == 0:
                logits = logit_rows
            else:
                logits += logit_rows
        condensed_feat = torch.cat(local_feat_list, dim=1)
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        return feat, logits