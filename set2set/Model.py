"""
Model definitions
Quan Yuan
2018-05-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
from BackBones import resnet50, resnet18, resnet34, resnet50_with_layers
from torchvision.models import inception_v3
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16_bn, vgg11_bn
from parts_models import se_resnet


class SwitchClassHeadModel(nn.Module):
    def __init__(self, local_conv_out_channels=128, final_conv_out_channels=512,
                 num_classes=(1,), base_model='resnet50', with_final_conv=False, parts_model=False, num_stripes=4):
        super(SwitchClassHeadModel, self).__init__()
        if base_model == 'resnet50':
            self.base = resnet50(pretrained=True)
            planes = 2048
        elif base_model == 'resnet34':
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

        self.with_final_conv = with_final_conv
        self.parts_model = parts_model
        self.num_stripes = num_stripes
        self.num_classes = num_classes
        self.num_classification_head = len(num_classes)
        if with_final_conv:
            self.final_conv = nn.Conv2d(planes, final_conv_out_channels, 1)
            self.final_bn = nn.BatchNorm2d(final_conv_out_channels)
            self.final_relu = nn.ReLU(inplace=True)
            planes = final_conv_out_channels

        if parts_model:
            self.local_conv_list = nn.ModuleList()
            for _ in range(num_stripes):
                self.local_conv_list.append(nn.Sequential(
                    nn.Conv2d(planes, local_conv_out_channels, 1),
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))
        else:
            self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
            self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
            self.local_relu = nn.ReLU(inplace=True)

        # list of heads for different input data sets
        if parts_model:
            self.parts_fc_list = nn.ModuleList()
            for i, num_class in enumerate(num_classes):
                fc_list = nn.ModuleList()
                for _ in range(num_stripes):
                    fc = nn.Linear(local_conv_out_channels, num_class)
                    init.normal_(fc.weight, std=0.001)
                    init.constant_(fc.bias, 0)
                    fc_list.append(fc)
                self.parts_fc_list.append(fc_list)
        elif num_classes is not None:
            self.fc_list = nn.ModuleList()
            for i, num_class in enumerate(num_classes):
                fc = nn.Linear(planes, num_class)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)

    def forward_parts(self, x, head_id):
        feat = self.base(x)
        #assert feat.size(2) % self.num_stripes == 0

        local_feat_list = []
        logits_list = []
        stripe_s = float(feat.size(2)) / self.num_stripes
        #stripe_h = int(np.ceil(stripe_s))
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            stripe_start = int(round(stripe_s*i))
            stripe_end = int(min(np.ceil(stripe_s*(i+1)), feat.size(2)))
            sh = stripe_end - stripe_start
            local_feat = F.avg_pool2d(
            feat[:, :, stripe_start: stripe_end, :], (sh, feat.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'parts_fc_list'):
                logits_list.append(self.parts_fc_list[head_id][i](local_feat))

        if hasattr(self, 'parts_fc_list'):
            #logits = torch.zeros((feat.size(0), self.num_classes))
            for i, logit_rows in enumerate(logits_list):
                if i==0:
                    logits = logit_rows
                else:
                    logits += logit_rows
            return torch.cat(local_feat_list,dim=1), None, logits
        else:
            return torch.cat(local_feat_list, dim=1), None, None

    def forward(self, x, head_id=0):
        """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
        # shape [N, C, H, W]
        if self.parts_model:
            return self.forward_parts(x, head_id)
        if self.with_final_conv:
            feat = self.final_relu(self.final_bn(self.final_conv((self.base(x)))))
        else:
            feat = self.base(x)
        #height_trucated_feat = feat[:,:,0:6,:]
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(global_feat.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        # switch head of fc layer for id loss
        if hasattr(self, 'fc_list'):
            logits = self.fc_list[head_id](global_feat)
            return global_feat, local_feat, logits

        return global_feat, local_feat, None

    def forward_roi(self, x, rois, pooled_height=8, pooled_width=4):
        feat = self.base(x)
        batch_size, num_channels, data_height, data_width = feat.size()
        num_rois = rois.size()[0]
        assert (num_rois == batch_size)
        outputs = Variable(torch.zeros(num_rois, num_channels, pooled_height, pooled_width)).cuda()

        for roi_ind, roi in enumerate(rois):
            batch_ind = roi_ind  # int(roi[0].data[0])
            roi_start_w, roi_start_h, roi_end_w, roi_end_h = np.round(
                roi[:].data.cpu().numpy()).astype(int)
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(pooled_width)
            bin_size_h = float(roi_height) / float(pooled_height)

            for ph in range(pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    is_empty = (hend <= hstart) or (wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        data = feat[batch_ind]
                        outputs[roi_ind, :, ph, pw] = torch.max(
                            torch.max(data[:, hstart:hend, wstart:wend], 1)[0], 2)[0].view(-1)

        return outputs

class PoseReIDModel(nn.Module):
    def __init__(
            self,
            last_conv_stride=1,
            last_conv_dilation=1,
            num_parts=5,
            local_conv_out_channels=256,
            num_classes=0,
            pose_ids = None
    ):
        super(PoseReIDModel, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.num_parts = num_parts
        self.planes = 2048

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_parts):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        if num_classes > 0:
            # fc for softmax loss
            self.fc_list = nn.ModuleList()
            for _ in range(num_parts):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal(fc.weight, std=0.001)
                init.constant(fc.bias, 0)
                self.fc_list.append(fc)
            # fc on full body
            fc = nn.Linear(self.planes, num_classes)
            init.normal(fc.weight, std=0.001)
            init.constant(fc.bias, 0)
            self.fc_list.append(fc)

        self.merge_layer = nn.Sequential(
                nn.Conv2d(self.planes+local_conv_out_channels*num_parts, self.planes, 1),
                nn.BatchNorm2d(self.planes),
                nn.ReLU(inplace=True))

        self.pose_ids = pose_ids

    def pool_region(self, x, normalized_boxes):
        """
        :param x: NCHW feature
        :param boxes: normalized boxes [left, top, w, h] within [0, 1]
        :return: pooled feature
        """
        feature_shape = x.size()
        roi_masks, roi_areas = self.compute_roi_masks(normalized_boxes, feature_shape)
        masked_feat = x * Variable(roi_masks)
        feature_sum = torch.sum(torch.sum(masked_feat, 3),2)
        roi_areas = roi_areas.unsqueeze(1)
        feature_average = feature_sum/Variable(roi_areas.expand((feature_shape[0], feature_shape[1])))
        condensed_feat = torch.squeeze(feature_average)
        return condensed_feat

    def compute_roi_masks(self, normalized_boxes, feat_size):
        n = feat_size[0]
        if torch.has_cudnn:
            roi_masks = torch.zeros(feat_size).cuda(device=normalized_boxes.get_device())
            roi_areas = torch.zeros(n).cuda(device=normalized_boxes.get_device())
        else:
            roi_masks = torch.zeros(feat_size)
            roi_areas = torch.zeros(n)

        for i in range(n):
            xs, xe, ys, ye = self.compute_roi(feat_size[3], feat_size[2], normalized_boxes[i])
            if xs+xe+xs+ye > 0:
                roi_masks[i, :, ys:ye, xs:xe] = 1
                roi_areas[i] = (ye-ys)*(xe-xs)
        return roi_masks, roi_areas

    def compute_roi(self, box, f_w, f_h):
        # x y of boxes
        if box[2] == 0 or box[3] == 0:
            return 0, 0, 0, 0
        else:
            xs = max(0, (f_w*box[0]).round())
            xe = min(f_w-1, (f_w*(box[0]+box[2])).round())
            ys = max(0, (f_h*box[1]).round())
            ye = min(f_h-1, (f_h*(box[1]+box[3])).round())
        return xs, xe, ys, ye

    def forward(self, x, poses):
        """
        :param x: input batch of images N 3 H W
        :param poses: input batch of N 17 4
        :return:
        """
        # shape [N, C, H, W]
        feature_map = self.base(x)
        normalized_boxes = torch.zeros((x.size()[0], 4)).cuda(x.get_device())
        part_feat_list = []
        for pose_id in self.pose_ids:
            normalized_boxes[:, 0] = torch.squeeze(poses[:, pose_id, 0]) - 0.25
            normalized_boxes[:, 1] = torch.squeeze(poses[:, pose_id, 1]) - 0.25
            normalized_boxes[:, 2] = 0.5
            normalized_boxes[:, 3] = 0.5
            normalized_boxes[normalized_boxes<0] = 0
            normalized_boxes[normalized_boxes>0.75] = 0.75
            normalized_boxes = normalized_boxes * torch.squeeze(poses[:,pose_id, 2])
            part_feature = self.pool_region(feature_map, normalized_boxes)
            part_feat_list.append(part_feature)

        global_feat = F.max_pool2d(feature_map, feature_map.size()[2:])
        part_feat_list.append(global_feat)
        condensed_feat = torch.cat(part_feat_list, dim=1)
        if len(condensed_feat.size()) == 1:  # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        final_feature = self.merge_layer(condensed_feat)
        feat = F.normalize(final_feature, p=2, dim=1)
        return feat


class SEModel(nn.Module):
    def __init__(self,
                 num_classes=None, base_model='resnet50'):
        super(SEModel, self).__init__()
        if base_model == 'resnet50':
            self.base = se_resnet.se_resnet50(pretrained=True)
            planes = 2048
        else:
            raise RuntimeError("unknown base model!")

        self.num_classes = num_classes
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
    Returns:
      global_feat: shape [N, C]
      logits: shape [N, nc]
    """
        #x: shape [N, C, H, W]
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(global_feat.size(0), -1)

        if len(global_feat.size()) == 1:  # in case of single feature
            global_feat = global_feat.unsqueeze(0)
        feat = F.normalize(global_feat, p=2, dim=1)

        if hasattr(self, 'fc'):
            logits = self.fc(feat)
            return feat,logits

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


class PCBModel(nn.Module):
  def __init__(
      self,
      last_conv_stride=1,
      last_conv_dilation=1,
      num_stripes=6,
      local_conv_out_channels=256,
      num_classes=0
  ):
    super(PCBModel, self).__init__()

    self.base = resnet50(
      pretrained=True,
      last_conv_stride=last_conv_stride,
      last_conv_dilation=last_conv_dilation)
    self.num_stripes = num_stripes

    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes):
      self.local_conv_list.append(nn.Sequential(
        nn.Conv2d(2048, local_conv_out_channels, 1),
        nn.BatchNorm2d(local_conv_out_channels),
        nn.ReLU(inplace=True)
      ))

    if num_classes > 0:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes):
        fc = nn.Linear(local_conv_out_channels, num_classes)
        init.normal(fc.weight, std=0.001)
        init.constant(fc.bias, 0)
        self.fc_list.append(fc)

  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    assert feat.size(2) % self.num_stripes == 0
    stripe_h = int(feat.size(2) / self.num_stripes)
    local_feat_list = []
    logits_list = []
    for i in range(self.num_stripes):
      # shape [N, C, 1, 1]
      local_feat = F.avg_pool2d(
        feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
        (stripe_h, feat.size(-1)))
      # shape [N, c, 1, 1]
      local_feat = self.local_conv_list[i](local_feat)
      # shape [N, c]
      local_feat = local_feat.view(local_feat.size(0), -1)
      local_feat_list.append(local_feat)
      if hasattr(self, 'fc_list'):
        logits_list.append(self.fc_list[i](local_feat))

    logits = None
    for i, logit_rows in enumerate(logits_list):
        if i == 0:
            logits = logit_rows
        else:
            logits += logit_rows
    condensed_feat = torch.cat(local_feat_list, dim=1)
    if len(condensed_feat.size()) == 1:  # in case of single feature
        condensed_feat = condensed_feat.unsqueeze(0)
    feat = F.normalize(condensed_feat, p=2, dim=1)
    return feat, logits


class MGNModel(nn.Module):
    def __init__(self,
                 num_classes=2, base_model='resnet50', local_conv_out_channels=128, parts_model=False):
        super(MGNModel, self).__init__()
        if base_model == 'resnet50':
            self.base = resnet50_with_layers(pretrained=True)
            planes = 2048
        elif base_model == 'resnet50se':
            self.base = se_resnet.se_resnet50_with_layers(pretrained=True)
            planes = 2048
        else:
            raise RuntimeError("unknown base model!")

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
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)
        fc = nn.Linear(local_conv_out_channels, num_classes)
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
            stripe_end = int(min(np.ceil(stripe_s2 * (i + 1)), feat_l2.size(2)))
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
            stripe_end = int(min(np.ceil(stripe_s3 * (i + 1)), feat_l3.size(2)))
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

#
# class WeightedReIDSeqFeatureModel(nn.Module):
#     def __init__(self, local_conv_out_channels=255, base_model='resnet18', device_id=-1):
#         super(WeightedReIDSeqFeatureModel, self).__init__()
#         if base_model == 'resnet50':
#           self.base = resnet50(pretrained=True, device=device_id)
#           planes = 2048
#         elif  base_model == 'resnet34':
#           self.base = resnet34(pretrained=True, device=device_id)
#           planes = 512
#         elif base_model == 'resnet18':
#           self.base = resnet18(pretrained=True,device=device_id)
#           planes = 512
#         elif base_model == 'inception_v3':
#           self.base = inception_v3(pretrained=True)
#           planes = 1024  # not correct
#         elif base_model == 'squeezenet':
#           self.base = squeezenet1_0(pretrained=True)
#           planes = 1000  # not correct
#         elif base_model == 'vgg16':
#           vgg = vgg16_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         elif base_model == 'vgg11':
#           vgg = vgg11_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         else:
#           raise RuntimeError("unknown base model!")
#         if device_id >= 0:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1).cuda(device_id)
#             self.quality_weight_fc = nn.Linear(planes, 1).cuda(device_id)
#         else:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
#             self.quality_weight_fc = nn.Linear(planes, 1)
#
#     def forward(self, x):
#         base_conv = self.base(x)
#         pre_feat = torch.squeeze(F.avg_pool2d(base_conv, base_conv.size()[2:])) # pre_feat for quality weight
#         final_conv_feat = self.final_conv(base_conv)
#         condensed_feat = torch.squeeze(F.avg_pool2d(final_conv_feat,final_conv_feat.size()[2:])) # descriptor were fist conv into shorter channels and then average
#         if len(condensed_feat.size()) == 1: # in case of single feature
#             condensed_feat = condensed_feat.unsqueeze(0)
#             pre_feat = pre_feat.unsqueeze(0)
#
#         feat = F.normalize(condensed_feat, p=2, dim=1)
#         quality_weight_fc = 1/(1+torch.exp(-4*self.quality_weight_fc(pre_feat)))  # quality measure of the feature
#         condensed_feat_with_quality = torch.cat((feat, quality_weight_fc),1)
#         return condensed_feat_with_quality
#
#
# class WeightedReIDFeatureROIModel(nn.Module):
#     def __init__(self, local_conv_out_channels=256, base_model='resnet18', device_id=-1):
#         super(WeightedReIDFeatureROIModel, self).__init__()
#         if base_model == 'resnet50':
#           self.base = resnet50(pretrained=True, device=device_id)
#           planes = 2048
#         elif  base_model == 'resnet34':
#           self.base = resnet34(pretrained=True, device=device_id)
#           planes = 512
#         elif base_model == 'resnet18':
#           self.base = resnet18(pretrained=True,device=device_id)
#           planes = 512
#         elif base_model == 'inception_v3':
#           self.base = inception_v3(pretrained=True)
#           planes = 1024  # not correct
#         elif base_model == 'squeezenet':
#           self.base = squeezenet1_0(pretrained=True)
#           planes = 1000  # not correct
#         elif base_model == 'vgg16':
#           vgg = vgg16_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         elif base_model == 'vgg11':
#           vgg = vgg11_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         else:
#           raise RuntimeError("unknown base model!")
#         if device_id >= 0:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1).cuda(device_id)
#             self.final_bn = nn.BatchNorm2d(local_conv_out_channels).cuda(device_id)
#             self.final_relu = nn.ReLU(inplace=True).cuda(device_id)
#         else:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
#             self.final_bn = nn.BatchNorm2d(local_conv_out_channels)
#             self.final_relu = nn.ReLU(inplace=True)
#
#     def compute_roi(self, map_w, map_h, w_h_ratio):
#         map_ratio = float(map_w)/map_h
#         if type(w_h_ratio) is float:
#             rv = w_h_ratio
#         else:
#             rv = w_h_ratio.data.cpu().numpy()
#         if rv < map_ratio:   # thin box
#             delta_w = min(map_w/2-1,int(round((map_w - map_h * w_h_ratio)/2))) # smaller than w/2, otherwise zero width in the roi
#             xs, xe = delta_w, map_w-delta_w
#             ys, ye = 0, map_h
#         else:   # fat box
#             delta_h = min(map_h/2-1, int(round((map_h - map_w / w_h_ratio)/2)))
#             ys, ye = delta_h, map_h-delta_h
#             xs, xe = 0, map_w
#
#         return xs, xe, ys, ye
#
#     def compute_roi_masks(self, w_h_ratios, feat_size):
#         n = feat_size[0]
#         if torch.has_cudnn:
#             roi_masks = torch.zeros(feat_size).cuda(device=w_h_ratios.get_device())
#             roi_areas = torch.zeros(n).cuda(device=w_h_ratios.get_device())
#         else:
#             roi_masks = torch.zeros(feat_size)
#             roi_areas = torch.zeros(n)
#
#         for i in range(n):
#             xs, xe, ys, ye = self.compute_roi(feat_size[3], feat_size[2], w_h_ratios[i])
#             roi_masks[i, :, ys:ye, xs:xe] = 1
#             roi_areas[i] = (ye-ys)*(xe-xs)
#         return roi_masks, roi_areas
#
#     def forward(self, x, w_h_ratios):
#         base_conv = self.base(x)
#         final_conv_feat = base_conv #self.final_relu(self.final_bn(self.final_conv(base_conv)))
#         w_h_ratios = w_h_ratios.view(-1)
#         # roi pooling, crop the feature map
#         conv_feat_size = list(final_conv_feat.size())
#         roi_masks, roi_areas = self.compute_roi_masks(w_h_ratios, conv_feat_size)
#         masked_feat = final_conv_feat*Variable(roi_masks)
#         feature_sum = torch.sum(torch.sum(masked_feat, 3),2)
#         roi_areas = roi_areas.unsqueeze(1)
#         feature_average = feature_sum/Variable(roi_areas.expand((conv_feat_size[0], conv_feat_size[1])))
#         condensed_feat = torch.squeeze(feature_average) # descriptor were fist conv into shorter channels and then average
#         if len(condensed_feat.size()) == 1: # in case of single feature
#             condensed_feat = condensed_feat.unsqueeze(0)
#         feat = F.normalize(condensed_feat, p=2, dim=1)
#         return feat
#
#
# class WeightedReIDFeatureModel(nn.Module):
#     def __init__(self, local_conv_out_channels=256, base_model='resnet18', device_id=-1, num_classes=None):
#         super(WeightedReIDFeatureModel, self).__init__()
#         if base_model == 'resnet50':
#           self.base = resnet50(pretrained=True, device=device_id)
#           planes = 2048
#         elif  base_model == 'resnet34':
#           self.base = resnet34(pretrained=True, device=device_id)
#           planes = 512
#         elif base_model == 'resnet18':
#           self.base = resnet18(pretrained=True,device=device_id)
#           planes = 512
#         elif base_model == 'inception_v3':
#           self.base = inception_v3(pretrained=True)
#           planes = 1024  # not correct
#         elif base_model == 'squeezenet':
#           self.base = squeezenet1_0(pretrained=True)
#           planes = 1000  # not correct
#         elif base_model == 'vgg16':
#           vgg = vgg16_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         elif base_model == 'vgg11':
#           vgg = vgg11_bn(pretrained=True)
#           self.base = vgg.features
#           planes = 512
#         else:
#           raise RuntimeError("unknown base model!")
#         if device_id >= 0:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1).cuda(device_id)
#         else:
#             self.final_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
#         if num_classes is not None:
#             self.fc = nn.Linear(planes, num_classes)
#             init.normal_(self.fc.weight, std=0.001)
#             init.constant_(self.fc.bias, 0)
#
#     def forward(self, x):
#         base_conv = self.base(x)
#         final_conv_feat = base_conv #self.final_conv(base_conv)
#         condensed_feat = torch.squeeze(F.avg_pool2d(final_conv_feat, final_conv_feat.size()[2:]))  # descriptor were fist conv into shorter channels and then average
#         if len(condensed_feat.size()) == 1: # in case of single feature
#             condensed_feat = condensed_feat.unsqueeze(0)
#         feat = F.normalize(condensed_feat, p=2, dim=1)
#         if hasattr(self, 'fc'):
#             logits = self.fc(feat)
#             return feat, logits
#         return feat, None