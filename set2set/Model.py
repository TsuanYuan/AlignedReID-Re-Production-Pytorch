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


def create_model(model_type, num_classes=None, num_stripes=None):
    if model_type == 'mgn':
        model = MGNModel(num_classes=num_classes)
    elif model_type == 'se':
        model = MGNModel(num_classes=num_classes, base_model='resnet50se')
    elif model_type == 'plain':
        model = PlainModel(num_classes=num_classes)
    elif model_type == 'pose_reid':
        model = PoseReIDModel(num_classes=num_classes)
    elif model_type == 'pose_reweight_reid':
        model = PoseReWeightModel(num_classes=num_classes)
    elif model_type == 'pcb':
        if num_stripes is None:
            model = PCBModel(num_classes=num_classes)
        else:
            model = PCBModel(num_classes=num_classes, num_stripes=num_stripes)
    else:
        raise Exception('unknown model type {}'.format(model_type))

    return model


def stop_gradient_on_module(m):
    for p in m.parameters():
        p.requires_grad = False


class PlainModel(nn.Module):
    def __init__(self,
                 num_classes=None, base_model='resnet50', ):
        super(PlainModel, self).__init__()
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

        self.num_classes = num_classes
        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

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

        if hasattr(self, 'fc'):
            logits = self.fc(global_feat)
            return global_feat, logits

        return global_feat, None


class SwitchClassHeadModel(nn.Module):
    def __init__(self, local_conv_out_channels=128, final_conv_out_channels=512, num_classification_head = (1,),
                 num_classes=None, base_model='resnet50', with_final_conv=False, parts_model=False, num_stripes=4):
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
        self.num_classification_head = len(num_classification_head)
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
            return torch.cat(local_feat_list,dim=1), logits
        else:
            return torch.cat(local_feat_list, dim=1), None

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
        #local_feat = torch.mean(feat, -1, keepdim=True)
        #local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        #local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        # switch head of fc layer for id loss
        if hasattr(self, 'fc_list'):
            logits = self.fc_list[head_id](global_feat)
            return global_feat, logits

        return global_feat, None

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
            local_conv_out_channels=512,
            num_classes=None,
            pose_ids=None,
            no_global=False
    ):
        super(PoseReIDModel, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        num_parts = len(pose_ids)
        self.planes = 2048

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_parts):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        if num_classes is not None:
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
        if no_global:
            self.merge_layer = nn.Sequential(
                nn.Conv2d(local_conv_out_channels * num_parts, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True))
        else:
            self.merge_layer = nn.Sequential(
                nn.Conv2d(self.planes+local_conv_out_channels*num_parts, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True))

        self.no_global = no_global
        self.pose_ids = pose_ids
        self.pose_id_2_local_id = {p: k for k, p in enumerate(pose_ids)}

    def pool_region(self, x, normalized_boxes, local_id):
        """
        :param x: NCHW feature
        :param boxes: normalized boxes [left, top, w, h] within [0, 1]
        :return: pooled feature
        """
        feature_shape = x.size()
        roi_masks, roi_areas = self.compute_roi_masks(normalized_boxes, feature_shape)
        masked_feat = x * Variable(roi_masks)
        max_pooled_feat = F.max_pool2d(masked_feat, masked_feat.size()[2:])
        # feature_sum = torch.sum(torch.sum(masked_feat, 3),2)
        # roi_areas = roi_areas.unsqueeze(1)
        # feature_average = feature_sum/Variable(roi_areas.expand((feature_shape[0], feature_shape[1]))+1e-8)
        condensed_feat = self.local_conv_list[local_id](max_pooled_feat)
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
            xs, xe, ys, ye = self.compute_roi(normalized_boxes[i,:], feat_size[3], feat_size[2])
            if xs+xe+xs+ye > 0:
                roi_masks[i, :, ys:ye, xs:xe] = 1
                roi_areas[i] = (ye-ys)*(xe-xs)
        return roi_masks, roi_areas

    def compute_roi(self, box, f_w, f_h):
        # x y of boxes
        if box[2] == 0 or box[3] == 0:
            return 0, 0, 0, 0
        else:
            xs = max(0, (f_w*box[0]).floor().int())
            xe = min(f_w-1, (f_w*(box[0]+box[2])).floor().int())
            ys = max(0, (f_h*box[1]).floor().int())
            ye = min(f_h-1, (f_h*(box[1]+box[3])).floor().int())
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
            normalized_boxes[:, 1] = torch.squeeze(poses[:, pose_id, 1]) - 0.25/2
            normalized_boxes[:, 2] = 0.5
            normalized_boxes[:, 3] = 0.5/2
            normalized_boxes[normalized_boxes<0] = 0
            normalized_boxes[normalized_boxes>0.75] = 0.75
            normalized_boxes[normalized_boxes[:,0] > 0.5,0] = 0.5 # x smaller than half
            # normalized_boxes[poses[:,pose_id, 2]==0,:] = 0 # harmful!
            # normalized_boxes = normalized_boxes * torch.squeeze(poses[:,pose_id, 2])
            local_id = self.pose_id_2_local_id[pose_id] # local id are [0,1,2,3,4], pose id are [2,9,10, 15,16]
            part_feature = self.pool_region(feature_map, normalized_boxes, local_id)
            part_feat_list.append(part_feature)

        if not self.no_global:
            global_feat = F.max_pool2d(feature_map, feature_map.size()[2:])
            part_feat_list.append(global_feat)
        ## todo: add fc list forward pass
        concat_feat = torch.cat(part_feat_list, dim=1)
        final_feature = torch.squeeze(self.merge_layer(concat_feat))
        if len(final_feature.size()) == 1:  # in case of single feature
            final_feature = final_feature.unsqueeze(0)

        feat = F.normalize(final_feature, p=2, dim=1)
        return feat


class PoseReWeightModel(nn.Module):
    def __init__(
            self,
            last_conv_stride=1,
            last_conv_dilation=1,
            local_conv_out_channels=256,
            num_classes=None,
            pose_ids=None,
    ):
        super(PoseReWeightModel, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        num_parts = len(pose_ids)
        self.planes = 2048

        self.local_conv_list = nn.ModuleList()
        for _ in range(num_parts):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.num_classes = num_classes
        if num_classes is not None:
            # fc on full body
            self.fc = nn.Linear(self.planes, num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0)

        self.merge_layer = nn.Linear(self.planes*2, self.planes)
        init.normal(self.merge_layer.weight, std=0.001)
        init.constant(self.merge_layer.bias, 0)
            # nn.Sequential(
            # nn.Conv2d(self.planes*2, self.planes, 1),
            # nn.BatchNorm2d(self.planes),
            # nn.ReLU(inplace=True))

        self.pose_ids = pose_ids
        self.pose_id_2_local_id = {p: k for k, p in enumerate(pose_ids)}

    def compute_roi(self, box, f_w, f_h):
        # x y of boxes
        if box[2] == 0 or box[3] == 0:
            return 0, 0, 0, 0
        else:
            xs = max(0, (f_w*box[0]).floor().int())
            xe = min(f_w-1, (f_w*(box[0]+box[2])).floor().int())
            ys = max(0, (f_h*box[1]).floor().int())
            ye = min(f_h-1, (f_h*(box[1]+box[3])).floor().int())
        return xs, xe, ys, ye

    def compute_roi_masks(self, normalized_boxes, feat_size):
        n = feat_size[0]
        if torch.has_cudnn:
            roi_mask = torch.ones(feat_size[1:]).cuda(device=normalized_boxes.get_device())

        else:
            roi_mask = torch.ones(feat_size[1:])

        for i in range(n):
            xs, xe, ys, ye = self.compute_roi(normalized_boxes[i,:], feat_size[3], feat_size[2])
            if xs+xe+xs+ye > 0:
                roi_mask[:, ys:ye, xs:xe] *= normalized_boxes[i,4]

        return roi_mask

    def forward(self, x, poses):
        """
        :param x: input batch of images N 3 H W
        :param poses: input batch of N 17 4
        :return:
        """
        # shape [N, C, H, W]
        feature_map = self.base(x)
        normalized_boxes = torch.zeros((x.size()[0], 5)).cuda(x.get_device())
        for pose_id in self.pose_ids:
            normalized_boxes[:, 0] = torch.squeeze(poses[:, pose_id, 0]) - 0.25
            normalized_boxes[:, 1] = torch.squeeze(poses[:, pose_id, 1]) - 0.25/2
            normalized_boxes[:, 2] = 0.5
            normalized_boxes[:, 3] = 0.5/2
            normalized_boxes[:,:4][normalized_boxes[:, :4]<0] = 0
            normalized_boxes[:,:4][normalized_boxes[:, :4]>0.75] = 0.75
            normalized_boxes[normalized_boxes[:,0] > 0.5,0] = 0.5 # x smaller than half
            normalized_boxes[:, 4] = 3/(1+2*torch.exp(10*(0.025-poses[:, pose_id, 3])))

        roi_mask = self.compute_roi_masks(normalized_boxes, feature_map.size())
        weighted_feature_map = feature_map*roi_mask
        concat_feature = torch.cat([weighted_feature_map, feature_map], dim=1)
        pool_feature = F.avg_pool2d(concat_feature, concat_feature.size()[2:])
        final_feature = self.merge_layer(torch.squeeze(pool_feature))
        if len(final_feature.size()) == 1:  # in case of single feature
            final_feature = final_feature.unsqueeze(0)
        ## todo: add number of softmax fc path

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
      num_classes=None
  ):
    super(PCBModel, self).__init__()

    self.base = resnet50(
      pretrained=True,
      last_conv_stride=last_conv_stride,
      last_conv_dilation=last_conv_dilation)
    self.num_stripes = num_stripes
    self.num_classes = num_classes

    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes):
      self.local_conv_list.append(nn.Sequential(
        nn.Conv2d(2048, local_conv_out_channels, 1),
        nn.BatchNorm2d(local_conv_out_channels),
        nn.ReLU(inplace=True)
      ))

    if num_classes is not None:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes):
        fc = nn.Linear(local_conv_out_channels, num_classes)
        init.normal_(fc.weight, std=0.001)
        init.constant_(fc.bias, 0)
        self.fc_list.append(fc)

  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
      logits_list: each member with shape [N, num_classes]
    """
    # shape [N, C, H, W]
    feat = self.base(x)
    local_feat_list = []
    logits_list = []
    stripe_h = float(feat.size(2)) / self.num_stripes
    # stripe_h = int(np.ceil(stripe_s))
    for i in range(self.num_strips):
        # shape [N, C, 1, 1]
        stripe_start = int(round(stripe_h * i))
        stripe_end = int(min(np.ceil(stripe_h * (i + 1)), feat.size(2)))
        sh = stripe_end - stripe_start
        local_feat = F.max_pool2d(
            feat[:, :, stripe_start: stripe_end, :], (sh, feat.size(-1)))
        # shape [N, c, 1, 1]
        local_feat = self.level2_conv_list[i](local_feat)
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
                 num_classes=None, base_model='resnet50', local_conv_out_channels=128):
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
        if num_classes is not None:
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
        self.num_classes = num_classes

    def compute_stripe_feature_list(self, x):
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
            if hasattr(self, 'fc_list'):
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
                logits_list.append(self.parts_fc_list[i + self.level2_strips](local_feat))
        if len(global_feat.size()) == 1:
            global_feat = global_feat.unsqueeze(0)
        local_feat_list.append(global_feat)
        if self.num_classes is not None:
            logits_list.append(self.fc_list[-1](torch.squeeze(global_feat)))
            # sum up logits and concatinate features
            logits = None
            for i, logit_rows in enumerate(logits_list):
                if i == 0:
                    logits = logit_rows
                else:
                    logits += logit_rows
        else:
            logits = None
        return local_feat_list, logits

    def concat_stripe_features(self, x):
        local_feat_list, logits = self.compute_stripe_feature_list(x)
        condensed_feat = torch.cat(local_feat_list, dim=1)
        return condensed_feat, logits

    def forward(self, x):
        """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
        condensed_feat, logits = self.concat_stripe_features(x)
        if len(condensed_feat.size()) == 1: # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
        feat = F.normalize(condensed_feat, p=2, dim=1)
        return feat, logits


class MGNWithHead(MGNModel):
    def __init__(self, pose_id = 0, attention_weight = False,
                 num_classes=None, base_model='resnet50', local_conv_out_channels=128):
        super(MGNWithHead, self).__init__(num_classes=num_classes, base_model=base_model,
                                          local_conv_out_channels=local_conv_out_channels)
        self.head_base = resnet34(pretrained=True)
        head_feature_len = 512
        mgn_feature_len = 1408
        output_feature_len = mgn_feature_len
        self.merge_layer = nn.Linear(head_feature_len+mgn_feature_len, output_feature_len)
        #init.normal_(self.merge_layer.weight, std=0.001)
        self.merge_layer.weight = torch.nn.Parameter(torch.cat((torch.zeros((output_feature_len, head_feature_len)), torch.eye(output_feature_len)), dim=1))
        init.constant_(self.merge_layer.bias, 0)
        self.pose_id = pose_id
        self.attention_weight = attention_weight
        if attention_weight:
            self.attention_weight_layer = nn.Linear(head_feature_len, 1)
            init.normal_(self.attention_weight_layer.weight, mean=0.0, std=0.00001)
            init.constant_(self.attention_weight_layer.bias, 0)
        stop_gradient_on_module(self.base) # no updates on mgn base part


    def get_head_crops(self, x, points):
        x_size = x.size()
        normalized_boxes = torch.zeros((x_size[0], 4)).cuda(x.get_device())
        normalized_boxes[:, 0] = torch.squeeze(points[:, 0]) - 0.25
        normalized_boxes[:, 1] = torch.squeeze(points[:, 1]) - 0.25 / 2
        normalized_boxes[:, 2] = 0.5
        normalized_boxes[:, 3] = 0.5/2
        normalized_boxes[normalized_boxes < 0] = 0
        normalized_boxes[normalized_boxes > 0.75] = 0.75
        normalized_boxes[normalized_boxes[:, 0] > 0.5, 0] = 0.5  # x smaller than half
        head_crops = torch.zeros((x_size[0], x_size[1], x_size[2]/4, x_size[3]/2)).cuda(x.get_device())
        for i in range(x_size[0]):
            head_crops[i, :,:,:] = x[i, :, torch.round(normalized_boxes[i, 1]*x_size[2]).int():torch.round((normalized_boxes[i, 1]+normalized_boxes[i, 3])*x_size[2]).int(),
                              torch.round(normalized_boxes[i, 0] * x_size[3]).int():torch.round((normalized_boxes[i, 0] + normalized_boxes[i, 2]) * x_size[3]).int()]
        #import debug_tool
        #debug_tool.dump_images(head_crops, '/tmp/head_crops/')
        #print 'dumped head crops to /tmp/head_crops/'
        return head_crops

    def forward(self, x, pose_points=None):
        head_points = pose_points[:, self.pose_id, :]
        head_crops = self.get_head_crops(x, head_points)
        head_base_feature = self.head_base(head_crops)
        head_feat = torch.squeeze(F.max_pool2d(head_base_feature, head_base_feature.size()[2:]))
        if self.attention_weight:
            head_weights = torch.clamp(self.attention_weight_layer(head_feat), min=0.0, max=10.0)
            head_feat = head_feat*head_weights
        if len(head_feat.size()) == 1: # in case of single feature
            head_feat = head_feat.unsqueeze(0)
        mgn_feat, _ = self.concat_stripe_features(x)
        combined_feat = torch.cat((head_feat, mgn_feat), dim=1)
        merged_feat = self.merge_layer(combined_feat)
        if len(merged_feat.size()) == 1: # in case of single feature
            merged_feat = merged_feat.unsqueeze(0)
        feat = F.normalize(merged_feat, p=2, dim=1)
        return feat



class MGNWithParts(MGNModel):
    def __init__(self, pose_ids, attention_weight = False,
                 num_classes=None, base_model='resnet50', local_conv_out_channels=128):
        super(MGNWithParts, self).__init__(num_classes=num_classes, base_model=base_model,
                                          local_conv_out_channels=local_conv_out_channels)
        self.parts_base = resnet34(pretrained=True) # model to extract part feature and part weights
        self.pose_ids = pose_ids
        num_parts = len(pose_ids)
        parts_feature_len = 512
        mgn_feature_len = 1408
        output_feature_len = mgn_feature_len
        self.merge_layer = nn.Linear(num_parts*parts_feature_len+mgn_feature_len, output_feature_len)
        self.merge_layer.weight = torch.nn.Parameter(torch.cat((torch.zeros((output_feature_len, num_parts*parts_feature_len)), torch.eye(output_feature_len)), dim=1))
        init.constant_(self.merge_layer.bias, 0)

        self.attention_weight = attention_weight
        if attention_weight:
            self.attention_weight_layers = nn.ModuleList()
            for _ in range(num_parts):
                attention_weight_layer = nn.Linear(parts_feature_len, 1)
                init.normal_(attention_weight_layer.weight, mean=0.0, std=0.00001)
                init.constant_(attention_weight_layer.bias, 0)
                self.attention_weight_layers.append(attention_weight_layer)
        stop_gradient_on_module(self.base) # no updates on mgn base part

    def pool_region(self, x, normalized_boxes, local_id):
        """
        :param x: NCHW feature
        :param normalized_boxes: normalized boxes [left, top, w, h] within [0, 1]
        :return: pooled feature
        """
        feature_shape = x.size()
        roi_masks, roi_areas = self.compute_roi_masks(normalized_boxes, feature_shape)
        masked_feat = x * Variable(roi_masks)
        max_pooled_feat = F.max_pool2d(masked_feat, masked_feat.size()[2:])
        # feature_sum = torch.sum(torch.sum(masked_feat, 3),2)
        # roi_areas = roi_areas.unsqueeze(1)
        # feature_average = feature_sum/Variable(roi_areas.expand((feature_shape[0], feature_shape[1]))+1e-8)
        condensed_feat = torch.squeeze(max_pooled_feat) #self.local_conv_list[local_id](max_pooled_feat)
        if len(condensed_feat.size()) == 1:  # in case of single feature
            condensed_feat = condensed_feat.unsqueeze(0)
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
            xs, xe, ys, ye = self.compute_roi(normalized_boxes[i,:], feat_size[3], feat_size[2])
            if xs+xe+xs+ye > 0:
                roi_masks[i, :, ys:ye, xs:xe] = 1
                roi_areas[i] = (ye-ys)*(xe-xs)
        return roi_masks, roi_areas

    def compute_roi(self, box, f_w, f_h):
        # x y of boxes
        if box[2] == 0 or box[3] == 0:
            return 0, 0, 0, 0
        else:
            xs = max(0, (f_w*box[0]).floor().int())
            xe = min(f_w-1, (f_w*(box[0]+box[2])).floor().int())
            ys = max(0, (f_h*box[1]).floor().int())
            ye = min(f_h-1, (f_h*(box[1]+box[3])).floor().int())
        return xs, xe, ys, ye

    def forward(self, x, pose_points=None):
        # shape [N, C, H, W]
        # parts feature with weights from parts network
        part_feature_map = self.parts_base(x)
        normalized_boxes = torch.zeros((x.size()[0], 4)).cuda(x.get_device())
        part_feat_list = []
        for moduel_id, pose_id in enumerate(self.pose_ids):
            normalized_boxes[:, 0] = torch.squeeze(pose_points[:, pose_id, 0]) - 0.25
            normalized_boxes[:, 1] = torch.squeeze(pose_points[:, pose_id, 1]) - 0.25 / 2
            normalized_boxes[:, 2] = 0.5
            normalized_boxes[:, 3] = 0.5 / 2
            normalized_boxes[normalized_boxes < 0] = 0
            normalized_boxes[normalized_boxes > 0.75] = 0.75
            normalized_boxes[normalized_boxes[:, 0] > 0.5, 0] = 0.5  # x smaller than half
              # local id are [0,1,2,3,4], pose id are [2,9,10, 15,16]
            part_feature = self.pool_region(part_feature_map, normalized_boxes, moduel_id)
            if self.attention_weight:
                part_feature_weight = torch.clamp(self.attention_weight_layers[moduel_id](part_feature), min=0.0, max=10.0)
            else:
                part_feature_weight = 1
            part_feat_list.append(part_feature*part_feature_weight)
        # mgn feature by mgn network
        mgn_feature, _ = self.concat_stripe_features(x)
        part_feat_list.append(mgn_feature)
        ## todo: add fc list forward pass
        concat_feat = torch.cat(part_feat_list, dim=1)
        final_feature = torch.squeeze(self.merge_layer(concat_feat))
        if len(final_feature.size()) == 1:  # in case of single feature
            final_feature = final_feature.unsqueeze(0)

        feat = F.normalize(final_feature, p=2, dim=1)
        return feat