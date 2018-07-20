import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .resnet import resnet50, resnet18, resnet34
from torchvision.models import inception_v3
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16_bn, vgg11_bn


class Model(nn.Module):
    def __init__(self, local_conv_out_channels=128, final_conv_out_channels=512,
                 num_classes=None, base_model='resnet50', with_final_conv=False, parts_model=False, num_stripes=4):
        super(Model, self).__init__()
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

        if parts_model:
            self.fc_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)
        elif num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal_(self.fc.weight, std=0.001)
            init.constant_(self.fc.bias, 0)

    def forward_parts(self, x):
        feat = self.base(x)
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
            feat[:, :, i * stripe_h: (i + 1) * stripe_h, :], (stripe_h, feat.size(-1)))
            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat)
            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        if hasattr(self, 'fc_list'):
            #logits = torch.zeros((feat.size(0), self.num_classes))
            for i, logit_rows in enumerate(logits_list):
                if i==0:
                    logits = logit_rows
                else:
                    logits += logit_rows
            return torch.cat(local_feat_list,dim=1), None, logits
        else:
            return torch.cat(local_feat_list, dim=1), None, None

    def forward(self, x):
        """
    Returns:
      global_feat: shape [N, C]
      local_feat: shape [N, H, c]
    """
        # shape [N, C, H, W]
        if self.parts_model:
            return self.forward_parts(x)

        if self.with_final_conv:
            feat = self.final_relu(self.final_bn(self.final_conv((self.base(x)))))
        else:
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


class SwitchClassHeadModel(nn.Module):
    def __init__(self, local_conv_out_channels=128, final_conv_out_channels=512,
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
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]
            local_feat = F.avg_pool2d(
            feat[:, :, i * stripe_h: (i + 1) * stripe_h, :], (stripe_h, feat.size(-1)))
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

    def forward(self, x, head_id):
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
