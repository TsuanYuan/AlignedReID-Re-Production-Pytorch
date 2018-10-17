"""
Model definitions
Quan Yuan
2018-05-11
"""


import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import se_resnet


def create_model(model_type, num_classes=None):
    if model_type == 'se':
        model = SEModel(num_classes=num_classes, base_model='resnet50se')
    else:
        raise Exception('unknown model type {}'.format(model_type))

    return model


def stop_gradient_on_module(m):
    for p in m.parameters():
        p.requires_grad = False



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
