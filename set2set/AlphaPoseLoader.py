# load alpha pose model
# Quan Yuan
# 2018-10-25

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "./AlphaPoseFork/SPPE/src/models"))
from FastPose import createModel


class AlphaPoseLoader(nn.Module):
    def __init__(self, gpu_id):
        super(AlphaPoseLoader, self).__init__()
        model = createModel().cuda()
        print('Loading pose model from {}'.format('./models/sppe/duc_se.pth'))
        model.load_state_dict(torch.load('./models/sppe/duc_se.pth',map_location={'cuda:0': 'cuda:{}'.format(str(gpu_id))}))
        model.eval()
        self.pyranet = model

    def forward(self, x):
        out = self.pyranet(x)
        return out

