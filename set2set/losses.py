"""
loss functions
Quan Yuan
2018-05-15
"""

import torch
from torch import nn
from torch.nn import functional

"""
Shorthands for loss:
- SetLoss: loss with weighted average of all features
"""
__all__ = ['WeightedAverageLoss']

def batch_euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [N, m, d]
    y: pytorch Variable, with shape [N, n, d]
  Returns:
    dist: pytorch Variable, with shape [N, m, n]
  """
  assert len(x.size()) == 3
  assert len(y.size()) == 3
  assert x.size(0) == y.size(0)
  assert x.size(-1) == y.size(-1)

  N, m, d = x.size()
  N, n, d = y.size()

  # shape [N, m, n]
  xx = torch.pow(x, 2).sum(-1, keepdim=True).expand(N, m, n)
  yy = torch.pow(y, 2).sum(-1, keepdim=True).expand(N, n, m).permute(0, 2, 1)
  dist = xx + yy
  dist.baddbmm_(1, -2, x, y.permute(0, 2, 1))
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def euclidean_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def pair_loss_func(feature, pids, margin):
    N = pids.size()[0]  # number of weighted features
    is_pos = pids.expand(N, N).eq(pids.expand(N, N).t())
    diag_one = torch.diag(torch.ones_like(torch.sum(is_pos, 1)))
    is_pos = is_pos - diag_one
    is_neg = pids.expand(N, N).ne(pids.expand(N, N).t())
    dist_mat = euclidean_distances(feature)
    dist_pos = dist_mat[is_pos].contiguous()
    dist_neg = dist_mat[is_neg].contiguous()
    dist_pos = dist_pos.view(-1, 1)
    dist_neg = dist_neg.view(1, -1)
    loss_mat = dist_pos + margin - dist_neg
    loss_ids = loss_mat.gt(0).detach()
    loss = torch.sum(loss_mat[loss_ids])
    return loss, torch.mean(dist_pos), torch.mean(dist_neg)

def weighted_seq_loss_func(feature, weight, pids, seq_size, margin):

    # weighted feature
    weight_size = list(weight.size())
    feature_size = list(feature.size())
    weight_fold = weight.view(weight_size[0], -1, seq_size)
    #assert numpy.all(weight_fold.data.numpy() > 0)
    weight_size = list(weight_fold.size())

    weight_expand = weight_fold.unsqueeze(3).expand(weight_size[0], weight_size[1], weight_size[2], feature_size[2])
    feature_fold = feature.view((feature_size[0], -1, seq_size, feature_size[2]))
    # assert unit length
    # t = torch.pow(feature_fold, 2).sum(3)
    # assert torch.abs(t-1.0) < 0.01
    feature_expand_seq = feature_fold * weight_expand

    num_feature_per_id = list(feature_expand_seq.size())[1]
    summed_feature_seq = feature_expand_seq.sum(2).squeeze()  # weighted sum
    summed_feature = summed_feature_seq.view((-1, feature_size[2]))  # unfold to [n*m, 255]
    summed_feature_normalize = functional.normalize(summed_feature, p=2, dim=1)  # normalized after weighted sum
    pid_expand = pids.expand(feature_size[0], num_feature_per_id).contiguous().view(-1)  # unfold to [n*m]

    return pair_loss_func(summed_feature_normalize, pid_expand, margin)



def element_loss_func(feature, pids, margin):
    return pair_loss_func(feature, pids, margin)


class WeightedAverageLoss(nn.Module):
    """Weighted avearge loss.
    assume the last element of the feature is a weight vector
    """

    def __init__(self, margin, seq_size=4):
        super(WeightedAverageLoss, self).__init__()
        self.seq_size = seq_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.margin = margin

    def forward(self, x, pids):
        feature = x[:, :, :-1].contiguous()
        weight = x[:, :, -1].contiguous()
        # sequence reID loss
        seq_loss = weighted_seq_loss_func(feature, weight, pids, self.seq_size, self.margin)
        # element reID loss
        weight_size = list(weight.size())
        pids_expand = pids.expand(weight_size).contiguous().view(-1)
        feature_expand = feature.view(weight_size[0]*weight_size[1], -1)
        element_loss = element_loss_func(feature_expand, pids_expand, self.margin)

        return element_loss[0]+seq_loss[0], element_loss[1]+seq_loss[1], element_loss[2]+seq_loss[2]
