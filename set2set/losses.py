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
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class WeightedAverageLoss(nn.Module):
    """Weighted avearge loss.
    assume the last element of the feature is a weight vector
    """

    def __init__(self, seq_size=4):
        super(WeightedAverageLoss, self).__init__()
        self.seq_size = seq_size


    def forward(self, x, pids, margin=0.1):
        feature = x[:,:,:-1].contiguous()
        weight = x[:,:, -1].contiguous()
        # weighted feature
        weight_size = list(weight.size())
        feature_size = list(feature.size())
        weight_fold = weight.view(weight_size[0],-1, self.seq_size)
        weight_size = list(weight_fold.size())
        weight_fold = functional.softmax(weight_fold, 2)
        weight_expand = weight_fold.unsqueeze(3).expand(weight_size[0],weight_size[1],weight_size[2], feature_size[2])
        feature_fold = feature.view((feature_size[0],-1, self.seq_size, feature_size[2]))
        # assert unit length
        # t = torch.pow(feature_fold, 2).sum(3)
        # assert torch.abs(t-1.0) < 0.01
        feature_expand_seq = feature_fold * weight_expand

        num_feature_per_id = list(feature_expand_seq.size())[1]
        summed_feature_seq = feature_expand_seq.sum(2).squeeze() # weighted sum
        summed_feature = summed_feature_seq.view((-1, feature_size[2])) # unfold to [n*m, 255]
        summed_feature_normalize = functional.normalize(summed_feature, p=2, dim=1) # normalized after weighted sum
        pid_expand = pids.expand(feature_size[0], num_feature_per_id).contiguous().view(-1) # unfold to [n*m]

        N = pid_expand.size()[0]  # number of weighted features
        is_pos = pid_expand.expand(N, N).eq(pid_expand.expand(N, N).t())
        is_neg = pid_expand.expand(N, N).ne(pid_expand.expand(N, N).t())
        dist_mat = euclidean_distances(summed_feature_normalize)

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = torch.max(
            dist_mat[is_pos].contiguous())
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an = torch.min(
            dist_mat[is_neg].contiguous())
        # shape [N]
        loss = dist_an+margin-dist_ap
        return loss
