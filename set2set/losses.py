"""
loss functions
Quan Yuan
2018-05-15
"""

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable
import torch.nn.functional as F

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


def pair_loss_func(feature, pids, margin, sub_sample_neg=10):
    N = pids.size()[0]  # number of weighted features
    is_pos = pids.expand(N, N).eq(pids.expand(N, N).t())
    diag_one = torch.diag(torch.ones_like(torch.sum(is_pos, 1).byte()))
    is_pos = is_pos - diag_one
    is_neg = pids.expand(N, N).ne(pids.expand(N, N).t())
    dist_mat = euclidean_distances(feature)
    dist_pos = dist_mat[is_pos].contiguous()
    dist_neg = dist_mat[is_neg].contiguous()[0::sub_sample_neg].contiguous()  # subsample neg pairs
    dist_pos = dist_pos.view(-1, 1)
    dist_neg = dist_neg.view(1, -1)
    loss_mat = dist_pos + margin - dist_neg
    loss_ids = loss_mat.gt(0).detach()
    loss = torch.sum(loss_mat[loss_ids])
    return loss, torch.max(dist_pos), torch.min(dist_neg)

def normalize(x, axis=-1):
  """Normalizing to unit length along the specified dimension.
  Args:
    x: pytorch Variable
  Returns:
    x: pytorch Variable, same shape as input
  """
  x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
  return x


def euclidean_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
  Returns:
    dist: pytorch Variable, with shape [m, n]
  """
  m, n = x.size(0), y.size(0)
  xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
  yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
  dist = xx + yy
  dist.addmm_(1, -2, x, y.t())
  dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
  return dist


def global_loss(tri_loss, global_feat, labels, normalize_feature=True):
  """
  Args:
    tri_loss: a `TripletLoss` object
    global_feat: pytorch Variable, shape [N, C]
    labels: pytorch LongTensor, with shape [N]
    normalize_feature: whether to normalize feature to unit length along the
      Channel dimension
  Returns:
    loss: pytorch Variable, with shape [1]
    p_inds: pytorch LongTensor, with shape [N];
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    =============
    For Debugging
    =============
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    ===================
    For Mutual Learning
    ===================
    dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
  """
  if normalize_feature:
    global_feat = normalize(global_feat, axis=-1)
  # shape [N, N]
  dist_mat = euclidean_dist(global_feat, global_feat)
  dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
    dist_mat, labels, return_inds=True)
  #dist_np, labels_np = pair_example_mining(dist_mat, labels)

  loss = tri_loss(dist_ap, dist_an)#+pair_loss(dist_np, labels_np)
  return loss, torch.mean(dist_ap), torch.mean(dist_an)
  # return loss, p_inds, n_inds, dist_ap, dist_an, dist_mat

def triplet_loss_func(feature, labels, ranking_loss, margin=0.2):
    #self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    dist_mat = euclidean_distances(feature)
    dist_ap, dist_an, p_inds, n_inds = hard_example_mining(
        dist_mat, labels, return_inds=True)
    loss_row = dist_ap + margin - dist_an
    #loss_ids = loss_row.gt(0).detach()
    #loss = torch.sum(loss_row[loss_ids])
    loss = torch.sum(loss_row[loss_row > 0])
    #y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    #loss = ranking_loss(dist_an, dist_ap, y)
    return loss, torch.mean(dist_ap), torch.mean(dist_an)

def fixed_th_loss_func(feature, pids, th, margin_pos, margin_neg):
    # threshold loss
    N = pids.size()[0]  # number of weighted features
    is_pos = pids.expand(N, N).eq(pids.expand(N, N).t())
    diag_one = torch.diag(torch.ones_like(torch.sum(is_pos, 1).byte()))
    is_pos = is_pos - diag_one
    is_neg = pids.expand(N, N).ne(pids.expand(N, N).t())
    dist_mat = euclidean_distances(feature)
    dist_pos = dist_mat[is_pos].contiguous()
    dist_neg = dist_mat[is_neg].contiguous()
    dist_pos = dist_pos.view(-1, 1)
    dist_neg = dist_neg.view(1, -1)
    loss_pos_ids = dist_pos.gt(th-margin_pos).detach()
    loss_neg_ids = dist_neg.lt(th+margin_neg).detach()
    loss_pos = torch.sum(dist_pos[loss_pos_ids] - th + margin_pos)
    loss_neg = torch.sum(th + margin_neg - dist_neg[loss_neg_ids])
    return loss_pos+loss_neg, torch.max(dist_pos), torch.min(dist_neg)


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
    dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    labels: pytorch LongTensor, with shape [N]
    return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
    dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
    dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    p_inds: pytorch LongTensor, with shape [N];
      indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
    n_inds: pytorch LongTensor, with shape [N];
      indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
    thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
    dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
    dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze( 0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
          ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
          ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def weighted_seq_loss_func(feature, weight, pids, seq_size, margin, th=-1.0):

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
    feature_expand_seq = feature_fold * weight_expand  # multiply in 5D

    num_feature_per_id = list(feature_expand_seq.size())[1]
    summed_feature_seq = feature_expand_seq.sum(2).squeeze()  # weighted sum
    summed_feature = summed_feature_seq.view((-1, feature_size[2]))  # unfold to [n*m, 255]
    summed_feature_normalize = functional.normalize(summed_feature, p=2, dim=1)  # normalized after weighted sum
    pid_expand = pids.expand(feature_size[0], num_feature_per_id).contiguous().view(-1)  # unfold to [n*m]
    if th < 0:
        return pair_loss_func(summed_feature_normalize, pid_expand, margin)
    else:
        return fixed_th_loss_func(summed_feature_normalize, pid_expand, th, th/2, th)

def element_loss_func(feature, pids, margin, ranking_loss, th=-1.0):
    if th<0:
        return triplet_loss_func(feature, pids, ranking_loss, margin=margin)#pair_loss_func(feature, pids, margin)
    else:
        return fixed_th_loss_func(feature, pids, th, th/2, th)



class TripletLoss(object):
  """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
  Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
  Loss for Person Re-Identification'."""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      self.ranking_loss = nn.SoftMarginLoss()

  def __call__(self, dist_ap, dist_an):
    """
    Args:
      dist_ap: pytorch Variable, distance between anchor and positive sample,
        shape [N]
      dist_an: pytorch Variable, distance between anchor and negative sample,
        shape [N]
    Returns:
      loss: pytorch Variable, with shape [1]
    """
    y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
    if self.margin is not None:
      loss = self.ranking_loss(dist_an, dist_ap, y)
    else:
      loss = self.ranking_loss(dist_an - dist_ap, y)
    return loss


class WeightedAverageSeqLoss(nn.Module):
    """Weighted avearge loss.
    assume the last element of the feature is a weight vector
    """

    def __init__(self, margin, seq_size=4):
        super(WeightedAverageSeqLoss, self).__init__()
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
        element_loss = element_loss_func(feature_expand, pids_expand, self.margin, self.ranking_loss)

        return element_loss[0]+seq_loss[0], element_loss[1], element_loss[2]


class WeightedAverageSeqThLoss(nn.Module):
    """Weighted avearge loss with a threshold, all pos < th-pos_margin, all neg > th+pos_margin
    assume the last element of the feature is a weight vector
    """

    def __init__(self, seq_size=4, th=0.1):
        super(WeightedAverageSeqThLoss, self).__init__()
        self.seq_size = seq_size
        self.th = th

    def forward(self, x, pids):
        feature = x[:, :, :-1].contiguous()
        weight = x[:, :, -1].contiguous()
        # sequence reID loss
        seq_loss = weighted_seq_loss_func(feature, weight, pids, self.seq_size, self.th, self.th) # margin with th
        # element reID loss
        weight_size = list(weight.size())
        pids_expand = pids.expand(weight_size).contiguous().view(-1)
        feature_expand = feature.view(weight_size[0]*weight_size[1], -1)
        element_loss = element_loss_func(feature_expand, pids_expand, self.th, None, th=self.th)

        return element_loss[0]+seq_loss[0], element_loss[1], element_loss[2]

class GlobalLoss(nn.Module):
    """Weighted avearge loss.
       assume the last element of the feature is a weight vector
       """

    def __init__(self, margin):
        super(GlobalLoss, self).__init__()
        self.margin = margin
        self.triple_loss = TripletLoss(margin=margin)

    def forward(self, feature, pids, logits):
        feature_size = list(feature.size())
        pids_expand = pids.expand(feature_size[0:2]).contiguous().view(-1)
        feature_expand = feature.view(feature_size[0] * feature_size[1], -1)
        element_loss, max_same_d, min_diff_d = global_loss(self.triple_loss, feature_expand, pids_expand)
        # mc_loss = self.id_loss(pids_expand, logits)
        return element_loss, max_same_d, min_diff_d


class WeightedAverageLoss(nn.Module):
    """Weighted avearge loss.
    assume the last element of the feature is a weight vector
    """

    def __init__(self, margin, num_classes):
        super(WeightedAverageLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.id_loss = MultiClassLoss(num_classes=num_classes)
        self.triple_loss = TripletLoss()

    def forward(self, feature, pids, logits):
        feature_size = list(feature.size())
        pids_expand = pids.expand(feature_size[0:2]).contiguous().view(-1)
        feature_expand = feature.view(feature_size[0]*feature_size[1], -1)
        element_loss, max_same_d, min_diff_d = element_loss_func(feature_expand, pids_expand, self.margin, self.ranking_loss)
        #mc_loss = self.id_loss(pids_expand, logits)
        return element_loss, element_loss, max_same_d, min_diff_d


class MultiClassLoss(nn.Module):
    def __init__(self, num_classes):
        super(MultiClassLoss, self).__init__()
        self.num_classes = num_classes
        self.id_criterion = nn.CrossEntropyLoss()
    def forward(self, pids, logits):
        id_loss = self.id_criterion(logits, pids)

        return id_loss


class WeightedAverageThLoss(nn.Module):
    """Weighted avearge loss with a threshold, all pos < th-pos_margin, all neg > th+pos_margin
    assume the last element of the feature is a weight vector
    """

    def __init__(self, th=0.1):
        super(WeightedAverageThLoss, self).__init__()
        self.th = th

    def forward(self, x, pids):
        feature = x
        feature_size = list(feature.size())
        pids_expand = pids.expand(feature_size).contiguous().view(-1)
        feature_expand = feature.view(feature_size[0]*feature_size[1], -1)
        element_loss = element_loss_func(feature_expand, pids_expand, self.th, None, th=self.th)

        return element_loss[0], element_loss[1], element_loss[2]