from torch import nn
from torch.autograd import Variable
import torch

class PairLoss(object):
  """lifted loss from https://arxiv.org/pdf/1511.06452.pdf"""
  def __init__(self, margin=None):
    self.margin = margin
    if margin is not None:
      self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    else:
      raise Exception('not none for margin!')

  def __call__(self, dist, labels):
      """
          Args:
            dist: pytorch Variable, distance matrix inside a batch
            labels: pytorch Variable, identity labels
          Returns:
            loss: pytorch Variable, with shape [1]
      """
      zeros = torch.zeros(dist.size(0))
      loss = self.ranking_loss(dist, zeros, labels)
      return loss