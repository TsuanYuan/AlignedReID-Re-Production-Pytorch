from torch import nn
from torch.autograd import Variable


class LiftedLoss(object):
  """lifted loss from https://arxiv.org/pdf/1511.06452.pdf"""
  def __init__(self, margin=None):
    self.margin = margin

  def __call__(self, dist, labels):
      """
          Args:
            dist: pytorch Variable, distance matrix inside a batch
            labels: pytorch Variable, identity labels
          Returns:
            loss: pytorch Variable, with shape [1]
      """
      y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))

      return loss