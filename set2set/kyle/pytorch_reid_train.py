from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from aligned_reid.utils.utils import load_ckpt
#from torchvision import datasets
import sys
sys.path.append('aligned_reid/model//')
from aligned_reid.model.Model  import Model
#from aligned_reid.model.Model  import SwitchClassHeadModel

import reid_transforms
import kyle_folder

import tensorboardX
import os
from tqdm import tqdm


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_list', type=str,
                        help='path to training data')
    parser.add_argument('log-dir', type=str,
                        help='tensorboard log directory')
    parser.add_argument('--val_list', default=os.path.expanduser('~/imagenet/validation'),
                        help='path to validation data')

    parser.add_argument('--checkpoint-format', default='./customer_classification_checkpoint-{epoch}.pth.tar',
                        help='checkpoint file format')

    parser.add_argument('--pretrain_model', default='/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/exp/0728_parts_360/ckpt.pth.ep_360.ckpt',
                        help='checkpoint file')


    # Default settings from https://arxiv.org/abs/1706.02677.
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

def train(epoches, resume=False, verbose=False):

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break


    # Horovod: write TensorBoard logs on first worker.
    log_writer = tensorboardX.SummaryWriter(args.log_dir) if hvd.rank() == 0 else None


    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_dataset = \
        kyle_folder.AibeeDatasetPartsFolder(args.train_list,
                             transform= reid_transforms.ReidTransform())
    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    val_dataset = \
        kyle_folder.AibeeDatasetPartsFolder(args.val_list,
                             transform= reid_transforms.ReidTransform())
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)


    # Set up standard ResNet-50 model.
    #model = [Model(num_classes = 1442)]
    model = [Model(num_classes = 607)]
    load_ckpt(model, args.pretrain_model)
    model = model[0]

    if args.cuda:
        # Move model to GPU.
        model.cuda()

    # Horovod: scale learning rate by the number of GPUs.
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr * hvd.size(),
                          momentum=args.momentum, weight_decay=args.wd)

    # # Horovod: wrap optimizer with DistributedOptimizer.
    # optimizer = hvd.DistributedOptimizer(
    #     optimizer, named_parameters=model.named_parameters())

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def train(epoch):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(train_loader),
                  desc='Train Epoch     #{}'.format(epoch + 1),
                  disable=not verbose) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                import time
                start = time.time()
                adjust_learning_rate(epoch, batch_idx)

                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                end = time.time()

                train_loss.update(loss)
                train_accuracy.update(accuracy(output, target))
                if hvd.rank() == 0:
                    print('inference time is ' + str(end - start))
                    print('loss: ' + str(train_loss.avg.item()))
                    print('acc: ' + str(100. * train_accuracy.avg.item()))
                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': 100. * train_accuracy.avg.item()})
                t.update(1)

        if log_writer:
            log_writer.add_scalar('train/loss', train_loss.avg, epoch)
            log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(model, epoch):
    model.eval()
    return
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)


# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        print('saving to ' + filepath)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)
