"""
train car appearance model with multi-task loss
Quan Yuan
2018-10-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
import DataLoader
import argparse
import os
import datetime

from torchvision import transforms
import transforms_reid, Model
import losses
from torch.autograd import Variable
from torch.nn.parallel import DataParallel


def save_ckpt(modules_optims, ep, scores, ckpt_file):
  """Save state_dict's of modules/optimizers to file.
  Args:
    modules_optims: A list, which members are either torch.nn.optimizer
      or torch.nn.Module.
    ep: the current epoch number
    scores: the performance of current model
    ckpt_file: The file path.
  Note:
    torch.save() reserves device type and id of tensors to save, so when
    loading ckpt, you have to inform torch.load() to load these tensors to
    cpu or your desired gpu, if you change devices.
  """
  state_dicts = [m.state_dict() for m in modules_optims]
  ckpt = dict(state_dicts=state_dicts,
              ep=ep,
              scores=scores)
  if not os.path.isdir(os.path.dirname(os.path.abspath(ckpt_file))):
      os.makedirs(os.path.dirname(os.path.abspath(ckpt_file)))
  torch.save(ckpt, ckpt_file)

def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True, skip_fc=False):
  """Load state_dict's of modules/optimizers from file.
  Args:
    modules_optims: A list, which members are either torch.nn.optimizer
      or torch.nn.Module.
    ckpt_file: The file path.
    load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
      to cpu type.
  """
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  ckpt = torch.load(ckpt_file, map_location=map_location)
  if skip_fc:
    print('skip fc layers when loading the model!')
  for m, sd in zip(modules_optims, ckpt['state_dicts']):
    if m is not None:
      if skip_fc:
        for k in sd.keys():
          if k.find('fc') >= 0:
            sd.pop(k, None)
      if hasattr(m, 'param_groups'):
        m.load_state_dict(sd)
      else:
        m.load_state_dict(sd, strict=False)
  if verbose:
    print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
      ckpt_file, ckpt['ep'], ckpt['scores']))
  return ckpt['ep'], ckpt['scores']


def adjust_lr_staircase(optimizer, base_lr, ep, decay_at_epochs, factor):
    """Multiplied by a factor at the BEGINNING of specified epochs. All
    parameters in the optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      decay_at_epochs: a list or tuple; learning rate is multiplied by a factor
        at the BEGINNING of these epochs
      factor: a number in range (0, 1)

    Example:
      base_lr = 1e-3
      decay_at_epochs = [51, 101]
      factor = 0.1
      It means the learning rate starts at 1e-3 and is multiplied by 0.1 at the
      BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
      BEGINNING of the 101'st epoch, then stays unchanged till the end of
      training.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = decay_at_epochs[ep]
    g = None
    for g in optimizer.param_groups:
        g['lr'] = base_lr * factor ** ind
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep, min_lr):
    """Decay exponentially in the later phase of training. All parameters in the
    optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      total_ep: total number of epochs to train
      start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
      base_lr = 2e-4
      total_ep = 300
      start_decay_at_ep = 201
      It means the learning rate starts at 2e-4 and begins decaying after 200
      epochs. And training stops after 300 epochs.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return
    g = None
    for g in optimizer.param_groups:
        g['lr'] = max((base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                            / (total_ep + 1 - start_decay_at_ep)))), min_lr)
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def init_optim(optim, params, lr, weight_decay, eps=1e-8):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, eps=eps, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))


def main(data_folder, model_file, sample_size, batch_size, model_type='mgn',
         num_epochs=200, gpu_ids=(), margin=0.1, num_workers=8,
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, start_decay = 50, id_loss_weight=0.01,
         desired_size = (128,128), frame_group_interval=128):
    composed_transforms = transforms.Compose([
                                              transforms_reid.Rescale(desired_size),  # not change the pixel range to [0,1.0]
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])
    car_id_dataset = DataLoader.ReIDCarDataset(data_folder,transform=composed_transforms, frame_group_interval=frame_group_interval,
                                               crops_per_id=sample_size, desired_size=desired_size)
    num_classes = len(car_id_dataset)
    if not torch.cuda.is_available():
        gpu_ids = None
    # if model_type == 'mgn':
    #     model = Model.MGNModel()
    # el
    if model_type == 'plain':
        model = Model.PlainModel(num_classes=num_classes)
    else:
        raise Exception('unknown model type {}'.format(model_type))
    if gpu_ids is not None  and len(gpu_ids)>=0:
        model = model.cuda(device=gpu_ids[0])
    optimizer = init_optim(optimizer_name, model.parameters(), lr=base_lr, weight_decay=weight_decay)
    model_folder = os.path.split(model_file)[0]
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    print('model path is {0}'.format(model_file))
    if os.path.isfile(model_file):
        if args.resume:
            load_ckpt([model], model_file, skip_fc=True)
        else:
            print('model file {0} already exist, will overwrite it.'.format(model_file))

    if gpu_ids is not None and len(gpu_ids) > 0:
        model_p = DataParallel(model, device_ids=gpu_ids)
    else:
        model_p = model

    min_lr = 1e-9

    loss_function = losses.CarLoss(margin, id_loss_weight)

    dataloader = torch.utils.data.DataLoader(car_id_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    for epoch in range(num_epochs):
        sum_loss, metric_loss = 0, 0
        for i_batch, sample_batched in enumerate(dataloader):

            # stair case adjust learning rate
            if i_batch ==0:
                adjust_lr_exp(
                    optimizer,
                    base_lr,
                    epoch + 1,
                    num_epochs,
                    start_decay, min_lr)

            images_5d = sample_batched['images'] #torch.cat([sample['images'] for sample in sample_batched], dim=0)  # [batch_id, crop_id, 3, 256, 128]
            #import debug_tool
            #debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/')
            class_ids = sample_batched['class_id']
            class_ids = torch.cat(class_ids, dim=0).reshape(8,-1).t().reshape((-1))
            time_group_ids = sample_batched['time_group_ids']
            time_group_ids = torch.cat(time_group_ids).reshape(8,-1).t().reshape((-1))
            # dates = torch.cat([sample['date'] for sample in sample_batched], dim=0)
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*actual_size[1],3,desired_size[0],desired_size[1]])  # unfolder to 4-D
            #import debug_tool
            #debug_tool.dump_images(images, '/tmp/images/')
            if gpu_ids is not None and len(gpu_ids)>0:
                with torch.cuda.device(gpu_ids[0]):
                    class_ids = class_ids.cuda(device=gpu_ids[0])
                    time_group_ids = time_group_ids.cuda(device=gpu_ids[0])
                    features, logits = model_p(Variable(images.cuda(device=gpu_ids[0], async=True), volatile=False)) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features, logits = model(Variable(images))
            # outputs = features.view([actual_size[0], sample_size, -1])
            loss, tri_loss, id_loss, dist_pos, dist_neg, _, _ = loss_function(features, logits,class_ids, time_group_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss+=loss.data.cpu().numpy()
            metric_loss += tri_loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            if i_batch==len(dataloader)-1:
                log_str = "{}: epoch={}, iter={}, train_loss={}, dist_pos={}, dist_neg={} metric_loss = {}, sum_loss_epoch={}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(metric_loss), str(sum_loss))
                print(log_str)
                if (epoch+1) %(max(1,min(25, num_epochs/8)))==0:
                    save_ckpt([model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                save_ckpt([model],  epoch, log_str, model_file)
            i_batch += 1
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training reid with day by day Dataset. Each subfolder is of one ID")

    parser.add_argument('data_folder', type=str, help="training folder, contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")

    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.3, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--start_decay', type=int, default=50, help="epoch to start learning rate decay")
    parser.add_argument('--num_data_workers', type=int, default=4, help="num of data batching workers")
    parser.add_argument('--model_type', type=str, default='plain', help="model_type")
    parser.add_argument('--optimizer', type=str, default='sgd', help="optimizer to use")
    parser.add_argument('--loss', type=str, default='triplet', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--softmax_loss_weight', type=float, default=0.01, help="ratio of softmax loss in total loss")
    parser.add_argument('--frame_group_interval', type=int, default=128, help="interval of frame index to group frames")

    args = parser.parse_args()
    print('training_parameters:')
    print('  data_folder={0}'.format(args.data_folder))
    print('  sample_size={}, batch_size={},  margin={}, loss={}, optimizer={}, lr={}, frame_interval={}'.
          format(str(args.sample_size), str(args.batch_size), str(args.margin), str(args.loss), str(args.optimizer),
                   str(args.lr), str(args.frame_group_interval)))

    torch.backends.cudnn.benchmark = False

    main(args.data_folder, args.model_file, args.sample_size, args.batch_size, model_type=args.model_type,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, start_decay=args.start_decay,
         optimizer_name=args.optimizer, base_lr=args.lr, num_workers=args.num_data_workers, frame_group_interval=args.frame_group_interval)
