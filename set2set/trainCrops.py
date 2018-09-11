"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
from DataLoader import ReIDAppearanceSet2SetDataset, ReIDSingleFileCropsDataset
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
    for g in optimizer.param_groups:
        g['lr'] = base_lr * factor ** ind
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

def main(data_folder, model_folder, sample_size, batch_size,
         num_epochs=200, gpu_ids=None, margin=0.1, base_model='resnet18', loss_name='ranking',
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, threshold=0.1, with_roi=False):
    if with_roi:
        composed_transforms = transforms.Compose([transforms_reid.RandomHorizontalFlip(),
                                                  transforms_reid.Rescale((256, 128)),
                                                  transforms_reid.PixelNormalize(),
                                                  transforms_reid.ToTensor(),
                                                  ])  # no random crop
    else:
        composed_transforms = transforms.Compose([transforms_reid.RandomHorizontalFlip(),
                                              transforms_reid.Rescale((256, 128)),  # not change the pixel range to [0,1.0]
                                              #transforms_reid.RandomCrop((256, 128)),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])

    # reid_dataset = ReIDAppearanceSet2SetDataset(data_folder,transform=composed_transforms,
    #                                             sample_size=sample_size, with_roi=with_roi)
    reid_dataset = ReIDSingleFileCropsDataset(data_folder, transform=composed_transforms,
                                                sample_size=sample_size)
    num_classes = len(reid_dataset)
    dataloader = torch.utils.data.DataLoader(reid_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    if not torch.cuda.is_available():
        gpu_ids = None

    model = Model.MGNModel()
    if len(gpu_ids)>=0:
        model = model.cuda()
    # else:
    #     model = Model.WeightedReIDFeatureModel(base_model=base_model,num_classes=num_classes)
    optimizer = init_optim(optimizer_name, model.parameters(), lr=base_lr, weight_decay=weight_decay)

    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    else:
        print('model folder {0} already exist, will overwrite it.'.format(model_folder))
    model_file = os.path.join(model_folder, 'model.ckpt')
    print('model path is {0}'.format(model_file))
    if args.resume and os.path.isfile(model_file):
        load_ckpt([model], model_file)
    if len(gpu_ids) > 0:
        model_p = DataParallel(model, device_ids=gpu_ids)
    else:
        model_p = model
    decay_at_epochs = {50:1, 100:2, 200:3}
    staircase_decay_multiply_factor = 0.1
    if loss_name == 'ranking':
        loss_function = losses.WeightedAverageLoss(margin=margin, num_classes=num_classes)
    elif loss_name == 'class_th':
        loss_function = losses.WeightedAverageThLoss(th=threshold)
    else:
        raise Exception('unknown loss name')

    for epoch in range(num_epochs):
        sum_loss = 0
        sum_tri_loss = 0
        for i_batch, sample_batched in enumerate(dataloader):
            # stair case adjust learning rate
            if i_batch ==0:
                adjust_lr_staircase(optimizer, base_lr, epoch + 1, decay_at_epochs, staircase_decay_multiply_factor)
            # load batch data
            images_5d = sample_batched['images']  # [batch_id, crop_id, 3, 256, 128]
            # debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/')
            person_ids = sample_batched['person_id']
            # w_h_ratios = sample_batched['w_h_ratios']
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,256,128])  # unfolder to 4-D

            if len(gpu_ids)>0:
                person_ids = person_ids.cuda()
                features, logits = model_p(Variable(images.cuda())) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features, logits = model(Variable(images)) #model(Variable(images), Variable(w_h_ratios))
            outputs = features.view([actual_size[0], sample_size, -1])
            loss,tri_loss, dist_pos, dist_neg = loss_function(outputs, person_ids, logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #average_meter.update(loss.data.cpu().numpy(), person_ids.cpu().size(0))
            sum_loss+=loss.data.cpu().numpy()
            sum_tri_loss += tri_loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            if i_batch==len(dataloader)-1:
                log_str = "time_str: epoch={0}, iter={1}, train_loss={2}, dist_pos={3}, dist_neg={4} sum_loss_epoch={5}, sum_tri_loss={6}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(sum_loss), str(sum_tri_loss))
                print(log_str)
                if (epoch+1) %(max(1,num_epochs/8))==0:
                    save_ckpt([model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                save_ckpt([model],  epoch, log_str, model_file)
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('data_folder', type=str, help="dataset original folder with subfolders of person id crops")
    parser.add_argument('model_folder', type=str, help="folder to save the model")
    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--batch_factor', type=float, default=1.5, help="increase batch size by this factor")
    parser.add_argument('--base_model', type=str, default='resnet50', help="base backbone model")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer to use")
    parser.add_argument('--loss', type=str, default='ranking', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--class_th', type=float, default=0.2, help="class threshold")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--with_roi', action='store_true', default=False, help="whether to use roi")
    parser.add_argument('--original_ar', action='store_true', default=False, help="whether use original aspect ratio")

    args = parser.parse_args()
    print('training_parameters:')
    print('  data_folder={0}'.format(args.data_folder))
    print('  sample_size={0}, batch_size={1},  margin={2}, original_ar={3}, with_roi={4}'.
          format(str(args.sample_size), str(args.batch_size), str(args.margin), str(args.original_ar), str(args.with_roi)))
    torch.backends.cudnn.benchmark = False
    main(args.data_folder, args.model_folder, args.sample_size, args.batch_size,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, base_model=args.base_model,
         optimizer_name=args.optimizer, base_lr=args.lr, with_roi=args.with_roi, threshold=args.class_th)