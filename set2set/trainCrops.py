"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
from DataLoader import ReIDSingleFileCropsDataset, ReIDAppearanceDataset
import argparse
import os
import datetime

from torchvision import transforms
import transforms_reid, Model
import losses
from torch.autograd import Variable



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




def main(data_folder, index_file, model_file, sample_size, batch_size, model_type='plain', num_stripes=None, same_day_camera=False,
         num_epochs=200, gpu_ids=None, margin=0.1, loss_name='ranking', ignore_pid_file=None, softmax_loss_ratio=0.2, num_data_workers=8,
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, index_format='list', desized_size=(256, 128)):

    composed_transforms = transforms.Compose([transforms_reid.RandomHorizontalFlip(),
                                              transforms_reid.Rescale(desized_size),
                                              #transforms_reid.Rescale((272, 136)),  # not change the pixel range to [0,1.0]
                                              #transforms_reid.RandomCrop((256, 128)),
                                              #transforms_reid.RandomBlockMask(8),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])

    print "index format is {}".format(index_format)

    if not torch.cuda.is_available():
        gpu_ids = None

    ignore_pid_list = None
    if ignore_pid_file is not None:
        with open(ignore_pid_file, 'r') as fp:
            ignore_pid_list= [int(line.rstrip('\n')) for line in fp if len(line.rstrip('\n'))>0]

    if len(args.index_file) == 0:
        reid_dataset = ReIDAppearanceDataset(data_folder,transform=composed_transforms,
                                                crops_per_id=sample_size)
    else:
        reid_dataset = ReIDSingleFileCropsDataset(data_folder, index_file, transform=composed_transforms, same_day_camera=same_day_camera,
                                                sample_size=sample_size, index_format=index_format, ignore_pid_list=ignore_pid_list, desired_size=desized_size)

    num_classes = len(reid_dataset)
    if num_classes == 0:
        raise Exception('0 classes are in the training set!')
    print "A total of {} classes are in the data set".format(str(num_classes))
    dataloader = torch.utils.data.DataLoader(reid_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_data_workers)
    # model and optimizer
    model_p, optimizer, single_model = Model.load_model_optimizer(model_file, optimizer_name, gpu_ids, base_lr, weight_decay, num_classes, model_type, num_stripes)
    # parameters and l,osses
    start_decay = 50
    min_lr = 1e-9
    if loss_name == 'triplet':
        metric_loss_function = losses.TripletLossK(margin=margin)
    elif loss_name == 'pair':
        metric_loss_function = losses.PairLoss(margin=margin)
    else:
        raise Exception('unknown loss name')
    softmax_loss_func = losses.MultiClassLoss(num_classes=num_classes)

    for epoch in range(num_epochs):
        sum_loss, sum_metric_loss = 0, 0
        for i_batch, sample_batched in enumerate(dataloader):
            # stair case adjust learning rate
            if i_batch ==0:
                adjust_lr_exp(
                    optimizer,
                    base_lr,
                    epoch + 1,
                    num_epochs,
                    start_decay, min_lr)

            # load batch data
            images_5d = sample_batched['images']  # [batch_id, crop_id, 3, 256, 128]
            person_ids = sample_batched['person_id']
            # # debug
            actual_size = images_5d.size()
            pids_expand = person_ids.expand(actual_size[0:2]).contiguous().view(-1)
            #import debug_tool
            #debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/', pids=pids_expand, name_tag=str(epoch)+'_'+str(i_batch)+'_')

            # w_h_ratios = sample_batched['w_h_ratios']
            # actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,desized_size[0],desized_size[1]])  # unfolder to 4-D

            if len(gpu_ids)>0:
                with torch.cuda.device(gpu_ids[0]):
                    person_ids = person_ids.cuda()
                    features, logits = model_p(Variable(images.cuda(device=gpu_ids[0], async=True), volatile=False)) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features, logits = model_p(Variable(images))
            outputs = features.view([actual_size[0], sample_size, -1])
            metric_loss,dist_pos, dist_neg, _, _ = metric_loss_function(outputs, person_ids)
            softmax_loss = softmax_loss_func(pids_expand.cuda(device=gpu_ids[0]), logits)
            loss = metric_loss+softmax_loss_ratio*softmax_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss+=loss.data.cpu().numpy()
            sum_metric_loss+=metric_loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            if i_batch==len(dataloader)-1:
                log_str = "{}: epoch={}, iter={}, train_loss={}, dist_pos={}, dist_neg={}, sum_metric_loss={}, sum_loss_epoch={}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(sum_metric_loss), str(sum_loss))
                print(log_str)
                if (epoch+1) %(max(1,min(50, num_epochs/8)))==0:
                    save_ckpt([single_model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                save_ckpt([single_model],  epoch, log_str, model_file)
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training with binary crop formats")

    parser.add_argument('index_file', type=str, help="index of binary dataset original folder")
    parser.add_argument('model_file', type=str, help="the model file")
    parser.add_argument('--data_folder', type=str, help="dataset original folder with subfolders of person id crops", default='')
    parser.add_argument('--index_format', type=str, default='list', help="format of index file")
    parser.add_argument('--ignore_pid_file', type=str, help="the file of ignored pids", default=None)
    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--num_data_workers', type=int, default=8, help="num of works in multiprocess data batching")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--batch_factor', type=float, default=1.5, help="increase batch size by this factor")
    parser.add_argument('--base_model', type=str, default='resnet50', help="base backbone model")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer to use")
    parser.add_argument('--crop_aspect_ratio', type=float, default=2.0, help="aspect ratio of crops")
    parser.add_argument('--loss', type=str, default='triplet', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--class_th', type=float, default=0.2, help="class threshold")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--softmax_loss_ratio', type=float, default=0.2, help="ratio of softmax loss in total loss")
    parser.add_argument('--model_type', type=str, default='plain', help="model_type. plain, pcb, mgn, pose_reid and etc")
    parser.add_argument('--num_stripes', type=int, default=None,
                        help="num stripes in a part based model")
    parser.add_argument('--same_day_camera', action='store_true', default=False, help="whether to train a tracking model with same day same camera")

    args = parser.parse_args()
    print('training_parameters:')
    print('  index_file={0}'.format(args.index_file))
    print('  sample_size={}, batch_size={},  margin={}, metric_loss={}, softmax_loss_ratio={}, model_type={}, crop_aspect_ratio={}'.
          format(str(args.sample_size), str(args.batch_size), str(args.margin), str(args.loss), str(args.softmax_loss_ratio), args.model_type, str(args.crop_aspect_ratio)))
    torch.backends.cudnn.benchmark = False
    if len(args.data_folder) == 0:
        args.data_folder = os.path.split(args.index_file)[0]
    if args.crop_aspect_ratio == 2.0:
        desized_size = (256, 128)
    elif args.crop_aspect_ratio == 3.0:
        desized_size = (384, 128)
    else:
        raise Exception('unknown crop aspect ratio {}'.format(str(args.crop_aspect_ratio)))

    main(args.data_folder, args.index_file, args.model_file, args.sample_size, args.batch_size,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, ignore_pid_file=args.ignore_pid_file, num_stripes=args.num_stripes,
         optimizer_name=args.optimizer, base_lr=args.lr, loss_name=args.loss, index_format=args.index_format, model_type=args.model_type,
         softmax_loss_ratio=args.softmax_loss_ratio, same_day_camera=args.same_day_camera, num_data_workers=args.num_data_workers, desized_size=desized_size)
