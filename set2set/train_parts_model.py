"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
from DataLoader import ReIDKeypointsDataset
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


def main(index_file, model_file, sample_size, batch_size, parts_type='head', attention_weight=False,
         num_epochs=200, gpu_ids=None, margin=0.1, loss_name='ranking',
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, num_data_workers=8, skip_merge=False):

    composed_transforms = transforms.Compose([#transforms_reid.RandomHorizontalFlip(),
                                              transforms_reid.Rescale((256, 128)),  # not change the pixel range to [0,1.0]
                                              #transforms_reid.RandomCrop((256, 128)),
                                              #transforms_reid.RandomBlockMask(8),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])
    data_folders = []
    with open(index_file) as f:
        for line in f:
            data_folders.append(line.strip())
    reid_datasets = []
    for data_folder in data_folders:
        if os.path.isdir(data_folder):
            reid_dataset = ReIDKeypointsDataset(data_folder,transform=composed_transforms,
                                                crops_per_id=sample_size)
            reid_datasets.append(reid_dataset)
            num_classes = len(reid_dataset)
            print "A total of {} classes are in the data set".format(str(num_classes))
        else:
            print 'cannot find data folder {}'.format(data_folder)

    if not torch.cuda.is_available():
        gpu_ids = None
    else:
        torch.cuda.set_device(gpu_ids[0])


    if parts_type=='limbs':
        pose_ids = (2,9,10,15,16)
        model = Model.PoseReIDModel(pose_ids=pose_ids)
    elif parts_type=='limbs_only':
        pose_ids = (2,9,10,15,16)
        model = Model.PoseReIDModel(pose_ids=pose_ids, no_global=True)
        print "limbs only model!"
    elif parts_type=='head_reweight':
        pose_ids = (0, 2 ,4)
        model = Model.PoseReWeightModel(pose_ids=pose_ids)
        print "head pose reweight model!"
    elif parts_type=='head':
        pose_ids = (0, 2, 4) # redundency for heads
        model = Model.PoseReIDModel(pose_ids=pose_ids)
    elif parts_type == 'head_extra':
        pose_id = 0
        model = Model.MGNWithHead(pose_id=pose_id, attention_weight=attention_weight)
    elif parts_type == 'limbs_extra':
        pose_ids = (2,9,10,15,16)
        model = Model.MGNWithParts(pose_ids=pose_ids, attention_weight=attention_weight)
    elif parts_type=='head_only':
        pose_ids = (2,)
        model = Model.PoseReIDModel(pose_ids=pose_ids, no_global=True)
        print "head only model!"
    elif parts_type=='mgnsa':
        model = Model.MGNSelfAtten(num_classes=None)
        print "mgn self attention model!"
    else:
        raise Exception("unknown parts definition {}".format(parts_type))
    print "parts type is {}".format(parts_type)

    if len(gpu_ids)>=0:
        model = model.cuda(device=gpu_ids[0])
    optimizer = init_optim(optimizer_name, model.parameters(), lr=base_lr, weight_decay=weight_decay)
    model_folder = os.path.split(model_file)[0]
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    print('model path is {0}'.format(model_file))
    if os.path.isfile(model_file):
        if args.resume:
            Model.load_ckpt([model], model_file, skip_fc=True, skip_merge=skip_merge)
        else:
            print('model file {0} already exist, will overwrite it.'.format(model_file))

    if len(gpu_ids) > 0:
        model_p = DataParallel(model, device_ids=gpu_ids)
    else:
        model_p = model

    start_decay = 50
    min_lr = 1e-9
    if loss_name == 'triplet':
        loss_function = losses.TripletLossK(margin=margin)
    elif loss_name == 'pair':
        loss_function = losses.PairLoss(margin=margin)
    else:
        raise Exception('unknown loss name')

    n_set = len(reid_datasets)
    dataloaders = [torch.utils.data.DataLoader(reid_datasets[set_id], batch_size=batch_size, drop_last = True,
                                             shuffle=True, num_workers=num_data_workers) for set_id in range(n_set)]
    dataloader_iterators = [iter(dataloaders[i]) for i in range(n_set)]
    num_iters_per_epoch = sum([len(dataloaders[i]) for i in range(n_set)])
    for epoch in range(num_epochs):
        sum_loss = 0
        i_batch = 0
        while i_batch < num_iters_per_epoch: #i_batch, sample_batched in enumerate(dataloader):
            set_id = i_batch % n_set
            it = dataloader_iterators[set_id]
            try:
                sample_batched = next(it)
            except:
                dataloader_iterators[set_id] = iter(dataloaders[set_id])
                sample_batched = next(dataloader_iterators[set_id])
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
            keypoints_5d = sample_batched['keypoints']
            person_ids = sample_batched['person_id']
            # import debug_tool
            # debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/')
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,256,128])  # unfolder to 4-D
            kp_size = keypoints_5d.size()
            keypoints = keypoints_5d.view([kp_size[0]*kp_size[1],kp_size[2],kp_size[3]])
            if len(gpu_ids)>0:
                with torch.cuda.device(gpu_ids[0]):
                    person_ids = person_ids.cuda(device=gpu_ids[0])
                    features = model_p(Variable(images.cuda(device=gpu_ids[0], async=True)), keypoints.cuda(device=gpu_ids[0])) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features = model(Variable(images), keypoints)
            if len(features) > 1:  # in case with logits outputs
                features = features[0]
            outputs = features.view([actual_size[0], sample_size, -1])
            loss,dist_pos, dist_neg,p_pids,n_pids = loss_function(outputs, person_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss+=loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            #if parts_type=='head' and dist_pos.data.cpu().numpy() > dist_neg.data.cpu().numpy():
            #    print("  a touch case. pos_pids are {}, neg_pids are {}".format(str(numpy.unique(p_pids.cpu().numpy())), str(numpy.unique(n_pids.cpu().numpy()))))
            #    pids_expand = person_ids.expand(actual_size[0:2]).contiguous().view(-1)
            #    import debug_tool
            #    debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/', pids=pids_expand, name_tag=str(epoch)+'_'+str(i_batch)+'_')
            if i_batch==num_iters_per_epoch-1:
                log_str = "{}: epoch={}, iter={}, train_loss={}, dist_pos={}, dist_neg={} sum_loss_epoch={}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(sum_loss))
                print(log_str)
                #if dist_pos.data.cpu().numpy() > dist_neg.data.cpu().numpy():
                #    print("  a touch case. pos_pids are {}, neg_pids are {}".format(str(numpy.unique(p_pids.cpu().numpy())), str(numpy.unique(n_pids.cpu().numpy()))))
                if (epoch+1) %(max(1,min(25, num_epochs/8)))==0:
                    save_ckpt([model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                save_ckpt([model],  epoch, log_str, model_file)
            i_batch += 1
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training with parts model")

    parser.add_argument('folder_list_file', type=str, help="index of training folders, each folder contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")

    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--num_data_workers', type=int, default=8, help="num of works in multiprocess data batching")
    parser.add_argument('--batch_factor', type=float, default=1.5, help="increase batch size by this factor")
    parser.add_argument('--model_type', type=str, default='mgn', help="model_type")
    parser.add_argument('--optimizer', type=str, default='sgd', help="optimizer to use")
    parser.add_argument('--loss', type=str, default='triplet', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--class_th', type=float, default=0.2, help="class threshold")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--skip_merge', action='store_true', default=False, help="whether to skip loading the merge layers that combines all parts")
    parser.add_argument('--parts_type', type=str, default='head', help="parts definitions")
    parser.add_argument('--attention_weight', action='store_true', default=False, help="whether to use attention weight for parts")

    args = parser.parse_args()
    print('training_parameters:')
    print('  index_file={0}'.format(args.folder_list_file))
    print('  sample_size={}, batch_size={},  margin={}, loss={}, optimizer={}, lr={}, skip_merge={}, num_data_workers={}'.
          format(str(args.sample_size), str(args.batch_size), str(args.margin), str(args.loss), str(args.optimizer),
                   str(args.lr), str(args.skip_merge), str(args.num_data_workers)))

    torch.backends.cudnn.benchmark = False

    main(args.folder_list_file, args.model_file, args.sample_size, args.batch_size, parts_type=args.parts_type,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, num_data_workers=args.num_data_workers,
         optimizer_name=args.optimizer, base_lr=args.lr, loss_name=args.loss, skip_merge=args.skip_merge, attention_weight=args.attention_weight)
