"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
from DataLoader import ReIDAppearanceDataset, ReIDSameIDOneDayDataset, ReIDSingleFileCropsDataset, ReIDHeadAppearanceDataset
import argparse
import os
import datetime

from torchvision import transforms
import transforms_reid, Model
import losses
from torch.autograd import Variable
from torch.nn.parallel import DataParallel


def main(index_file, model_file, sample_size, batch_size, model_type='mgn', desired_size=(256, 128),
         num_epochs=200, gpu_ids=None, margin=0.1, softmax_loss_weight=0.01, num_data_workers=4,
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, head_train=False):

    if head_train:
        assert desired_size[0] == desired_size[1]
        rescale_ext = int(desired_size[0]*0.1)
        composed_transforms = transforms.Compose([
                                                  transforms_reid.Rescale((desired_size[0]+rescale_ext, desired_size[1]+rescale_ext)),
                                                  transforms_reid.RandomCrop(desired_size),
                                                  transforms_reid.PixelNormalize(),
                                                  transforms_reid.ToTensor(),
                                                  ])
    else:
        composed_transforms = transforms.Compose([transforms_reid.RandomHorizontalFlip(),
                                              transforms_reid.Rescale(desired_size),  # not change the pixel range to [0,1.0]
                                              #transforms_reid.RandomCrop((256, 128)),
                                              transforms_reid.RandomBlockMask(8),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])
    data_folders, data_loader_names, ignore_paths = [], [], []
    with open(index_file) as f:
        for line in f:
            parts = line.strip().split()
            data_folders.append(parts[0])
            data_loader_names.append(parts[1])
            if len(parts) > 2:
                ignore_paths.append(parts[2])
            else:
                ignore_paths.append(None)

    reid_datasets = []
    softmax_loss_functions = []
    for data_path, data_path_extra, ignore_path in zip(data_folders, data_loader_names, ignore_paths):
        if os.path.isdir(data_path):
            if head_train:
                reid_dataset = ReIDHeadAppearanceDataset(data_path, transform=composed_transforms,
                                                crops_per_id=sample_size)
            elif data_path_extra.find('same_day')>=0:
                reid_dataset = ReIDSameIDOneDayDataset(data_path,transform=composed_transforms,
                                                crops_per_id=sample_size)
            elif data_path_extra.find('all')>=0:
                reid_dataset = ReIDAppearanceDataset(data_path,transform=composed_transforms,
                                                crops_per_id=sample_size)
            else:
                raise Exception('unknown data loader name {}'.format(data_path_extra))
            reid_datasets.append(reid_dataset)
            num_classes = len(reid_dataset)
            softmax_loss_function = losses.MultiClassLoss(num_classes=num_classes)
            softmax_loss_functions.append(softmax_loss_function)
            print "A total of {} classes are in the data set".format(str(num_classes))
        elif os.path.isfile(data_path): # training list in wanda or dfxtd case
            index_file = data_path
            data_folder = data_path_extra
            ignore_pid_list = None
            if ignore_path is not None:
                with open(ignore_path, 'r') as fp:
                    ignore_pid_list = [int(line.rstrip('\n')) for line in fp if len(line.rstrip('\n')) > 0]
            reid_dataset = ReIDSingleFileCropsDataset(data_folder, index_file, transform=composed_transforms, same_day_camera=False, ignore_pid_list=ignore_pid_list,
                                                sample_size=sample_size, index_format='list', desired_size=desired_size)
            reid_datasets.append(reid_dataset)
            num_classes = len(reid_dataset)
            softmax_loss_function = losses.MultiClassLoss(num_classes=num_classes)
            softmax_loss_functions.append(softmax_loss_function)
        else:
            print 'cannot find data path {}'.format(data_path)

    if not torch.cuda.is_available():
        gpu_ids = None
    if head_train:
        model = Model.PlainModel(base_model='resnet34')
        model_type = 'plain'
    elif model_type == 'mgn':
        model = Model.MGNModel()
    elif model_type == 'se':
        model = Model.MGNModel(base_model='resnet50se')
    elif model_type == 'plain':
        model = Model.PlainModel()
    else:
        raise Exception('unknown model type {}'.format(model_type))
    if len(gpu_ids)>=0:
        model = model.cuda(device=gpu_ids[0])
    else:
        torch.cuda.set_device(gpu_ids[0])

    optimizer = Model.init_optim(optimizer_name, model.parameters(), lr=base_lr, weight_decay=weight_decay)
    model_folder = os.path.split(model_file)[0]
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    print('model path is {0}'.format(model_file))
    if os.path.isfile(model_file):
        if args.resume:
            Model.load_ckpt([model], model_file, skip_fc=True)
        else:
            print('model file {0} already exist, will overwrite it.'.format(model_file))

    if len(gpu_ids) > 0:
        model_p = DataParallel(model, device_ids=gpu_ids)
    else:
        model_p = model

    start_decay = 50
    min_lr = 1e-9
    metric_loss_function = losses.TripletLossK(margin=margin)

    n_set = len(reid_datasets)
    dataloaders = [torch.utils.data.DataLoader(reid_datasets[set_id], batch_size=batch_size,
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
                Model.adjust_lr_exp(
                    optimizer,
                    base_lr,
                    epoch + 1,
                    num_epochs,
                    start_decay, min_lr)

            # load batch data
            images_5d = sample_batched['images']  # [batch_id, crop_id, 3, 256, 128]
            #import debug_tool
            #debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d_head/')
            person_ids = sample_batched['person_id']
            # w_h_ratios = sample_batched['w_h_ratios']
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,desired_size[0],desired_size[1]])  # unfolder to 4-D

            if len(gpu_ids)>0:
                with torch.cuda.device(gpu_ids[0]):
                    person_ids = person_ids.cuda(device=gpu_ids[0])
                    features, logits = model_p(Variable(images.cuda(device=gpu_ids[0], async=True), volatile=False)) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features, logits = model(Variable(images))
            outputs = features.view([actual_size[0], sample_size, -1])
            metric_loss, dist_pos, dist_neg, _, _ = metric_loss_function(outputs, person_ids)
            if softmax_loss_weight > 0:
                raise Exception('Not implemented! need to implement multi-head mgn class first!')
                # actual_size = images_5d.size()
                # pids_expand = person_ids.expand(actual_size[0:2]).contiguous().view(-1)
                # softmax_loss = softmax_loss_functions[set_id](pids_expand.cuda(device=gpu_ids[0]), logits)
                # loss = metric_loss + softmax_loss_weight * softmax_loss
            else:
                loss = metric_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss+=loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            if i_batch==num_iters_per_epoch-1:
                log_str = "{}: epoch={}, iter={}, train_loss={}, dist_pos={}, dist_neg={} sum_loss_epoch={}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(sum_loss))
                print(log_str)
                if (epoch+1) %(max(1,min(25, num_epochs/8)))==0:
                    Model.save_ckpt([model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                Model.save_ckpt([model],  epoch, log_str, model_file)
            i_batch += 1
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="multi training set training")

    parser.add_argument('folder_list_file', type=str, help="index of training folders, each folder contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")

    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--batch_factor', type=float, default=1.5, help="increase batch size by this factor")
    parser.add_argument('--model_type', type=str, default='mgn', help="model_type")
    parser.add_argument('--optimizer', type=str, default='sgd', help="optimizer to use")
    parser.add_argument('--loss', type=str, default='triplet', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--class_th', type=float, default=0.2, help="class threshold")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--reid_same_day', action='store_false', default=True, help="whether to put same pair same day constrain on reid training")
    parser.add_argument('--softmax_loss_weight', type=float, default=0, help="weight of softmax loss in total loss")
    parser.add_argument('--num_data_workers', type=int, default=4, help="num of data batching workers")
    parser.add_argument('--head', action='store_true', default=False, help="training head model with head parameters")
    parser.add_argument('--desired_aspect', type=int, default=2, help="crop aspect ratio")
    
    args = parser.parse_args()
    print('training_parameters:')
    print('  index_file={0}'.format(args.folder_list_file))
    print('  sample_size={}, batch_size={},  margin={}, loss={}, optimizer={}, lr={}, model_type={}, reid_same_day={}, softmax_weight={}, head={}'.
          format(str(args.sample_size), str(args.batch_size), str(args.margin), str(args.loss), str(args.optimizer),
                   str(args.lr), args.model_type, str(args.reid_same_day), str(args.softmax_loss_weight), str(args.head)))

    torch.backends.cudnn.benchmark = False
    if args.head:
        desired_size = (64, 64)
    elif args.desired_aspect == 2:
        desired_size = (256, 128)
    elif args.desired_aspect == 3:
        desired_size = (384, 128)
    else:
        raise Exception('unknown aspect ratio {}'.format(str(args.desired_aspect)))

    main(args.folder_list_file, args.model_file, args.sample_size, args.batch_size, model_type=args.model_type,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, num_data_workers=args.num_data_workers, desired_size=desired_size,
         optimizer_name=args.optimizer, base_lr=args.lr, softmax_loss_weight=args.softmax_loss_weight, head_train=args.head)
