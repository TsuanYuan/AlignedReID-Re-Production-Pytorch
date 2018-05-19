"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
from DataLoader import ReIDAppearanceSet2SetDataset
import argparse
import os

from torchvision import transforms
import transforms_reid, Model
import utils
import losses
from torch.autograd import Variable


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
        g['lr'] = base_lr * factor ** (ind + 1)
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))

def main(data_folder, model_folder, sample_size, batch_size, seq_size,
         num_epochs=200, gpu_id=-1, margin=0.1, base_model='resnet18',
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, batch_factor=4):
    #scale = transforms_reid.Rescale((272, 136))
    #crop = transforms_reid.RandomCrop((256, 128))
    # transforms.RandomHorizontalFlip(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composed_transforms = transforms.Compose([transforms_reid.Rescale((272, 136)),  # not change the pixel range to [0,1.0]
                                              transforms_reid.RandomCrop((256,128)),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                             ])
    reid_dataset = ReIDAppearanceSet2SetDataset(data_folder,transform=composed_transforms, sample_size=sample_size)
    dataloader_less = torch.utils.data.DataLoader(reid_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8)
    dataloader_more = torch.utils.data.DataLoader(reid_dataset, batch_size=int(batch_size*batch_factor),
                                                  shuffle=True, num_workers=8)
    if not torch.cuda.is_available():
        gpu_id = -1

    if gpu_id>=0:
        model = Model.WeightedReIDFeatureModel(base_model=base_model).cuda(device=gpu_id)
    else:
        model = Model.WeightedReIDFeatureModel()
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    else:
        print('model folder {0} already exist, will overwrite it.'.format(model_folder))
    model_file = os.path.join(model_folder, 'model.ckpt')
    print('model path is {0}'.format(model_file))

    decay_at_epochs = {int(num_epochs/3):1, int(num_epochs/3*2):2}
    staircase_decay_multiply_factor = 0.1
    loss_function = losses.WeightedAverageLoss(seq_size=seq_size, margin=margin)

    optimizer = init_optim(optimizer_name, model.parameters(), lr=base_lr, weight_decay=weight_decay)
    average_meter = utils.AverageMeter()
    pdist = torch.nn.PairwiseDistance(p=2)
    dataloader = dataloader_less
    start_loss = 0

    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            # stair case adjust learning rate
            if i_batch ==0:
                adjust_lr_staircase(optimizer, base_lr, epoch + 1, decay_at_epochs, staircase_decay_multiply_factor)
            # load batch data
            images_5d = sample_batched['images']  # [batch_id, crop_id, 3, 256, 128]
            person_ids = sample_batched['person_id']
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,256,128])  # unfolder to 4-D
            optimizer.zero_grad()
            if gpu_id >= 0:
                outputs = model(Variable(images.cuda(device=gpu_id)))
                person_ids = person_ids.cuda(device=gpu_id)
            else:
                outputs = model(Variable(images))
            outputs = outputs.view([actual_size[0], sample_size, -1])
            loss, dist_pos,dist_neg = loss_function(outputs, person_ids)
            loss.backward()
            optimizer.step()
            average_meter.update(loss.data.cpu().numpy(), person_ids.cpu().size(0))
            if (i_batch+1)%20==0:
                if epoch == 0:
                    start_loss = loss.data.cpu().numpy()
                if average_meter.avg < start_loss / batch_factor/batch_factor/10:
                    dataloader = dataloader_more
                log_str = "epoch={0}, iter={1}, train_loss={2}, dist_pos={3}, dist_neg={4} avg_loss={5}"\
                    .format(str(epoch), str(i_batch), str(average_meter.val), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(average_meter.avg))
                print(log_str)
                #print('    first_feature={0}'.format(str(outputs[0,0,0:6].data.cpu().numpy())))
                #print('    last_feature={0}'.format(str(outputs[-1,-1,0:6].data.cpu().numpy())))
                pd = pdist(outputs[0,0,:-1].squeeze().unsqueeze(0), outputs[-1,-1,:-1].squeeze().unsqueeze(0))
                print('    distance between first and last crop={0}'.format(str(pd.data.cpu().numpy())))
                if (epoch+1) %(num_epochs/4)==0:
                    torch.save(model, model_file+'.epoch_{0}'.format(str(epoch)))
                torch.save(model, model_file)
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('data_folder', type=str, help="dataset original folder with subfolders of person id crops")
    parser.add_argument('model_folder', type=str, help="folder to save the model")
    parser.add_argument('--sample_size', type=int, default=32, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=8, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--seq_size', type=int, default=4, help="num images in a sequence, will folder sample_size by seq_size")
    parser.add_argument('--gpu_id', type=int, default=0, help="gpu id to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--batch_factor', type=float, default=1.5, help="increase batch size by this factor")
    parser.add_argument('--base_model', type=str, default='resnet18', help="base backbone model")
    parser.add_argument('--optimizer', type=str, default='adam', help="optimizer to use")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    print('training_parameters:')
    print('  data_folder={0}'.format(args.data_folder))
    print('  sample_size={0}, batch_size={1}, seq_size={2}, margin={3}'.
          format(str(args.sample_size), str(args.batch_size), str(args.seq_size), str(args.margin)))
    torch.backends.cudnn.benchmark = False
    main(args.data_folder, args.model_folder, args.sample_size, args.batch_size, args.seq_size,
         gpu_id=args.gpu_id, margin=args.margin, num_epochs= args.num_epoch, base_model=args.base_model,
         optimizer_name=args.optimizer, base_lr=args.lr, batch_factor=args.batch_factor)