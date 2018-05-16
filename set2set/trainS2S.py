"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
from DataLoader import ReIDAppearanceSet2SetDataset
import argparse
import os

from torchvision import transforms
import transforms_reid, Model
import utils
import losses
from torch.autograd import Variable

def init_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optim: {}".format(optim))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    model.train()
    losses = utils.AverageMeter()

    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        outputs = model(imgs)
        loss = criterion(outputs, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Epoch {}/{}\t Batch {}/{}\t Loss {:.6f} ({:.6f})".format(
                epoch+1, args.max_epoch, batch_idx+1, len(trainloader), losses.val, losses.avg
            ))

def main(data_folder, model_folder, sample_size, batch_size, seq_size, gpu_id=-1):
    #scale = transforms_reid.Rescale((272, 136))
    #crop = transforms_reid.RandomCrop((256, 128))
    # transforms.RandomHorizontalFlip(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composed_transforms = transforms.Compose([transforms_reid.Rescale((272, 136)), # also change the pixel range to [0,1.0]
                                              transforms_reid.RandomCrop((256,128)),
                                              transforms_reid.ToTensor(),
                                             ])
    reid_dataset = ReIDAppearanceSet2SetDataset(data_folder,transform=composed_transforms, sample_size=sample_size)
    dataloader = torch.utils.data.DataLoader(reid_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8)
    if torch.cuda.is_available() and gpu_id>=0:
        model = Model.WeightedReIDFeatureModel().cuda(device=gpu_id)
    else:
        model = Model.WeightedReIDFeatureModel()
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    else:
        print('model folder {0} already exist, will overwrite it.'.format(model_folder))
    model_file = os.path.join(model_folder, 'model.ckpt')
    print('model path is {0}'.format(model_file))

    loss_function = losses.WeightedAverageLoss(seq_size=seq_size)

    optimizer = init_optim('adam', model.parameters(), lr=0.001, weight_decay=5e-04)
    average_meter = utils.AverageMeter()
    num_epochs = 100
    for epoch in range(num_epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            images = sample_batched['images']
            person_ids = sample_batched['person_id']
            actual_size = list(images.size()) #
            images = images.view([actual_size[0]*sample_size,3,256,128])
            outputs = model(Variable(images.cuda(device=gpu_id)))
            outputs = outputs.view([actual_size[0], sample_size, -1])
            loss = loss_function(outputs, person_ids.cuda(device=gpu_id))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_meter.update(loss.data.cpu().numpy(), person_ids.cpu().size(0))
            if (i_batch+1)%2==0:
                log_str = "epoch={0}, iter={1}, train_margin={2}".format(str(epoch), str(i_batch), str(average_meter.val))
                print(log_str)
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
    args = parser.parse_args()
    print('training_parameters:')
    print('  data_folder={0}'.format(args.data_folder))
    print('  sample_size={0}, batch_size={1}, seq_size={2}'.format(str(args.sample_size), str(args.batch_size), str(args.seq_size)))
    torch.backends.cudnn.benchmark = False
    main(args.data_folder, args.model_folder, args.sample_size, args.batch_size, args.seq_size, gpu_id=args.gpu_id)