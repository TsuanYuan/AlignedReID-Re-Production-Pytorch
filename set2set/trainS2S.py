"""
train appearance model with quality weight
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
from DataLoader import ReIDAppearanceSet2SetDataset
import argparse

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

def main(data_folder):
    #scale = transforms_reid.Rescale((272, 136))
    #crop = transforms_reid.RandomCrop((256, 128))
    # transforms.RandomHorizontalFlip(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    composed_transforms = transforms.Compose([transforms_reid.Rescale((272, 136)), # also change the pixel range to [0,1.0]
                                              transforms_reid.RandomCrop((256,128)),
                                              transforms_reid.ToTensor(),
                                             ])

    sample_size = 8
    reid_dataset = ReIDAppearanceSet2SetDataset(data_folder,transform=composed_transforms, sample_size=sample_size)
    batch_size = 4
    dataloader = torch.utils.data.DataLoader(reid_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    use_gpu = True
    loss_function = losses.WeightedAverageLoss()
    model = Model.WeightedReIDFeatureModel()

    optimizer = init_optim('adam', model.parameters(), lr=0.001, weight_decay=5e-04)
    average_meter = utils.AverageMeter()
    for i_batch, sample_batched in enumerate(dataloader):
        images = sample_batched['images']
        person_ids = sample_batched['person_id']
        images = images.view([batch_size*sample_size,3,256,128])
        outputs = model(Variable(images))
        outputs = outputs.view([batch_size, sample_size, -1])
        loss = loss_function(outputs, person_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_meter.update(loss.item(), person_ids.size(0))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('data_folder', type=str, help="dataset original folder with subfolders of person id crops")
    args = parser.parse_args()
    main(args.data_folder)