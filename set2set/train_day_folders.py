"""
train appearance model on days
Quan Yuan
2018-05-15
"""
import torch.utils.data, torch.optim
import torch.backends.cudnn
import DataLoader
import argparse
import datetime

from torchvision import transforms
import transforms_reid, Model
import losses
from torch.autograd import Variable


def main(data_folder, model_file, sample_size, batch_size, model_type='mgn',
         num_epochs=200, gpu_ids=None, margin=0.1, num_workers=8, softmax_loss_weight=0, resume=False,
         optimizer_name='adam', base_lr=0.001, weight_decay=5e-04, start_decay = 250, desired_size=(256, 128), num_stripes=6):

    composed_transforms = transforms.Compose([transforms_reid.RandomHorizontalFlip(),
                                              transforms_reid.Rescale((desired_size[0]+16, desired_size[1]+8)),  # not change the pixel range to [0,1.0]
                                              transforms_reid.RandomCrop(desired_size),
                                              #transforms_reid.RandomBlockMask(8),
                                              transforms_reid.PixelNormalize(),
                                              transforms_reid.ToTensor(),
                                              ])

    #reid_datasets = DataLoader.create_list_of_days_datasets(data_folder, transform=composed_transforms, crops_per_id=sample_size)
    #reid_data_concat = DataLoader.ConcatDayDataset(reid_datasets, batch_size, data_size_factor=data_size_factor)
    pid_one_day_dataset = DataLoader.ReIDSameIDOneDayDataset(data_folder, transform=composed_transforms,
                                                             crops_per_id=sample_size, desired_size=desired_size)
    num_classes = len(pid_one_day_dataset)
    if not torch.cuda.is_available():
        gpu_ids = None

    # model and optimizer
    model_p, optimizer, single_model = Model.load_model_optimizer(model_file, optimizer_name, gpu_ids, base_lr,
                                                                  weight_decay, num_classes, model_type,
                                                                  num_stripes, resume=resume)

    min_lr = 1e-9

    metric_loss_function = losses.TripletLossK(margin=margin)
    softmax_loss_func = losses.MultiClassLoss(num_classes=num_classes)

    dataloader = torch.utils.data.DataLoader(pid_one_day_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    for epoch in range(num_epochs):
        sum_loss, sum_metric_loss = 0, 0
        for i_batch, sample_batched in enumerate(dataloader):
            # stair case adjust learning rate
            if i_batch ==0:
                Model.adjust_lr_exp(
                    optimizer,
                    base_lr,
                    epoch + 1,
                    num_epochs,
                    start_decay, min_lr)

            # debug date
            # for sample in sample_batched:
            #     print "date of {}".format(sample['date'])
            images_5d = sample_batched['images'] #torch.cat([sample['images'] for sample in sample_batched], dim=0)  # [batch_id, crop_id, 3, 256, 128]
            # import debug_tool
            # debug_tool.dump_images_in_batch(images_5d, '/tmp/images_5d/')
            person_ids = sample_batched['person_id'] #torch.cat([sample['person_id'] for sample in sample_batched], dim=0)
            # dates = torch.cat([sample['date'] for sample in sample_batched], dim=0)
            actual_size = list(images_5d.size())
            images = images_5d.view([actual_size[0]*sample_size,3,desired_size[0],desired_size[1]])  # unfolder to 4-D

            if len(gpu_ids)>0:
                with torch.cuda.device(gpu_ids[0]):
                    person_ids = person_ids.cuda(device=gpu_ids[0])
                    features, logits = model_p(Variable(images.cuda(device=gpu_ids[0], async=True), volatile=False)) #, Variable(w_h_ratios.cuda(device=gpu_id)))m
            else:
                features, logits = model_p(Variable(images))
            outputs = features.view([actual_size[0], sample_size, -1])
            metric_loss,dist_pos, dist_neg, _, _ = metric_loss_function(outputs, person_ids)
            actual_size = images_5d.size()
            pids_expand = person_ids.expand(actual_size[0:2]).contiguous().view(-1)
            softmax_loss = softmax_loss_func(pids_expand.cuda(device=gpu_ids[0]), logits)
            loss = metric_loss + softmax_loss_weight * softmax_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss+=loss.data.cpu().numpy()
            sum_metric_loss+= metric_loss.data.cpu().numpy()
            time_str = datetime.datetime.now().ctime()
            if i_batch==len(dataloader)-1:
                log_str = "{}: epoch={}, iter={}, train_loss={}, dist_pos={}, dist_neg={}, sum_metric_loss={}, sum_loss_epoch={}"\
                    .format(time_str, str(epoch), str(i_batch), str(loss.data.cpu().numpy()), str(dist_pos.data.cpu().numpy()),
                            str(dist_neg.data.cpu().numpy()), str(sum_metric_loss), str(sum_loss))
                print(log_str)
                if (epoch+1) %(max(1,min(25, num_epochs/8)))==0:
                    Model.save_ckpt([single_model], epoch, log_str, model_file+'.epoch_{0}'.format(str(epoch)))
                Model.save_ckpt([single_model],  epoch, log_str, model_file)
            i_batch += 1
    print('model saved to {0}'.format(model_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training reid with day by day Dataset. Each subfolder is of one ID")

    parser.add_argument('data_folder', type=str, help="training folder, contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")

    parser.add_argument('--sample_size', type=int, default=8, help="total number of images of each ID in a sample")
    parser.add_argument('--batch_size', type=int, default=32, help="num samples in a mini-batch, each sample is a sequence of images")
    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")
    parser.add_argument('--margin', type=float, default=0.1, help="margin for the loss")
    parser.add_argument('--num_epoch', type=int, default=200, help="num of epochs")
    parser.add_argument('--start_decay', type=int, default=50, help="epoch to start learning rate decay")
    parser.add_argument('--num_workers', type=int, default=4, help="num of data batching workers")
    parser.add_argument('--model_type', type=str, default='mgnc', help="model_type")
    parser.add_argument('--optimizer', type=str, default='sgd', help="optimizer to use")
    parser.add_argument('--loss', type=str, default='triplet', help="loss to use")
    parser.add_argument('--lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('--desired_aspect', type=int, default=2, help="crop aspect ratio")
    parser.add_argument('--class_th', type=float, default=0.2, help="class threshold")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume from existing ckpt")
    parser.add_argument('--softmax_loss_weight', type=float, default=0, help="weight of softmax loss in total loss")

    args = parser.parse_args()
    print('training_parameters:')
    print('  data_folder={0}'.format(args.data_folder))
    print('  sample_size={}, batch_size={},  modeltype={}, loss={}, optimizer={}, lr={}, desired_aspect={}, softmax_loss_weight={}'.
          format(str(args.sample_size), str(args.batch_size), str(args.model_type), str(args.loss), str(args.optimizer),
                   str(args.lr), str(args.desired_aspect), str(args.softmax_loss_weight)))

    torch.backends.cudnn.benchmark = False
    if args.desired_aspect == 2:
        desired_size = (256, 128)
    elif args.desired_aspect == 3:
        desired_size = (384, 128)
    else:
        raise Exception('unknown aspect ratio {}'.format(str(args.desired_aspect)))
    main(args.data_folder, args.model_file, args.sample_size, args.batch_size, model_type=args.model_type,
         num_epochs=args.num_epoch, gpu_ids=args.gpu_ids, margin=args.margin, start_decay=args.start_decay,
         optimizer_name=args.optimizer, base_lr=args.lr, num_workers=args.num_workers,
         desired_size=desired_size, softmax_loss_weight=args.softmax_loss_weight, resume = args.resume)
