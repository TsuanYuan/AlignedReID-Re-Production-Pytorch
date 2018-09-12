
import cv2
import os, glob
import argparse, logging
import numpy
import time
import torch
from torch.autograd import Variable
from load_model import AppearanceModelForward
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128)):
    color = [0, 0, 0]  # if padding
    aspect_ratio = desired_size[0] / float(desired_size[1])
    current_ar = im.shape[0] / float(im.shape[1])
    if current_ar > aspect_ratio:  # current height is too high, pad width
        delta_w = int(round(im.shape[0] / aspect_ratio - im.shape[1]))
        left, right = delta_w / 2, delta_w - (delta_w / 2)
        new_im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    else:  # current width is too wide, pad height
        delta_h = int(round(im.shape[1] * aspect_ratio - im.shape[0]))
        top, bottom = delta_h / 2, delta_h - (delta_h / 2)
        new_im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                    value=color)
    # debug
    # import scipy.misc
    # scipy.misc.imsave('/tmp/new_im.jpg', new_im)
    return new_im


def get_descriptors(top_folder,model, force_compute=False, ext='dsc',
                    sample_size=16, batch_max=32):
    id_folders = os.listdir(top_folder)
    data,item = {},{}
    # im_mean, im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]
    batch_full = True
    for k, id_folder in enumerate(id_folders):
        if not id_folder.isdigit():
            continue
        p = os.path.join(top_folder, id_folder)
        # print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        if len(crop_files) <= 0:
            print "folder {0} is empty.".format(p)
            if not k==len(id_folders)-1:
                continue
        else:
            # interval = max(len(crop_files) / sample_size, 1)
            inds = numpy.linspace(0, len(crop_files), sample_size)
            crop_files = [crop_files[int(round(k))] for k in inds]
            if len(crop_files) > sample_size:
                crop_files = crop_files[:sample_size]
            if len(crop_files) < sample_size:
                pad_num = sample_size - len(crop_files)
                if pad_num <= sample_size/2:
                    crop_files = crop_files + crop_files[:pad_num]
                else:
                    crop_files = [random.choice(crop_files) for _ in range(sample_size)]

        if batch_full:
            ims = []
            descriptor_files = []
            batch_full = False
        for i, crop_file in enumerate(crop_files):

            descriptor_file = crop_file[:-4]+'.'+ext
            descriptor_files.append(descriptor_file)
            if os.path.isfile(descriptor_file) and (not force_compute):
                continue
            else:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im = crop_pad_fixed_aspect_ratio(im)
                im = cv2.resize(im, (128, 256))
                ims.append(im)

        if len(descriptor_files) >= batch_max or k==len(id_folders)-1:
            batch_full = True
            if len(ims) > 0:
                if torch.has_cudnn:
                    descriptor_batch = model.compute_features_on_batch(numpy.array(ims))
                else:
                    descriptor_batch = model.compute_features_on_batch(numpy.array(ims)) #,torch.from_numpy(numpy.array([aspect_ratio])).float())
            di = 0
            for i, descriptor_file in enumerate(descriptor_files):
                if os.path.isfile(descriptor_file) and (not force_compute):
                    descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
                else:
                    descriptor = descriptor_batch[di]
                    descriptor.tofile(descriptor_file)
                    di+=1

                descriptor = numpy.squeeze(descriptor)
                item['descriptor'] = descriptor
                item['file'] = descriptor_file
                descriptor_folder, _ = os.path.split(descriptor_file)
                if descriptor_folder not in data:
                    data[descriptor_folder] = []
                data[descriptor_folder].append(item.copy())
        print "finished folder {0}".format(p)
    return data

def distance(a,b):
    # cosine
    d0 = (1-numpy.dot(a,b))
    # euclidean
    d1 = numpy.linalg.norm(a-b)
    if abs(d0*2-d1)>0.0001:
        raise Exception('cosine and euclidean distance not equal')
    return d0


def process(model,folder, force_compute_desc, ext, sample_size):
    get_descriptors(folder, model, force_compute=force_compute_desc, ext=ext, sample_size=sample_size)
    mlog.info('descriptors were computed in {0}'.format(folder))


def process_root_folder(model,root_folder, force_compute_desc, ext, sample_size):
    folders = os.listdir(root_folder)
    for folder in folders:
        process(model, os.path.join(root_folder,folder), force_compute_desc, ext, sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='the path to the crops')

    parser.add_argument('model_path', type=str,
                        help='the path to appearance model file')

    parser.add_argument('ext', type=str,
                        help='the ext to appearance descriptor file')

    parser.add_argument('--device_id', type=int, default=0, required=False,
                        help='the gpu id')

    parser.add_argument('--batch_size', type=int, default=200, required=False,
                        help='the max size of a batch to feed to a gpu')

    parser.add_argument('--force_compute', action='store_true', default=False,
                    help='whether to force computing descriptors')

    parser.add_argument('--sample_size', type=int, default=16,
                        help='the num of samples from each ID')

    parser.add_argument('--parent_folder', action='store_true', default=False,
                        help='single folder process')



    args = parser.parse_args()
    start_time = time.time()
    print "sample size per ID={0}".format(args.sample_size)
    model = AppearanceModelForward(args.model_path, sys_device_ids=(args.device_id,))
    if args.parent_folder:
        process_root_folder(model, args.folder, args.force_compute, args.ext, args.sample_size)
    else:
        process(model, args.folder,args.force_compute, args.ext,args.sample_size)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'total time = {0}'.format(str(elapsed))