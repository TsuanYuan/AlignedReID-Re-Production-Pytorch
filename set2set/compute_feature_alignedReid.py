
import cv2
import os, glob
import argparse, logging
import numpy
import time
import torch
from torch.autograd import Variable
from load_model import AppearanceModelForward

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128)):
    color = [0, 0, 0]  # zero padding
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

def get_descriptors(top_folder,model, device_id, force_compute=False, ext='dsc',
                    sample_size=16, batch_max=128):
    id_folders = os.listdir(top_folder)
    data,item = {},{}

    batch_full = True
    for i, id_folder in enumerate(id_folders):
        if not id_folder.isdigit():
            continue
        p = os.path.join(top_folder, id_folder)
        # print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        interval = max(len(crop_files) / sample_size, 1)
        crop_files = [crop_file for i, crop_file in enumerate(crop_files) if i % interval == 0]
        if batch_full:
            ims = []
            descriptor_files = []
            batch_full = False
        for i, crop_file in enumerate(crop_files):

            descriptor_file = crop_file[:-4]+'.'+ext
            descriptor_files.append(descriptor_file)
            if os.path.isfile(descriptor_file) and (not force_compute):
                continue
                #descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            else:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im = crop_pad_fixed_aspect_ratio(im)
                im = cv2.resize(im, (128, 256))
                imt = im.transpose(2, 0, 1)
                imt = imt/255.0
                #imt = numpy.expand_dims(imt, 0)
                ims.append(imt)
                # basename, _ = os.path.splitext(crop_file)
                #json_file = basename + '.json'
        if len(ims) == 0:
            continue
        if len(ims) >= batch_max or i==len(id_folders)-1:
            batch_full = True
            if torch.has_cudnn:
                descriptor_batch = model.extract_feature(numpy.array(ims))
            else:
                descriptor_batch = model.extract_feature(numpy.array(ims)) #,torch.from_numpy(numpy.array([aspect_ratio])).float())
            for i, descriptor_file in enumerate(descriptor_files):
                if os.path.isfile(descriptor_file) and (not force_compute):
                    descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
                else:
                    descriptor = descriptor_batch[i]
                    descriptor.tofile(descriptor_file)

                descriptor = numpy.squeeze(descriptor)
                item['descriptor'] = descriptor
                item['file'] = descriptor_file
                if id_folder not in data:
                    data[id_folder] = []
                data[id_folder].append(item.copy())
    return data

def distance(a,b):
    # cosine
    d0 = (1-numpy.dot(a,b))
    # euclidean
    d1 = numpy.linalg.norm(a-b)
    if abs(d0*2-d1)>0.0001:
        raise Exception('cosine and euclidean distance not equal')
    return d0


def process(model,folder, device, force_compute_desc, ext, sample_size):
    get_descriptors(folder, model, device, force_compute=force_compute_desc, ext=ext, sample_size=sample_size)
    mlog.info('descriptors were computed in {0}'.format(folder))


def process_root_folder(model,root_folder, device, force_compute_desc, ext, sample_size):
    folders = os.listdir(root_folder)
    for folder in folders:
        process(model, os.path.join(root_folder,folder), device, force_compute_desc, ext, sample_size)


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
    model = AppearanceModelForward(args.model_path, sys_device_ids=((args.device_id,), ))
    if args.parent_folder:
        process_root_folder(model, args.folder,
                            args.device_id, args.force_compute, args.ext, args.sample_size)
    else:
        process(model, args.folder,
                args.device_id, args.force_compute, args.ext,
                args.sample_size)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'total time = {0}'.format(str(elapsed))