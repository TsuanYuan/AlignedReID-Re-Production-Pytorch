
import cv2
import os, glob, json
import argparse, logging
import numpy
import torch
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def get_descriptors(top_folder,model, device_id, force_compute=False, ext='dsc',
                    debug=False, with_roi=False, sample_size=64):
    id_folders = os.listdir(top_folder)
    data,item = {},{}
    if debug:
        print 'warning: writing images with weights!'
    for id_folder in id_folders:
        if not id_folder.isdigit():
            continue
        p = os.path.join(top_folder, id_folder)
        # print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        interval = max(len(crop_files) / sample_size, 1)
        crop_files = [crop_file for i, crop_file in enumerate(crop_files) if i % interval == 0]

        for i, crop_file in enumerate(crop_files):

            descriptor_file = crop_file[:-4]+'.'+ext
            if os.path.isfile(descriptor_file) and (not force_compute):
                descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            else:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (128, 256))
                imt = im.transpose(2, 0, 1)
                imt = (imt -128.0)/255
                imt = numpy.expand_dims(imt, 0)
                basename, _ = os.path.splitext(crop_file)
                json_file = basename + '.json'
                if os.path.isfile(json_file) and with_roi:
                    data = json.load(open(json_file, 'r'))
                    aspect_ratio = data['box'][2]/float(data['box'][3])
                else:
                    aspect_ratio = 0.5
                if torch.has_cudnn:
                    descriptor_var = model(Variable(torch.from_numpy(imt).float().cuda(device=device_id)),
                                           Variable(torch.from_numpy(numpy.array([aspect_ratio])).float().cuda(device=device_id)))

                else:
                    descriptor_var = model(Variable(torch.from_numpy(imt).float()),torch.from_numpy(numpy.array([aspect_ratio])).float())


                descriptor = descriptor_var.data.cpu().numpy()
                descriptor.tofile(descriptor_file)

            descriptor = numpy.squeeze(descriptor)
            item['descriptor'] = descriptor
            item['file'] = crop_file
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


def process(model,folder, device, force_compute_desc, ext, debug, with_roi, sample_size):
    get_descriptors(folder, model, device, force_compute=force_compute_desc, ext=ext, debug=debug,
                    with_roi=with_roi, sample_size=sample_size)
    mlog.info('descriptors were computed in {0}'.format(folder))

def process_root_folder(model,root_folder, device, force_compute_desc, ext, debug, with_roi, sample_size):
    folders = os.listdir(root_folder)
    for folder in folders:
        process_all_sub_folders(model, folder, device, force_compute_desc, ext, debug, with_roi, sample_size)


def process_all_sub_folders(model_path, folder, device, force_compute_desc, ext, debug, with_roi, sample_size):
    if device >=0:
        model = torch.load(model_path, map_location='cuda:{0}'.format(device))
    else:
        model = torch.load(model_path, map_location = lambda storage, loc: storage)

    sub_folders = next(os.walk(folder))[1]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        process(model, sub_folder_full, device, force_compute_desc, ext, debug, with_roi, sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='the path to the crops')

    parser.add_argument('model_path', type=str,
                        help='the path to appearance model file')

    parser.add_argument('ext', type=str,
                        help='the ext to appearance descriptor file')

    parser.add_argument('--device_id', type=int, default=-1, required=False,
                        help='the gpu id')

    parser.add_argument('--force_descriptor', action='store_true', default=False,
                    help='whether to force computing descriptors')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether save debug image crop with weights')

    parser.add_argument('--process_root', action='store_true', default=False,
                        help='take root folders of all tests')

    parser.add_argument('--with_roi', action='store_true', default=False,
                        help='whether to input aspect ratio')

    parser.add_argument('--sample_size', type=int, default=16,
                        help='the num of samples from each ID')

    args = parser.parse_args()
    print "sample size per ID={0}".format(args.sample_size)
    if not args.process_root:
        process(args.model_path, args.folder,
            args.device_id, args.force_descriptor, args.ext, args.debug, args.with_roi, args.sample_size)
    else:
        process_all_sub_folders(args.model_path, args.folder,
            args.device_id, args.force_descriptor, args.ext, args.debug, args.with_roi, args.sample_size)