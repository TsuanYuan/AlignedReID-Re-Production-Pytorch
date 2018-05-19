
import cv2
import os, glob
import argparse, logging
import numpy
import torch
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def get_descriptors(top_folder,model, device_id, max_count_per_id=-1, force_compute=False, ext='dsc', debug=False):
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
        for i, crop_file in enumerate(crop_files):
            if max_count_per_id>0 and i > max_count_per_id:
                break
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
                if torch.has_cudnn:
                    descriptor_var = model(Variable(torch.from_numpy(imt).float().cuda(device=device_id)))
                else:
                    descriptor_var = model(Variable(torch.from_numpy(imt).float()))
                descriptor = descriptor_var.data.cpu().numpy()
                descriptor.tofile(descriptor_file)
                # only for debug
                if debug:
                    dump_folder = '/tmp/seq_weights/'
                    if not os.path.isdir(dump_folder):
                        os.makedirs(dump_folder)
                    _, file_only = os.path.split(crop_file)
                    dump_file = os.path.join(dump_folder, file_only)
                    cv2.putText(im, '%.3f'%float(numpy.squeeze(descriptor)[-1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    im_rgb = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(dump_file, im_rgb)
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


def process(model,folder, device, force_compute_desc, ext, debug):
    # if torch.has_cudnn:
    #     model = torch.load(model_path, map_location = lambda storage, loc: 'cuda:{0}'.format(str(device)))
    # else:

    get_descriptors(folder, model, device, force_compute=force_compute_desc, ext=ext, debug=debug)
    mlog.info('descriptors were computed in {0}'.format(folder))


def process_all_sub_folders(model_path, folder, device, force_compute_desc, ext, debug):
    if device >=0:
        model = torch.load(model_path, map_location='cuda:{0}'.format(device))
    else:
        model = torch.load(model_path, map_location = lambda storage, loc: storage)

    sub_folders = next(os.walk(folder))[1] #[x[0] for x in os.walk(folder)]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        process(model, sub_folder_full, device, force_compute_desc, ext, debug)


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

    args = parser.parse_args()
    # print 'max count per folder is {0}'.format(str(MAX_COUNT_PER_ID))
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    process_all_sub_folders(args.model_path, args.folder,
            args.device_id, args.force_descriptor, args.ext, args.debug)