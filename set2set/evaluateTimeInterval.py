
"""
evaluation protocol of experts mix
Quan Yuan
2018-07-06
"""
import os, glob, types
import numpy as np
import logging
import argparse
import cv2
import sklearn.metrics.pairwise
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import aligned_reid.utils.utils
import aligned_reid.model.Model
import aligned_reid.model.SwitchClassHeadModel
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch
import numpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def resize_original_aspect_ratio(im, desired_size=(256, 128)):
    aspect_ratio = desired_size[1] / float(desired_size[0])
    current_ar = im.shape[1] / float(im.shape[0])
    if current_ar > aspect_ratio:  # current height is not high
        new_h = int(round(desired_size[1] / current_ar))
        new_im = cv2.resize(im, (desired_size[1], new_h))
    else:  # current width is not wide
        new_w = int(round(desired_size[0] * current_ar))
        new_im = cv2.resize(im, (new_w, desired_size[0]))
    # debug
    # import scipy.misc
    # scipy.misc.imsave('/tmp/new_im.jpg', new_im)
    return new_im

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

class ExtractFeature(object):
    """A function to be called in the val/test set, to extract features.
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, ims):
        old_train_eval_model = self.model.training
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model.eval()
        ims = Variable(torch.from_numpy(ims).float())
        global_feat, local_feat = self.model(ims)[:2]
        global_feat = global_feat.data.cpu().numpy()

        # Restore the model to its old train/eval mode.
        self.model.train(old_train_eval_model)
        return global_feat

def extract_image_patch(image, bbox, patch_shape, padding='zero'):
    """Extract image patch from bounding box.
    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """
    bbox = np.array(bbox)
    if padding=='full':
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
    else:
        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        if padding == 'zero':
            image = crop_pad_fixed_aspect_ratio(image, patch_shape)
        else:
            image = image
    if padding == 'roi':
        image = resize_original_aspect_ratio(image, patch_shape)
    else:
        image = cv2.resize(image, patch_shape[::-1])
    return image


def create_alignedReID_model_ml(model_weight_file, sys_device_ids=((0,),), image_shape = (256, 128, 3),
                                local_conv_out_channels=128, num_classes=301, num_models=1,
                                num_planes=2048, base_name='resnet50', with_final_conv=False,
                                parts_model=False, skip_fc=False):

    im_mean, im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]

    TVTs, TMOs, relative_device_ids = aligned_reid.utils.utils.set_devices_for_ml(sys_device_ids)
    models = [aligned_reid.model.Model.Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes,
                    final_conv_out_channels=num_planes, base_model=base_name,with_final_conv=with_final_conv,
                    parts_model=parts_model)
        for _ in range(num_models)]
    model_ws = [DataParallel(models[i], device_ids=relative_device_ids[0]) for i in range(num_models)]

    optimizers = [None for m in models]
    model_opt = models + optimizers
    aligned_reid.utils.utils.load_ckpt(model_opt, model_weight_file, verbose=False, skip_fc=skip_fc)

    feature_extraction_func = ExtractFeature(model_ws[0])

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                # out of bound
                patch = np.zeros((box[3], box[2], 3),dtype=np.uint8)

            # normalize image
            patch = patch/255.0
            if im_mean is not None:
                patch = patch - np.array(im_mean)
            if im_mean is not None and im_std is not None:
                patch = patch / np.array(im_std).astype(float)

            patch = patch.transpose((2, 0, 1))
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        global_feat = feature_extraction_func(image_patches)

        l2_norm = np.sqrt((global_feat * global_feat+1e-10).sum(axis=1))
        global_feat = global_feat / (l2_norm[:, np.newaxis])
        return global_feat

    return encoder


def load_experts(experts_file, sys_device_ids, skip_fc, num_classes=1442):
    expert_models_feature_funcs, exts = [], []

    with open(experts_file, 'r') as fp:
        for line in fp:
            parts = False
            base_name = 'resnet50'
            num_planes = 2048
            fields = line.rstrip('\n').rstrip(' ').split(' ')
            model_path, ext = fields[0], fields[1]
            folder_only, _ = os.path.split(model_path)
            folder_name = os.path.basename(os.path.normpath(folder_only))
            #file_only = os.path.basename(model_path)
            if folder_name.find('parts') >= 0:
                parts = True
            if folder_name.find('resnet34') >= 0 or folder_name.find('res34') >= 0:
                base_name = 'resnet34'
                num_planes = 512
            encoder = create_alignedReID_model_ml(model_path, sys_device_ids=sys_device_ids,
                                num_classes=num_classes, num_planes=num_planes, base_name=base_name,
                                parts_model=parts)
            expert_models_feature_funcs.append(encoder)
            exts.append(ext)
    return expert_models_feature_funcs, exts

def decode_raw_image_name(im_path):
    # get camera id, person id, frame index
    # assume xxxx_0002_00000121.jpg format
    folder_path, im_file = os.path.split(im_path)
    im_name, _ = os.path.splitext(im_file)
    id_folder = os.path.basename(folder_path)
    person_id = int(id_folder)
    us = im_name.split('_')
    frame_id = int(us[-1])
    camera_id = im_name[0:-len(us[-1])-len(us[-2])-2] # assume everything before person_id is camera id
    return camera_id, person_id, frame_id

def get_crop_files_at_interval(crop_files, frame_interval):
    crop_files = sorted(crop_files)
    current_cid = None
    current_pid = None
    current_fid = None
    crop_files_with_interval = []
    for crop_file in crop_files:
        camera_id, person_id, frame_id = decode_raw_image_name(crop_file)
        if current_cid is None:
            current_cid, current_pid, current_fid = camera_id, person_id, frame_id
            continue
        elif current_cid != camera_id or current_pid != person_id:
            #current_cid, current_pid, current_fid = camera_id, person_id, frame_id
            # only keep the crops from the first camera
            break
        else:
            frame_diff = frame_id - current_fid
            if frame_diff >= frame_interval and frame_diff < frame_interval*1.2: # allow slight variation if not exact
                crop_files_with_interval.append(crop_file)
                current_cid, current_pid, current_fid = camera_id, person_id, frame_id

    return crop_files_with_interval

def encode_folder(person_folder, encoder, frame_interval, ext, force_compute):
    p = person_folder
    # print 'descriptor computing in {0}'.format(p)
    crop_files = glob.glob(os.path.join(p, '*.jpg'))
    interval = frame_interval # max(len(crop_files) / sample_size, 1)

    #crop_files = [crop_file for i, crop_file in enumerate(crop_files) if i % interval == 0]
    crop_files_with_interval = get_crop_files_at_interval(crop_files, frame_interval)
    descriptors = []
    for i, crop_file in enumerate(crop_files_with_interval):

        descriptor_file = crop_file[:-4] + '.' + ext
        if os.path.isfile(descriptor_file) and (not force_compute):
            descriptor = np.fromfile(descriptor_file, dtype=np.float32)
        else:
            im_bgr = cv2.imread(crop_file)
            im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            box = np.array([0, 0, im.shape[1], im.shape[0]])
            descriptor = encoder(im, [box])
            if isinstance(descriptor, types.TupleType):
                descriptor = descriptor[0]
            #descriptor.tofile(descriptor_file)
        descriptor = np.squeeze(descriptor)
        #descriptor.tofile(descriptor_file)
        descriptors.append(descriptor)

    return descriptors, crop_files_with_interval

def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts'):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        feature_arr = np.concatenate(tuple(descriptors))
        feature_arr = feature_arr / np.sqrt(float(len(descriptors)))
        feature_arr.tofile(descriptor_file)

def load_descriptor_list(person_folder, encoders, exts, frame_interval, force_compute, device_id):
    descriptors_for_encoders = []#[None]*len(exts)
    crop_files = None
    k = 0

    for encoder, ext in zip(encoders,exts):
        descriptors_for_encoders_t, crop_files = encode_folder(person_folder, encoder, frame_interval, ext, force_compute)
        if len(crop_files)>0:
            descriptors_for_encoders.append(descriptors_for_encoders_t)    
	   
    descriptors_for_encoders = zip(*descriptors_for_encoders)
    save_joint_descriptors(descriptors_for_encoders, crop_files)
    return descriptors_for_encoders

def compute_experts_distance_matrix(feature_list):
    concat_list = []
    for feature_item in feature_list:
        concat_list.append(np.concatenate(tuple(feature_item)))
    feature_arr = np.array(concat_list)/np.sqrt(float(len(feature_list[0])))
    distance_matrix = sklearn.metrics.pairwise.cosine_distances(feature_arr)
    return distance_matrix

def compute_same_pair_distance_interval(features):
    if len(features) <= 1:
        return []
    else:
        features = numpy.squeeze(numpy.array(features))
        a = numpy.array(features[:-1])*numpy.array(features[1:])
        cos_dist = 1-numpy.sum(a, axis=1)
        return cos_dist.tolist()

def compute_diff_pair_distance(features1, features2):
    d = sklearn.metrics.pairwise.cosine_distances(numpy.array(features1), numpy.array(features2))
    return d.ravel().tolist()

def compute_diff_pair_distance_interval(feature_list):
    n = len(feature_list)
    d = []
    for i in range(n):
        for j in range(i+1, n):
            d += compute_diff_pair_distance(numpy.squeeze(feature_list[i]), numpy.squeeze(feature_list[j]))
    return d


def compute_interval_pair_distances(feature_list):
    same_pair_dist = []
    for features in feature_list:
        same_pair_dist += compute_same_pair_distance_interval(features)
    diff_pair_dist = compute_diff_pair_distance_interval(feature_list)
    return same_pair_dist, diff_pair_dist

def report_TP_at_FP(same_distances, diff_distances, fp_th=0.001):
    # AUC with true negative rate >= 95
    n_same = same_distances.size
    n_diff = diff_distances.size
    scores = 1-numpy.concatenate((same_distances, diff_distances))
    labels = numpy.concatenate((numpy.ones(n_same), -numpy.ones(n_diff)))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
    #fp_th = 0.05
    idx = numpy.argmax(fpr > fp_th)  # fp lower than 0.05 for auc 95
    if idx == 0:  # no points with fpr<=0.05
        return 0
    fpr = fpr[idx]
    tpr = tpr[idx]
    th = 1-thresholds[idx]
    #auc95 = sklearn.metrics.auc(fpr005, tpr005)/fp_th

    return tpr, fpr, th


def process(data_folder,frame_interval, encoder_list, exts, force_compute, device_id):

    sub_folders = os.listdir(data_folder)
    feature_list, file_seq_list, person_id_list = [], [], []
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit():
            person_id = int(sub_folder)
            descriptors = load_descriptor_list(os.path.join(data_folder,sub_folder),encoder_list, exts, frame_interval, force_compute, device_id)
            #person_id_seqs = [person_id]*len(descriptors)
            if len(descriptors) > 1:
                feature_list.append(descriptors)
            #person_id_list += person_id_seqs

    _, tail = os.path.split(data_folder)
    same_pair_dist, diff_pair_dist = compute_interval_pair_distances(feature_list)
    #distance_matrix = compute_experts_distance_matrix(feature_list)
    # auc95, dist_th,mAP = evaluateCrops.compute_metrics(distance_matrix, person_id_list, file_seq_list, file_tag=tail)
    same_pair_dist = numpy.array(same_pair_dist)
    diff_pair_dist = numpy.array(diff_pair_dist)
    tpr3, fpr3, th3 = report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.001)
    tpr4, fpr4, th4 = report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.0001)
    
    mlog.info('same_pairs are {0}, diff_pairs are {1}'.format(str(same_pair_dist.size), str(diff_pair_dist.size)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
            .format('%.3f'%tpr3, '%.6f'%th3, '%.5f'%fpr3, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
              .format('%.3f' % tpr4, '%.6f'%th4, '%.5f' % fpr4, data_folder, str(exts)))


def process_all(folder, sample_size, experts, exts, force_compute, sys_device_ids):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        process(sub_folder_full,sample_size, experts, exts, force_compute, sys_device_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of ground truth crops with computed features')

    parser.add_argument('experts_file', type=str,
                        help='the file of list of expert model paths')

    parser.add_argument('--frame_interval', type=int, default=16,
                        help='the num of samples from each ID')

    parser.add_argument('--force_compute', action='store_true', default=False,
                        help='whether to force compute features')

    parser.add_argument('--device_id', type=int, default=0,
                        help='device id to run model')

    parser.add_argument('--single_folder', action='store_true', default=False,
                        help='process only current folder')

    parser.add_argument('--skip_fc', action='store_true', default=False,
                        help='skip the fc layers')

    args = parser.parse_args()
    print 'frame interval={0}'.format(args.frame_interval)
    sys_device_ids = ((args.device_id,),)
    experts, exts = load_experts(args.experts_file, sys_device_ids, args.skip_fc)
    if args.single_folder:
        process(args.test_folder, args.frame_interval, experts, exts, args.force_compute, sys_device_ids)
    else:
        process_all(args.test_folder, args.frame_interval, experts, exts, args.force_compute, sys_device_ids)
