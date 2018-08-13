
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
import evaluateCrops
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import aligned_reid.utils.utils
import aligned_reid.model.Model
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch

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
                                parts_model=False):

    im_mean, im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]

    TVTs, TMOs, relative_device_ids = aligned_reid.utils.utils.set_devices_for_ml(sys_device_ids)
    models = [aligned_reid.model.Model.Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes,
                    final_conv_out_channels=num_planes, base_model=base_name,with_final_conv=with_final_conv,
                    parts_model=parts_model)
        for _ in range(num_models)]
    model_ws = [DataParallel(models[i], device_ids=relative_device_ids[0]) for i in range(num_models)]

    optimizers = [None for m in models]
    model_opt = models + optimizers
    aligned_reid.utils.utils.load_ckpt(model_opt, model_weight_file, verbose=False)

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


def load_experts(experts_file, sys_device_ids, num_classes=1442):
    expert_models_feature_funcs, exts = [], []
    parts = False
    base_name = 'resnet50'
    num_planes = 2048

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


def encode_folder(person_folder, encoder, sample_size, ext, force_compute):
        p = person_folder
        # print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        interval = max(len(crop_files) / sample_size, 1)
        crop_files = [crop_file for i, crop_file in enumerate(crop_files) if i % interval == 0]
        descriptors = []
        for i, crop_file in enumerate(crop_files):

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

        return descriptors, crop_files

def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts', single_expert=False):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        if not single_expert:
            feature_arr = np.concatenate(tuple(descriptors))
            feature_arr = feature_arr / np.sqrt(float(len(descriptors)))
        else:
            feature_arr = np.array(descriptors)
        feature_arr.tofile(descriptor_file)

def load_descriptor_list(person_folder, encoders, exts, sample_size, force_compute, device_id):
    descriptors_for_encoders = [None]*len(exts)
    crop_files = None
    k = 0

    for encoder, ext in zip(encoders,exts):
        descriptors_for_encoders[k], crop_files = encode_folder(person_folder, encoder, sample_size, ext, force_compute)
        k += 1
    if len(encoders) > 1:
        descriptors_for_encoders = zip(*descriptors_for_encoders)    
    else:
        descriptors_for_encoders = descriptors_for_encoders[0]
    save_joint_descriptors(descriptors_for_encoders, crop_files, single_expert=(len(encoders) == 1))
    return descriptors_for_encoders

def compute_experts_distance_matrix(feature_list, single_expert=False):
    concat_list = []
    for feature_item in feature_list:
        if single_expert:
            concat_list += feature_item
            #feature_arr = np.array(concat_list)
        else:
            concat_list.append(np.concatenate(tuple(feature_item)))
    if single_expert:
        feature_arr = np.array(concat_list)
    else:
        feature_arr = np.array(concat_list)/np.sqrt(float(len(feature_list[0])))
    distance_matrix = sklearn.metrics.pairwise.cosine_distances(feature_arr)
    return distance_matrix

def process(data_folder,sample_size, encoder_list, exts, force_compute, device_id):

    sub_folders = os.listdir(data_folder)
    feature_list, file_seq_list, person_id_list = [], [], []
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit():
            person_id = int(sub_folder)
            descriptors = load_descriptor_list(os.path.join(data_folder,sub_folder),encoder_list, exts, sample_size, force_compute, device_id)
            person_id_seqs = [person_id]*len(descriptors)
            feature_list += descriptors
            person_id_list += person_id_seqs

    _, tail = os.path.split(data_folder)
    distance_matrix = compute_experts_distance_matrix(feature_list, len(exts)==1)
    auc95, dist_th,mAP = evaluateCrops.compute_metrics(distance_matrix, person_id_list, file_seq_list, file_tag=tail)
    mlog.info('AUC95={0} at dist_th={1}, mAP={2} on data set {3} with model extension {4}'
            .format('%.3f'%auc95, '%.6f'%dist_th, '%.3f'%mAP, data_folder, str(exts)))


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

    parser.add_argument('ext', type=str,
                        help='the extention of feature file')

    parser.add_argument('--sample_size', type=int, default=16,
                        help='the num of samples from each ID')

    parser.add_argument('--force_compute', action='store_true', default=False,
                        help='whether to force compute features')

    parser.add_argument('--device_id', type=int, default=0,
                        help='device id to run model')

    parser.add_argument('--single_folder', action='store_true', default=False,
                        help='process only current folder')


    args = parser.parse_args()
    print 'sample size per ID={0}'.format(args.sample_size)
    sys_device_ids = ((args.device_id,),)
    experts, exts = load_experts(args.experts_file, sys_device_ids)
    if args.single_folder:
        process(args.test_folder, args.sample_size, experts, exts, args.force_compute, sys_device_ids)
    else:
        process_all(args.test_folder, args.sample_size, experts, exts, args.force_compute, sys_device_ids)
