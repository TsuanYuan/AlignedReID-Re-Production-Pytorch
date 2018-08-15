
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
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch
import numpy
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

HEAD_TOP = False

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

def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128), head_top=False):
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
        if head_top:
            top, bottom = 0, delta_h
        else:
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
        if local_feat is not None:
            local_feat = numpy.squeeze(local_feat.data.cpu().numpy())
        # Restore the model to its old train/eval mode.
        # self.model.train(old_train_eval_model)
        return global_feat, local_feat

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
            image = crop_pad_fixed_aspect_ratio(image, patch_shape, head_top=HEAD_TOP)
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
                                parts_model=False, skip_fc=False, local_feature_flag=False, model_name=''):

    im_mean, im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]

    TVTs, TMOs, relative_device_ids = aligned_reid.utils.utils.set_devices_for_ml(sys_device_ids)

    if len(model_name) ==0:
        models = [aligned_reid.model.Model.Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes,
                                                 final_conv_out_channels=num_planes, base_model=base_name, with_final_conv=with_final_conv,
                                                 parts_model=parts_model)
                  for _ in range(num_models)]
    elif model_name == 'attn':
        models = [
            aligned_reid.model.Model.AttentionModel(local_conv_out_channels=local_conv_out_channels,
                                              base_model=base_name, parts_model=parts_model)
            for _ in range(num_models)]
    elif model_name == 'mgn':
        models = [
            aligned_reid.model.Model.MGNModel(local_conv_out_channels=local_conv_out_channels,
                                            base_model=base_name, parts_model=parts_model)
            for _ in range(num_models)]
    elif model_name == 'upper':
        models = [
            aligned_reid.model.Model.UpperModel()
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
        global_feat, local_feature = feature_extraction_func(image_patches)
        if local_feature_flag:
            return local_feature
        else:
            l2_norm = np.sqrt((global_feat * global_feat+1e-10).sum(axis=1))
            global_feat = global_feat / (l2_norm[:, np.newaxis])
            return global_feat

    return encoder


def load_experts(experts_file, sys_device_ids, skip_fc, local_feature_flag, num_classes=1442):
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

            if folder_name.find('attn') >= 0:
                model_name = 'attn'
            elif folder_name.find('mgn') >= 0:
                model_name = 'mgn'
            elif folder_name.find('upper') >= 0:
                model_name = 'upper'
            else:
                model_name = ''
            print 'model name is {0}'.format(model_name)
            encoder = create_alignedReID_model_ml(model_path, sys_device_ids=sys_device_ids,
                                num_classes=num_classes, num_planes=num_planes, base_name=base_name,
                                parts_model=parts, local_feature_flag=local_feature_flag, skip_fc=skip_fc, model_name=model_name)
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
    sequence_crop_files = []
    for crop_file in crop_files:
        camera_id, person_id, frame_id = decode_raw_image_name(crop_file)
        if current_cid is None:
            current_cid, current_pid, current_fid = camera_id, person_id, frame_id
            sequence_crop_files.append([])
            sequence_crop_files[-1].append(crop_file)
            continue
        elif current_cid != camera_id or current_pid != person_id:
            #current_cid, current_pid, current_fid = camera_id, person_id, frame_id
            # only keep the crops from the first camera
            break
        else:
            frame_diff = frame_id - current_fid
            if frame_diff >= frame_interval and frame_diff < frame_interval*1.75: # allow slight variation if not exact
                sequence_crop_files[-1].append(crop_file)
                current_cid, current_pid, current_fid = camera_id, person_id, frame_id
                continue
            elif frame_diff >= frame_interval*1.5:
                sequence_crop_files.append([crop_file])
                current_cid, current_pid, current_fid = camera_id, person_id, frame_id
                continue

    longest = 0
    for crop_files in sequence_crop_files:
        if len(crop_files) > longest:
            longest = len(crop_files)
            crop_files_with_interval = crop_files
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


def load_descriptor_list(person_folder, encoders, exts, frame_interval, force_compute):
    descriptors_for_encoders = []#[None]*len(exts)
    crop_files = []
    k = 0
    for encoder, ext in zip(encoders,exts):
        descriptors_for_encoders_t, crop_files = encode_folder(person_folder, encoder, frame_interval, ext, force_compute)
        if len(crop_files)>0:
            descriptors_for_encoders.append(descriptors_for_encoders_t)    

    descriptors_for_encoders = zip(*descriptors_for_encoders)
    save_joint_descriptors(descriptors_for_encoders, crop_files)
    return descriptors_for_encoders, crop_files


def compute_experts_distance_matrix(feature_list):
    concat_list = []
    for feature_item in feature_list:
        concat_list.append(np.concatenate(tuple(feature_item)))
    feature_arr = np.array(concat_list)/np.sqrt(float(len(feature_list[0])))
    distance_matrix = sklearn.metrics.pairwise.cosine_distances(feature_arr)
    return distance_matrix


def shortest_dist(dist_mat):
  """Parallel version.
  Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
  Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
  """
  m, n = dist_mat.shape[:2]
  # Just offering some reference for accessing intermediate distance.
  dist = [[0 for _ in range(n)] for _ in range(m)]
  for i in range(m):
    for j in range(n):
      if (i == 0) and (j == 0):
        dist[i][j] = dist_mat[i, j]
      elif (i == 0) and (j > 0):
        dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
      elif (i > 0) and (j == 0):
        dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
      else:
        dist[i][j] = min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
  dist = dist[-1][-1]
  return dist


def local_dist(x, y):
  """
  Args:
    x: pytorch Variable, with shape [M, m, d]
    y: pytorch Variable, with shape [N, n, d]
  Returns:
    dist: pytorch Variable, with shape [M, N]
  """

  # shape [M * m, N * n]
  dist_mat = sklearn.metrics.pairwise.pairwise_distances(x, y)
  # dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
  # # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
  # dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
  # # shape [M, N]
  d = shortest_dist(dist_mat)
  return d

def compute_same_pair_distance_interval(features):
    if len(features) <= 1:
        return []
    else:
        if len(features[0][0].shape) == 1: # global
            features = numpy.squeeze(numpy.array(features))
            a = numpy.array(features[:-1])*numpy.array(features[1:])
            cost_dist = (1-numpy.sum(a, axis=1)).tolist()
        else:
            cost_dist = []
            n = len(features)
            for i in range(n-1):
                cost_dist.append(local_dist(numpy.array(numpy.squeeze(features[i])), numpy.array(numpy.squeeze(features[i+1]))))
        return cost_dist

def compute_diff_pair_distance(features1, features2):
    if len(features1[0].shape) == 1: # global
        d = sklearn.metrics.pairwise.cosine_distances(numpy.array(numpy.squeeze(features1)), numpy.array(numpy.squeeze(features2)))
        return d.ravel().tolist()
    else:
        d = []
        for feature1 in features1:
            for feature2 in features2:
                d.append(local_dist(feature1, feature2))
    return d

def compute_diff_pair_distance_interval(feature_list):
    n = len(feature_list)
    d = []
    for i in range(n):
        for j in range(i+1, n):
            d += compute_diff_pair_distance(numpy.squeeze(feature_list[i]), numpy.squeeze(feature_list[j]))
    return d

def make_diff_compare_list(crop_files_a, crop_files_b):
    compare_list = []
    for a in crop_files_a:
        for b in crop_files_b:
            compare_list.append((a,b))
    return compare_list

def pair_files(crop_file_list):
    same_file_pairs = []
    for crop_files in crop_file_list:
        same_file_pairs += zip(crop_files[:-1], crop_files[1:])

    diff_file_pairs = []
    n = len(crop_file_list)

    for i in range(n):
        for j in range(i + 1, n):
            diff_file_pairs += make_diff_compare_list(crop_file_list[i], crop_file_list[j])

    return same_file_pairs, diff_file_pairs

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
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)
    #fp_th = 0.05
    idx = numpy.argmin(numpy.abs(fpr - fp_th))
    #if idx == 0:  # no points with fpr<=0.05
    #    return 0
    fpr = fpr[idx]
    tpr = tpr[idx]
    th = 1-thresholds[idx]
    #auc95 = sklearn.metrics.auc(fpr005, tpr005)/fp_th

    return tpr, fpr, th

def get_filename_for_display(file_path):
    p1, _ = os.path.split(file_path)
    folder_name = os.path.basename(p1)
    bn = os.path.basename(file_path)
    bn, _ = os.path.splitext(bn)
    parts = bn.split('_')
    return parts[-2]+'_'+parts[-1], folder_name

def dump_pair_in_folder(file_pairs, pair_dist, output_path):
    import cv2
    im0 = cv2.imread(file_pairs[0])
    im1 = cv2.imread(file_pairs[1])
    im0 = cv2.resize(im0, (256, 512))
    im1 = cv2.resize(im1, (256, 512))
    canvas = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
    canvas[:,:256,:] = im0
    canvas[:,256:,:] = im1

    top_name, folder_name = get_filename_for_display(file_pairs[0])
    cv2.putText(canvas, str(top_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    top_name, folder_name = get_filename_for_display(file_pairs[1])
    cv2.putText(canvas, str(top_name), (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    cv2.putText(canvas, str(pair_dist), (120, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.imwrite(output_path, canvas)

def parse_im_files(image_path):
    file_only = os.path.basename(image_path)
    file_base, _ = os.path.splitext(file_only)
    parts = file_base.split('_')
    pid = int(parts[-2])
    frame_index = int(parts[-1])
    return pid, frame_index

def dump_difficult_pair_files(same_pair_dist, same_pair_files, diff_pair_dist, diff_pair_files, tough_diff_th=0.1, tough_same_th = 0.3, output_folder='/tmp/difficult/'):
    same_sort_ids = numpy.argsort(same_pair_dist)
    tough_same_ids = [i for i in same_sort_ids if same_pair_dist[i]>tough_same_th]
    if len(tough_same_ids) < 8:
        tough_num = min(max(int(round(len(same_sort_ids)*0.1)), 32), 128)
        tough_same_ids = same_sort_ids[-tough_num:]
    same_select_files, same_select_dist = [],[]
    same_dict = {}
    for id in tough_same_ids:
        p = same_pair_files[id]
        d = same_pair_dist[id]
        pid, _ = parse_im_files(p[0])
        if pid not in same_dict:
            same_dict[pid] = 1
        elif same_dict[pid] >= 3:
            continue
        else:
            same_dict[pid] += 1
        same_select_files.append(p)
        same_select_dist.append(d)

    tough_same_pairs = numpy.array(same_select_files)
    tough_same_dist = numpy.array(same_select_dist)

    diff_sort_ids = numpy.argsort(diff_pair_dist)
    tough_diff_ids = [i for i in diff_sort_ids if diff_pair_dist[i] < tough_diff_th]
    if len(tough_diff_ids) < 8:
        tough_num = min(max(int(round(len(diff_sort_ids)*0.1)), 32), 128)
        tough_diff_ids = diff_sort_ids[0:tough_num]
    diff_select_files, diff_select_dist = [], []
    diff_dict = {}
    for id in tough_diff_ids:
        p = diff_pair_files[id]
        d = diff_pair_dist[id]
        pid0, _ = parse_im_files(p[0])
        pid1, _ = parse_im_files(p[1])
        sorted_pids = tuple(sorted((pid0, pid1)))
        if sorted_pids not in diff_dict:
            diff_dict[sorted_pids] = 1
        elif diff_dict[sorted_pids] >= 3:
            continue
        else:
            diff_dict[sorted_pids] += 1
        diff_select_files.append(p)
        diff_select_dist.append(d)

    tough_diff_pairs = numpy.array(diff_select_files)
    tough_diff_dist = numpy.array(diff_select_dist)

    if os.path.isdir(output_folder):
        print 'remove existing {0} for difficult pairs output'.format(output_folder)
        shutil.rmtree(output_folder)

    same_folder = os.path.join(output_folder, 'same')
    if not os.path.isdir(same_folder):
        os.makedirs(same_folder)
    count = 0
    for dist, same_pair in zip(tough_same_dist, tough_same_pairs):
        file_path = os.path.join(same_folder, '{0}.jpg'.format(str(count)))
        dump_pair_in_folder(same_pair,dist, file_path)
        count+=1

    diff_folder = os.path.join(output_folder, 'diff')
    if not os.path.isdir(diff_folder):
        os.makedirs(diff_folder)
    count = 0
    for dist, file_pair in zip(tough_diff_dist, tough_diff_pairs):
        file_path = os.path.join(diff_folder, '{0}.jpg'.format(str(count)))
        dump_pair_in_folder(file_pair,dist, file_path)
        count+=1

    print 'difficult pairs were dumped to {0}'.format(output_folder)

def process(data_folder,frame_interval, encoder_list, exts, force_compute, dump_folder):

    sub_folders = os.listdir(data_folder)
    feature_list, file_seq_list, person_id_list,crops_file_list = [], [], [], []

    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit():
            person_id = int(sub_folder)
            descriptors, crop_files = load_descriptor_list(os.path.join(data_folder,sub_folder),encoder_list, exts, frame_interval, force_compute)
            #person_id_seqs = [person_id]*len(descriptors)
            if len(descriptors) > 1:
                feature_list.append(descriptors)
                crops_file_list.append(crop_files)
            #person_id_list += person_id_seqs
    # avoid bias towards person of long tracks
    mean_len = sum([len(crop_files) for crop_files in crops_file_list])/len(crops_file_list)
    len_limit = int(mean_len*1.5)
    for i, crop_files in enumerate(crops_file_list):
        if len(crop_files) > len_limit:
            crops_file_list[i] = crop_files[:len_limit]
            feature_list[i] = feature_list[:len_limit]
    _, tail = os.path.split(data_folder)
    same_pair_dist, diff_pair_dist = compute_interval_pair_distances(feature_list)
    same_pair_files, diff_pair_files = pair_files(crops_file_list)
    dump_difficult_pair_files(same_pair_dist, same_pair_files, diff_pair_dist, diff_pair_files, output_folder=dump_folder)

    #distance_matrix = compute_experts_distance_matrix(feature_list)
    # auc95, dist_th,mAP = evaluateCrops.compute_metrics(distance_matrix, person_id_list, file_seq_list, file_tag=tail)
    same_pair_dist = numpy.array(same_pair_dist)
    diff_pair_dist = numpy.array(diff_pair_dist)
    tpr2, fpr2, th2 = report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.01)
    tpr3, fpr3, th3 = report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.001)
    tpr4, fpr4, th4 = report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.0001)
    
    mlog.info('same_pairs are {0}, diff_pairs are {1}'.format(str(same_pair_dist.size), str(diff_pair_dist.size)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
            .format('%.3f'%tpr2, '%.6f'%th2, '%.5f'%fpr2, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
            .format('%.3f'%tpr3, '%.6f'%th3, '%.5f'%fpr3, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
              .format('%.3f' % tpr4, '%.6f'%th4, '%.5f' % fpr4, data_folder, str(exts)))

    return tpr2, tpr3, tpr4

def process_all(folder, frame_interval, experts, exts, force_compute, dump_folder):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    tps = []
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        tp3 = process(sub_folder_full,frame_interval, experts, exts, force_compute, dump_folder)
        tps.append(tp3)
    tps = numpy.array(tps)
    mean_tps = numpy.mean(tps, axis=0)
    mlog.info('average of tprs are {0}'.format(str(mean_tps)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of ground truth crops with computed features')

    parser.add_argument('experts_file', type=str,
                        help='the file of list of expert model paths')

    parser.add_argument('--frame_interval', type=int, default=16,
                        help='the num of samples from each ID')

    parser.add_argument('--dump_folder', type=str, default='/tmp/difficult',
                        help='whether to dump tough pairs')

    parser.add_argument('--force_compute', action='store_true', default=False,
                        help='whether to force compute features')

    parser.add_argument('--device_id', type=int, default=0,
                        help='device id to run model')

    parser.add_argument('--single_folder', action='store_true', default=False,
                        help='process only current folder')

    parser.add_argument('--skip_fc', action='store_true', default=False,
                        help='skip the fc layers')

    parser.add_argument('--local_feature', action='store_true', default=False,
                        help='use local feature to compare')

    parser.add_argument('--head_top', action='store_true', default=False,
                        help='crop attach at top')

    args = parser.parse_args()
    print 'frame interval={0}'.format(args.frame_interval)
    sys_device_ids = ((args.device_id,),)
    experts, exts = load_experts(args.experts_file, sys_device_ids, args.skip_fc, args.local_feature)
    import time
    HEAD_TOP = args.head_top
    if HEAD_TOP:
        print 'put partial head crop at top'

    start_time = time.time()
    if args.single_folder:
        process(args.test_folder, args.frame_interval, experts, exts, args.force_compute, args.dump_folder)
    else:
        process_all(args.test_folder, args.frame_interval, experts, exts, args.force_compute, args.dump_folder)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'total time = {0}'.format(str(elapsed))
