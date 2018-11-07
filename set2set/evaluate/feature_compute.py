"""
utils for feature computing
Quan Yuan
2018-09-30
"""
import glob
import os
import numpy
import cv2
import pickle
from load_model import AppearanceModelForward, Model_Types
import misc
import json


def median_feature(features):
    x = numpy.median(features, axis=0)
    feature = x / (numpy.linalg.norm(x) + 0.0000001)
    return feature


def load_experts(experts_file, device_ids):
    models, exts = [], []

    with open(experts_file, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split(' ')
            model_path, ext = fields[0], fields[1]
            model = AppearanceModelForward(model_path, device_ids=device_ids)
            models.append(model)
            exts.append(ext)
    return models, exts


def adjust_keypoints_to_normalized_shape(keypoints, w_h_ratio, normalized_ratio=0.5):
    kp = numpy.copy(keypoints)
    if w_h_ratio < normalized_ratio:
        kp[:, 0] = (keypoints[:, 0] - 0.5) * w_h_ratio / normalized_ratio + 0.5
    else:
        kp[:, 1] = (keypoints[:, 1] - 0.5) * normalized_ratio / w_h_ratio + 0.5

    return kp


def best_keypoints(keypoints):
    if len(keypoints) == 1:
        return keypoints[0]
    else:
        best_score = 0
        best_kp = None
        for kp in keypoints:
            kp_score = misc.keypoints_quality(kp)
            if kp_score > best_score:
                best_score = kp_score
                best_kp = kp
    return best_kp


def load_valid_head(head_json_file, head_score_threshold, min_aspect_ratio):
    best_head_corner_box = None
    if os.path.isfile(head_json_file):
        with open(head_json_file, 'r') as fp:
            head_info = json.load(fp)
        head_boxes = head_info['head_boxes']
        head_scores = head_info['scores']
        n = len(head_boxes)
        if n > 0:
            valid_heads = [head_boxes[k] for k in range(n) if head_scores[k] > head_score_threshold]
            valid_heads = [valid_head for valid_head in valid_heads if float(min(valid_head[2:4])) / max(valid_head[2:4]) > min_aspect_ratio]
            if len(valid_heads) > 0:
                best_head_corner_box = sorted(valid_heads, key=lambda x: x[1])[0]
    return best_head_corner_box


def encode_image_files(crop_files, model, ext, force_compute, keypoint_file = None, batch_max=128, keypoints_score_th=0.75,
                  same_sample_size=-1, w_h_quality_th=0.9, min_crop_h=96):
    if same_sample_size > 0:
        sample_ids = numpy.linspace(0, len(crop_files)-1, same_sample_size).astype(int)
        sample_ids = numpy.unique(sample_ids)
        crop_files = numpy.array(crop_files)[sample_ids].tolist()

    files_from_files, files_from_gpus, descriptors_from_files, descriptors_from_gpus = [], [], [], []
    ims, kps = [], []
    keypoints = {}
    if keypoint_file is not None:
        with open(keypoint_file, 'rb') as fp:
            keypoints = pickle.load(fp)

    for i, crop_file in enumerate(crop_files):
        descriptor_file = crop_file[:-4] + '.' + ext
        skip_reading = False
        if os.path.isfile(descriptor_file) and (not force_compute):
            descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            descriptors_from_files.append(descriptor) #.reshape((1,descriptor.size)))
            files_from_files.append(crop_file)
        else:
            im_bgr = cv2.imread(crop_file)
            w_h_ratio = float(im_bgr.shape[1]) / im_bgr.shape[0]
            if model.get_model_type()!=Model_Types.HEAD_PLAIN and (w_h_ratio > w_h_quality_th or im_bgr.shape[0] < min_crop_h):  # a crop that is too wide, possibly a partial crop of head only or too small
                skip_reading = True

            if keypoint_file is not None and (not skip_reading):
                file_only = os.path.basename(crop_file)
                if file_only not in keypoints:  # no keypoints detected on this crop image
                    skip_reading = True
                else:
                    kp = best_keypoints(keypoints[file_only])
                    kp_score = misc.keypoints_quality(kp)
                    if kp_score < keypoints_score_th:
                        skip_reading = True
                    else:
                        kp = adjust_keypoints_to_normalized_shape(kp, w_h_ratio)
                        kps.append(kp)
            if model.get_model_type() == Model_Types.HEAD_PLAIN:
                jhd_file = os.path.splitext(crop_file)[0]+'.jhd'
                if not os.path.isfile(jhd_file):
                    skip_reading = True
                else:
                    head_detection_threshold, min_aspect_ratio = model.get_head_detection_quality_parameters()
                    head_box = load_valid_head(jhd_file, head_detection_threshold, min_aspect_ratio)
                    if head_box is None:
                        skip_reading = True
            if not skip_reading:
                if model.get_model_type() == Model_Types.HEAD_PLAIN:
                    im = model.crop_im(im_bgr, numpy.asarray(head_box))
                else:
                    im = model.crop_im(im_bgr)
                #im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                #im = misc.crop_pad_fixed_aspect_ratio(im, desired_size=desired_size)
                #im = cv2.resize(im, (desired_size[1], desired_size[0]))
                ims.append(im)
                files_from_gpus.append(crop_file)

        if len(ims) > 0 and (len(ims) == batch_max or i == len(crop_files)-1):
            if keypoint_file is not None and (model.get_model_type() == Model_Types.HEAD_POSE or model.get_model_type() == Model_Types.LIMB_POSE
                                   or model.get_model_type() == Model_Types.HEAD_ONLY or model.get_model_type() == Model_Types.HEAD_POSE_REWEIGHT
                                   or model.get_model_type() == Model_Types.HEAD_EXTRA or model.get_model_type() == Model_Types.LIMB_ONLY
                                   or model.get_model_type() == Model_Types.LIMB_EXTRA):
                assert len(ims) == len(kps)
                descriptor_batch = model.compute_features_on_batch(ims, kps)
            else:
                descriptor_batch = model.compute_features_on_batch(ims)
            descriptors_from_gpus.append(descriptor_batch)
            ims, kps = [], []
    if len(descriptors_from_gpus) + len(descriptors_from_files) == 0:
        return numpy.array([]), []
    else:
        return numpy.squeeze(numpy.concatenate(descriptors_from_gpus+descriptors_from_files)), files_from_files+files_from_gpus


def encode_folder(person_folder, model, ext, force_compute, batch_max=128, load_keypoints=False, keypoints_score_th=0.75,
                  same_sample_size=-1, w_h_quality_th=0.9, min_crop_h=96):
    p = person_folder
    crop_files = glob.glob(os.path.join(p, '*.jpg'))
    if len(crop_files) == 0:
        return numpy.array([]), []

    if load_keypoints:
        keypoint_file = os.path.join(person_folder, 'keypoints.pkl')
        if os.path.isfile(keypoint_file) == False:
            pid_folder_only = os.path.basename(os.path.normpath(person_folder))
            keypoint_file = os.path.join(person_folder, pid_folder_only+'_keypoints.pkl')
            if os.path.isfile(keypoint_file) == False:
                keypoint_file = None
    elif model.get_model_type() == Model_Types.HEAD_BOX_ATTEN or model.get_model_type() == Model_Types.HEAD_PLAIN:
        jhd_files = glob.glob(os.path.join(p, '*.jhd'))
        crop_files = [os.path.splitext(jhd_file)[0]+'.jpg' for jhd_file in jhd_files if os.path.isfile(os.path.splitext(jhd_file)[0]+'.jpg')]
        keypoint_file = None
    else:
        keypoint_file = None

    if len(crop_files) == 0:
        return numpy.array([]),[]
    else:
        return encode_image_files(crop_files, model, ext, force_compute, keypoint_file=keypoint_file, batch_max=batch_max,
                       keypoints_score_th=keypoints_score_th,
                       same_sample_size=same_sample_size, w_h_quality_th=w_h_quality_th, min_crop_h=min_crop_h)


def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts'):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        feature_arr = descriptors
        feature_arr = feature_arr / numpy.sqrt(float(len(descriptors)))
        feature_arr.tofile(descriptor_file)


def save_array_descriptors(descriptors_for_encoders, crop_files, ext):
    if len(descriptors_for_encoders.shape) == 1:
        descriptors_for_encoders = descriptors_for_encoders.reshape((1, descriptors_for_encoders.size))
    n = descriptors_for_encoders.shape[0]
    assert(len(crop_files)==n)
    for i in range(n):
        crop_file = crop_files[i]
        descriptor = descriptors_for_encoders[i,:]
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        descriptor.tofile(descriptor_file)


def load_descriptor_list(person_folder, model, ext, force_compute, batch_max, load_keypoints, keypoints_score_th, same_sampel_size):

    descriptors_for_encoders, crop_files = encode_folder(person_folder, model, ext, force_compute,
                                                         batch_max=batch_max,load_keypoints=load_keypoints, keypoints_score_th=keypoints_score_th,
                                                         same_sample_size=same_sampel_size)
    if len(crop_files) > 0:
        save_array_descriptors(descriptors_for_encoders, crop_files, ext)
    #save_joint_descriptors(descriptors_for_encoders, crop_files, ext=ext)
    return descriptors_for_encoders, crop_files


def load_descriptor_list_on_files(image_files, model, ext, force_compute, batch_max, keypoint_file, keypoints_score_th, same_sampel_size,
                                  w_h_quality_th=0.90, min_crop_h=96):

    descriptors_for_encoders, crop_files = encode_image_files(image_files, model, ext, force_compute, keypoint_file=keypoint_file,
                                                         batch_max=batch_max, keypoints_score_th=keypoints_score_th,
                                                         same_sample_size=same_sampel_size, w_h_quality_th=w_h_quality_th, min_crop_h=min_crop_h)
    if len(crop_files) > 0:
        save_array_descriptors(descriptors_for_encoders, crop_files, ext)
    return descriptors_for_encoders, crop_files
