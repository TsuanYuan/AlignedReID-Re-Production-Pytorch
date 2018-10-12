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
    return new_im


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


def encode_folder(person_folder, model, ext, force_compute, batch_max=128, load_keypoints=False, keypoints_score_th=0.75, same_sample_size=-1, w_h_quality_th=1.0):
    p = person_folder
    crop_files = glob.glob(os.path.join(p, '*.jpg'))
    if len(crop_files) == 0:
        return numpy.array([]), []

    files_from_files, files_from_gpus, descriptors_from_files, descriptors_from_gpus = [], [], [], []
    ims, kps = [], []
    keypoints = {}
    if load_keypoints:
        keypoint_file = os.path.join(person_folder, 'keypoints.pkl')
        with open(keypoint_file, 'rb') as fp:
            keypoints = pickle.load(fp)

    if same_sample_size > 0:
        sample_ids = numpy.linspace(0, len(crop_files)-1, same_sample_size).astype(int)
        sample_ids = numpy.unique(sample_ids)
        crop_files = numpy.array(crop_files)[sample_ids].tolist()

    for i, crop_file in enumerate(crop_files):
        descriptor_file = crop_file[:-4] + '.' + ext
        skip_reading = False
        if os.path.isfile(descriptor_file) and (not force_compute):
            descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            descriptors_from_files.append(descriptor.reshape((descriptor.size, 1)))
            files_from_files.append(crop_file)
        else:
            if load_keypoints:
                file_only = os.path.basename(crop_file)
                if file_only not in keypoints:  # no keypoints detected on this crop image
                    skip_reading = True
                else:
                    kp = best_keypoints(keypoints[file_only])
                    kp_score = misc.keypoints_quality(kp)
                    if kp_score < keypoints_score_th:
                        skip_reading = True
                    else:
                        im_bgr = cv2.imread(crop_file)
                        w_h_ratio = float(im_bgr.shape[1])/im_bgr.shape[0]
                        if w_h_ratio > w_h_quality_th:  # a crop that is too wide, possibly a partial crop of head only
                            skip_reading = True
                        else:
                            kp = adjust_keypoints_to_normalized_shape(kp, w_h_ratio)
                            kps.append(kp)
            if not skip_reading:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im = crop_pad_fixed_aspect_ratio(im)
                im = cv2.resize(im, (128, 256))
                ims.append(im)
                files_from_gpus.append(crop_file)

        if len(ims) > 0 and (len(ims) == batch_max or i == len(crop_files)-1):
            if load_keypoints and (model.get_model_type() == Model_Types.HEAD_POSE or model.get_model_type() == Model_Types.LIMB_POSE
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
        return numpy.concatenate((descriptors_from_files + descriptors_from_gpus)), files_from_files+files_from_gpus


def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts'):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        feature_arr = descriptors
        feature_arr = feature_arr / numpy.sqrt(float(len(descriptors)))
        feature_arr.tofile(descriptor_file)


def load_descriptor_list(person_folder, model, ext, force_compute, batch_max, load_keypoints, keypoints_score_th, same_sampel_size):

    descriptors_for_encoders, crop_files = encode_folder(person_folder, model, ext, force_compute,
                                                         batch_max=batch_max,load_keypoints=load_keypoints, keypoints_score_th=keypoints_score_th,
                                                         same_sample_size=same_sampel_size)
    save_joint_descriptors(descriptors_for_encoders, crop_files)
    return descriptors_for_encoders, crop_files
