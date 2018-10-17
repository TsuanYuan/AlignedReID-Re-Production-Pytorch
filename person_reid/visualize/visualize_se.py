"""
visualize excitation of se model
Quan Yuan
2018-10-08
"""

import argparse
import sys
import os
import glob
import cv2
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from struct_format import utils
from evaluate import feature_compute, load_model, misc
import numpy


def process_folder(person_folder, model, load_keypoints=False, batch_max=128, keypoints_score_th=0.75):
    p = person_folder
    crop_files = glob.glob(os.path.join(p, '*.jpg'))
    if len(crop_files) == 0:
        return numpy.array([]), []
    if load_keypoints:
        keypoint_file = os.path.join(person_folder, 'keypoints.pkl')
        with open(keypoint_file, 'rb') as fp:
            keypoints = pickle.load(fp)

    ims, kps, descriptors_from_gpus, files_used =[], [], [], []
    for i, crop_file in enumerate(crop_files):
        skip_reading = False
        file_only = os.path.basename(crop_file)
        if file_only not in keypoints:  # no keypoints detected on this crop image
            continue
        else:
            kp = feature_compute.best_keypoints(keypoints[file_only])
            kp_score = misc.keypoints_quality(kp)
            if kp_score < keypoints_score_th:
                continue
            else:
                if not skip_reading:
                    im_bgr = cv2.imread(crop_file)
                    w_h_ratio = float(im_bgr.shape[1]) / im_bgr.shape[0]
                    kp = feature_compute.adjust_keypoints_to_normalized_shape(kp, w_h_ratio)
                    kps.append(kp)
                    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                    im = feature_compute.crop_pad_fixed_aspect_ratio(im)
                    im = cv2.resize(im, (128, 256))
                    ims.append(im)
                    files_used.append(crop_file)

        if len(ims) > 0 and (len(ims) == batch_max or i == len(crop_files) - 1):
            if load_keypoints and (model.get_model_type() == load_model.Model_Types.HEAD_POSE or model.get_model_type() == load_model.Model_Types.LIMB_POSE
                                   or model.get_model_type() == load_model.Model_Types.HEAD_ONLY or model.get_model_type() == load_model.Model_Types.HEAD_POSE_REWEIGHT):
                descriptor_batch = model.compute_features_on_batch(ims, kps)
            else:
                descriptor_batch = model.compute_features_on_batch(ims)
            descriptors_from_gpus.append(descriptor_batch)
            ims, kps = [], []
    features = numpy.array(descriptors_from_gpus)
    files_used = numpy.array(files_used)
    return features, files_used


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize SE model excitations")

    parser.add_argument('folder', type=str, help="index of training folders, each folder contains multiple pid folders")
    parser.add_argument('model_file', type=str, help="the model file")
    parser.add_argument('--gpu_ids', type=int, default= 0, help="gpu id to use")

    args = parser.parse_args()