"""
example of compute features on a few images
Quan Yuan
2018-10-02
"""

import os
import glob
import argparse
import cv2
import load_model
import misc
import pickle
import numpy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of crops ')

    parser.add_argument('model_path', type=str,
                        help='the model path')

    parser.add_argument('--device_id', type=int, default=0,
                        help='device id to run model')

    args = parser.parse_args()
    model = load_model.AppearanceModelForward(args.model_path, single_device=args.device_id)
    kp_file = os.path.join(args.test_folder, 'keypoints.pkl')
    model_type = model.get_model_type()
    if model_type == load_model.Model_Types.LIMB_POSE or model_type == load_model.Model_Types.HEAD_POSE or model_type == load_model.Model_Types.HEAD_ONLY:
        if not os.path.isfile(kp_file):
            raise Exception('cannot find key points detection result file {}'.format(kp_file))
        with open(kp_file, 'rb') as fp:
            keypoints_dict = pickle.load(fp)
    else:
        keypoints_dict = None

    jpg_list = glob.glob(os.path.join(args.test_folder, "*.jpg"))
    ims = []
    keypoints = []
    for jpg_file in jpg_list:
        im_bgr = cv2.imread(jpg_file)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im = misc.crop_pad_fixed_aspect_ratio(im_rgb, (256, 128))
        ims.append(im)
        file_only = os.path.basename(jpg_file)
        if keypoints_dict is not None:
            keypoints_one = keypoints_dict[file_only][0]
            keypoints.append(keypoints_one)
    if keypoints_dict is not None:
        features = model.compute_features_on_batch(ims, numpy.array(keypoints))
    else:
        features = model.compute_features_on_batch(ims)