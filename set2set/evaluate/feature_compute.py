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


def load_experts(experts_file, device_id):
    models, exts = [], []

    with open(experts_file, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split(' ')
            model_path, ext = fields[0], fields[1]
            model = AppearanceModelForward(model_path, single_device=device_id)
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


def decode_wcc_image_name(image_name):
    # decode ch00002_20180816102633_00005504_00052119.jpg
    image_base, _ = os.path.splitext(image_name)
    parts = image_base.split('_')
    channel = parts[0]
    date = parts[1][:8]
    video_time = parts[1][8:]
    pid = parts[2]
    frame_id = parts[3]
    return channel, int(date), int(video_time), int(pid), int(frame_id)


def encode_folder(person_folder, model, ext, force_compute, batch_max=128, load_keypoints=False):
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
                    kp = keypoints[file_only][0]
                    kps.append(kp)
            if not skip_reading:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                im = crop_pad_fixed_aspect_ratio(im)
                im = cv2.resize(im, (128, 256))
                ims.append(im)
                files_from_gpus.append(crop_file)
                
        if len(ims) == batch_max or i == len(crop_files)-1:
            if load_keypoints and (model.get_model_type() == Model_Types.HEAD_POSE or model.get_model_type() == Model_Types.LIMB_POSE):
                assert len(ims) == len(kps)
                descriptor_batch = model.compute_features_on_batch(ims, kps)
            else:
                descriptor_batch = model.compute_features_on_batch(ims)
            descriptors_from_gpus.append(descriptor_batch)
            ims, kps = [], []

    return numpy.concatenate((descriptors_from_files + descriptors_from_gpus)), files_from_files+files_from_gpus


def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts'):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        feature_arr = descriptors
        feature_arr = feature_arr / numpy.sqrt(float(len(descriptors)))
        feature_arr.tofile(descriptor_file)


def load_descriptor_list(person_folder, model, ext, force_compute, batch_max, load_keypoints):

    descriptors_for_encoders, crop_files = encode_folder(person_folder, model, ext, force_compute,
                                                         batch_max=batch_max,load_keypoints=load_keypoints)
    save_joint_descriptors(descriptors_for_encoders, crop_files)
    return descriptors_for_encoders, crop_files
