"""
utils for feature computing
Quan Yuan
2018-09-30
"""
import glob
import os
import numpy
import cv2
import types
from load_model import AppearanceModelForward

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


def get_crop_files_at_interval(crop_files, frame_interval):
    crop_files = sorted(crop_files)
    current_cid = None
    current_pid = None
    current_fid = None
    crop_files_with_interval = []
    sequence_crop_files = []
    for crop_file in crop_files:
        camera_id, _, _, person_id, frame_id = decode_wcc_image_name(os.path.basename(crop_file))
        if frame_interval < 0:
            if len(sequence_crop_files) == 0:
                sequence_crop_files.append([crop_file])
            else:
                sequence_crop_files[0].append(crop_file)
        else:
            if current_cid is None:
                current_cid, current_pid, current_fid = camera_id, person_id, frame_id
                sequence_crop_files.append([])
                sequence_crop_files[-1].append(crop_file)
                continue
            elif current_cid != camera_id or current_pid != person_id:
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


def encode_folder(person_folder, model, frame_interval, ext, force_compute, batch_max=128):
    p = person_folder
    crop_files = glob.glob(os.path.join(p, '*.jpg'))
    crop_files_with_interval = get_crop_files_at_interval(crop_files, frame_interval)
    files_from_files = []
    files_from_gpus = []
    descriptors_from_files = []
    descriptors_from_gpus = []
    ims = []

    for i, crop_file in enumerate(crop_files_with_interval):
        descriptor_file = crop_file[:-4] + '.' + ext

        if os.path.isfile(descriptor_file) and (not force_compute):
            descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            descriptors_from_files.append(descriptor.reshape((descriptor.size, 1)))
            files_from_files.append(crop_file)
        else:
            im_bgr = cv2.imread(crop_file)
            im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            im = crop_pad_fixed_aspect_ratio(im)
            im = cv2.resize(im, (128, 256))
            ims.append(im)
            files_from_gpus.append(crop_file)
        if len(ims) == batch_max or i == len(crop_files_with_interval)-1:
            descriptor_batch = model.compute_features_on_batch(ims)
            descriptors_from_gpus.append(descriptor_batch)
            ims = []

    return numpy.concatenate((descriptors_from_files + descriptors_from_gpus)), files_from_files+files_from_gpus


def save_joint_descriptors(descriptors_for_encoders, crop_files, ext='experts'):
    for descriptors, crop_file in zip(descriptors_for_encoders, crop_files):
        no_ext, _ = os.path.splitext(crop_file)
        descriptor_file = no_ext + '.' + ext
        feature_arr = descriptors
        feature_arr = feature_arr / numpy.sqrt(float(len(descriptors)))
        feature_arr.tofile(descriptor_file)


def load_descriptor_list(person_folder, model, ext, frame_interval, force_compute, batch_max):

    descriptors_for_encoders, crop_files = encode_folder(person_folder, model, frame_interval, ext, force_compute, batch_max=batch_max)
    save_joint_descriptors(descriptors_for_encoders, crop_files)
    return descriptors_for_encoders, crop_files

