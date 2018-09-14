
from __future__ import print_function, division
import os, glob
import torch
from skimage import io
import cv2
import numpy
import random
from torch.utils.data import Dataset
import json
from struct_format.utils import SingleFileCrops, MultiFileCrops


def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128)):
    color = [0, 0, 0]  # zero padding
    aspect_ratio = desired_size[0] / float(desired_size[1])
    current_ar = im.shape[0] / float(im.shape[1])
    if current_ar > aspect_ratio:  # current height is too high, pad width
        delta_w = int(round(im.shape[0] / aspect_ratio - im.shape[1]))
        left, right = delta_w / 2, delta_w - (delta_w / 2)
        new_im = cv2.copyMakeBorder(im, 0, 0, int(left), int(right), cv2.BORDER_CONSTANT,
                                    value=color)
    else:  # current width is too wide, pad height
        delta_h = int(round(im.shape[1] * aspect_ratio - im.shape[0]))
        top, bottom = delta_h / 2, delta_h - (delta_h / 2)
        new_im = cv2.copyMakeBorder(im, int(top), int(bottom), 0, 0, cv2.BORDER_CONSTANT,
                                    value=color)

    return new_im, im.shape[1] / float(im.shape[0])


class ReIDSingleFileCropsDataset(Dataset):
    """ReID data set with single file crops format"""
    def __init__(self, data_folder, index_file, transform=None, sample_size=8, desired_size=(256, 128),
                 index_ext='.list'):
        """
        Args:
            root_dir (string): Directory with all the index files and binary data files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_folder
        self.single_file_data = MultiFileCrops(data_folder, index_file)
        self.person_ids = self.single_file_data.get_pid_list()
        self.sample_size = sample_size
        self.transform = transform
        self.desired_size = desired_size

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_ids[set_id]
        ims = self.single_file_data.load_fixed_count_images_of_one_pid(person_id, self.sample_size)
        for i, im in enumerate(ims):
            ims[i], w_h_ratio = crop_pad_fixed_aspect_ratio(im, desired_size=self.desired_size)

        sample = {'images': ims, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([person_id]))

        return sample



class ReIDAppearanceDataset(Dataset):
    """ReID dataset."""

    def __init__(self, root_dir, transform=None, id_sample_size=64, crops_per_id=8, with_roi=True,
                 original_ar=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ## folder structure, root_folder->person_id_folders->image_crops
        self.root_dir = root_dir
        subfolders = os.listdir(root_dir)
        self.person_id_im_paths = {}
        skip_count = 0
        for item in subfolders:
            path = os.path.join(root_dir, item)
            if os.path.isdir(path) and item.isdigit():
                person_id = int(item)
                jpgs = glob.glob(os.path.join(path, '*.jpg'))
                if len(jpgs) >= crops_per_id:
                    self.person_id_im_paths[person_id] = jpgs
                else:
                    skip_count+=1
        print('skipped {0} out of {1} sets for the size are smaller than the sample_size={2}'.format(str(skip_count),str(len(subfolders)), str(id_sample_size)))

        self.transform = transform
        self.id_sample_size = id_sample_size
        self.crop_per_id = crops_per_id
        self.original_ar = original_ar # whether to use fixed aspect ratio
        self.with_roi = with_roi

    def crop_pad_fixed_aspect_ratio(self, im, desired_size=(256, 128)):
        color = [0, 0, 0]  # zero padding
        aspect_ratio = desired_size[0] / float(desired_size[1])
        current_ar = im.shape[0] / float(im.shape[1])
        if current_ar > aspect_ratio:  # current height is too high, pad width
            delta_w = int(round(im.shape[0] / aspect_ratio - im.shape[1]))
            left, right = delta_w / 2, delta_w - (delta_w / 2)
            new_im = cv2.copyMakeBorder(im, 0, 0, int(left), int(right), cv2.BORDER_CONSTANT,
                                        value=color)
        else:  # current width is too wide, pad height
            delta_h = int(round(im.shape[1] * aspect_ratio - im.shape[0]))
            top, bottom = delta_h / 2, delta_h - (delta_h / 2)
            new_im = cv2.copyMakeBorder(im, int(top), int(bottom), 0, 0, cv2.BORDER_CONSTANT,
                                        value=color)
        # debug
        # import scipy.misc
        # scipy.misc.imsave('/tmp/new_im.jpg', new_im)
        return new_im, im.shape[1] / float(im.shape[0])


    def resize_original_aspect_ratio(self, im, desired_size=(256, 128)):

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


    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_im_paths.keys()[set_id]
        im_paths = self.person_id_im_paths[person_id]
        random.shuffle(im_paths)
        im_paths_sample = im_paths[0:min(self.crop_per_id, len(im_paths))]
        ims = []
        w_h_ratios = []
        for im_path in im_paths_sample:

            im, w_h_ratio = self.crop_pad_fixed_aspect_ratio(io.imread(im_path))
            basename, _ = os.path.splitext(im_path)
            json_path = basename+'.json'
            if not self.with_roi:
                w_h_ratio = 0.5
            elif os.path.isfile(json_path) and self.with_roi:
                data = json.load(open(json_path, 'r'))
                w_h_ratio = data['box'][2]/float(data['box'][3])

            w_h_ratios.append(w_h_ratio)
            ims.append(im)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)
        sample = {'images': ims, 'w_h_ratios':w_h_ratios, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([person_id]))
        sample['w_h_ratios'] = torch.from_numpy(numpy.array(w_h_ratios))
        return sample


# to load reID set to set matching data set
class ReIDAppearanceSet2SetDataset(Dataset):
    """ReID dataset."""

    def __init__(self, root_dir, transform=None, sample_size=64,  with_roi=True,
                 original_ar=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ## folder structure, root_folder->person_id_folders->image_crops
        self.root_dir = root_dir
        subfolders = os.listdir(root_dir)
        self.person_id_im_paths = {}
        self.person_id_mapping = {}  # map raw ids to [0,...,n]
        skip_count = 0
        id_count = 0
        for item in subfolders:
            path = os.path.join(root_dir, item)
            if os.path.isdir(path) and item.isdigit():
                person_id = int(item)
                jpgs = glob.glob(os.path.join(path, '*.jpg'))
                if len(jpgs) >= sample_size:
                    self.person_id_im_paths[person_id] = jpgs
                    self.person_id_mapping[person_id] = id_count
                    id_count+=1
                else:
                    skip_count+=1
        print('skipped {0} out of {1} sets for the size are smaller than the sample_size={2}'.format(str(skip_count),str(len(subfolders)), str(sample_size)))

        self.transform = transform
        self.sample_size = sample_size
        self.original_ar = original_ar # whether to use fixed aspect ratio
        self.with_roi = with_roi

    def crop_pad_fixed_aspect_ratio(self, im, desired_size=(256, 128)):
        color = [0, 0, 0]  # zero padding
        aspect_ratio = desired_size[0] / float(desired_size[1])
        current_ar = im.shape[0] / float(im.shape[1])
        if current_ar > aspect_ratio:  # current height is too high, pad width
            delta_w = int(round(im.shape[0] / aspect_ratio - im.shape[1]))
            left, right = delta_w / 2, delta_w - (delta_w / 2)
            new_im = cv2.copyMakeBorder(im, 0, 0, int(left), int(right), cv2.BORDER_CONSTANT,
                                        value=color)
        else:  # current width is too wide, pad height
            delta_h = int(round(im.shape[1] * aspect_ratio - im.shape[0]))
            top, bottom = delta_h / 2, delta_h - (delta_h / 2)
            new_im = cv2.copyMakeBorder(im, int(top), int(bottom), 0, 0, cv2.BORDER_CONSTANT,
                                        value=color)
        # debug
        # import scipy.misc
        # scipy.misc.imsave('/tmp/new_im.jpg', new_im)
        return new_im, im.shape[1] / float(im.shape[0])


    def resize_original_aspect_ratio(self, im, desired_size=(256, 128)):

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


    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_im_paths.keys()[set_id]
        im_paths = self.person_id_im_paths[person_id]
        random.shuffle(im_paths)
        im_paths_sample = im_paths[0:min(self.sample_size, len(im_paths))]
        ims = []
        w_h_ratios = []
        for im_path in im_paths_sample:

            im, w_h_ratio = self.crop_pad_fixed_aspect_ratio(io.imread(im_path))
            basename, _ = os.path.splitext(im_path)
            json_path = basename+'.json'
            if not self.with_roi:
                w_h_ratio = 0.5
            elif os.path.isfile(json_path) and self.with_roi:
                data = json.load(open(json_path, 'r'))
                w_h_ratio = data['box'][2]/float(data['box'][3])

            w_h_ratios.append(w_h_ratio)
            ims.append(im)
        sample = {'images': ims, 'w_h_ratios':w_h_ratios, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([self.person_id_mapping[person_id]]))
        sample['w_h_ratios'] = torch.from_numpy(numpy.array(w_h_ratios))
        return sample