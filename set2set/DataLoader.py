
from __future__ import print_function, division
import os, glob
import torch
import pandas as pd
from skimage import io, transform
import numpy
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings

# to load reID set to set matching data set
class ReIDAppearanceSet2SetDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, sample_size=64):
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
        for item in subfolders:
            path = os.path.join(root_dir, item)
            if os.path.isdir(path) and item.isdigit():
                person_id = int(item)
                jpgs = glob.glob(os.path.join(path, '*.jpg'))
                self.person_id_im_paths[person_id] = jpgs

        self.transform = transform
        self.sample_size = sample_size

    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, person_id):
        im_paths = self.person_id_im_paths[person_id]
        #sample_size = numpy.random.random_integers(self.sample_size_range[0], self.sample_size_range[1])
        shuffled = random.shuffle(im_paths)
        im_paths_sample = shuffled[0:min(self.sample_size, len(im_paths))]
        ims = []
        for im_path in im_paths_sample:
            ims.append(io.imread(im_path))
        # get personID
        sample = {'images': ims, 'person_id': person_id}

        if self.transform:
            sample = self.transform(sample)

        return sample