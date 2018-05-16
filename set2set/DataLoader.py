
from __future__ import print_function, division
import os, glob
import torch
from skimage import io
import numpy
import random
from torch.utils.data import Dataset
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
                if len(jpgs) >= sample_size:
                    self.person_id_im_paths[person_id] = jpgs
                else:
                    warnings.showwarning('skipped person id = {0} for the size is smaller than the sample_size={1}'.format(str(person_id), str(sample_size)))

        self.transform = transform
        self.sample_size = sample_size

    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_im_paths.keys()[set_id]
        im_paths = self.person_id_im_paths[person_id]
        random.shuffle(im_paths)
        im_paths_sample = im_paths[0:min(self.sample_size, len(im_paths))]
        ims = []
        for im_path in im_paths_sample:
            ims.append(io.imread(im_path))
        sample = {'images': ims, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([person_id]))
        return sample