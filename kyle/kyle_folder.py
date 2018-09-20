#################################
# author renliang Weng
# date: 2018/09/13
################################
import torch.utils.data as data

from PIL import Image

import os
import struct 
import cv2 
import numpy as np 
import json 
import os.path

def get_int(f):
    tmp_str = f.read(4)
    if tmp_str == '': 
        return None
    else:
        number = struct.unpack('i', tmp_str)
        number = number[0]
    return number

def part_loader(part_info):
    # read binary part_file
    part_file = part_info[0]
    within_idx = part_info[1]
    with open(part_file, 'rb') as f:
        f.seek(within_idx)
        pid = get_int(f)
        img_name_len = get_int(f)
        img_path = f.read(img_name_len)
        img_size = get_int(f)
        img_bytes = np.asarray(bytearray(f.read(img_size)), dtype="uint8")
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        #result_path = 'debug/' + str(pid) + '-' + img_path.split('/')[-1].split('.')[0] + '.jpg'
        #print 'writing to '+ result_path
        #cv2.imwrite(result_path, image)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(image)
        return img.convert('RGB')



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def load_cls_pid_map(cls_pid_map_file):
    assert os.path.exists(cls_pid_map_file)
    f =open(cls_pid_map_file)
    class_to_idx = json.loads(f.read())
    f.close()
    classes = class_to_idx.keys()
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_parts_dataset(filelist):
    part_infos_map = {}
    assert os.path.exists(filelist)
    f = open(filelist)
    lines = f.readlines()
    images = []
    class_to_idx = {}
    classes = []
    print('making part datasets now')
    for idx, line in enumerate(lines):
        line = line.strip()
        cls_id = int(line.split()[0])
        classes.append(cls_id)
        class_to_idx[cls_id] = cls_id
        num_imgs = len(line.split()[1:]) / 2
        print('processing ' + str(idx) + ' line with ' + str(num_imgs) + ' imgs')
        groups = line.split()
        for i in xrange(num_imgs):
            part_file = groups[2 * i + 1]
            within_idx = int(groups[2 * i + 2])
            images.append([[part_file, within_idx], cls_id])

    print('we got ' + str(len(images)) + ' images')
    return images, class_to_idx, classes


def make_dataset(filelist, class_to_idx):
    images = []
    root = os.path.expanduser(root)
    assert os.path.exists(filelist)
    f = open(filelist)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        target = int(line.strip().split()[-1])
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

class AibeeDatasetPartsFolder(data.Dataset):
    """A data loader designed for aibee customer classification training: ::
    Args:
        filelist training absolute img_path and label file
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, filelist, transform=None):
        samples, class_to_idx, classes = make_parts_dataset(filelist)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in filelist of: " + filelist + "\n"))

        self.loader = part_loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform 
        self.target_transform = None 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class AibeeDatasetFolder(data.Dataset):
    """A data loader designed for aibee customer classification training: ::
    Args:
        root folder contains the common root path for all the training images
        filelist training relative img_path and label file
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, filelist, cls_pid_map_file, loader=pil_loader, transform=None):
        classes, class_to_idx = load_cls_pid_map(cls_pid_map_file)
        samples = make_dataset(root, filelist, class_to_idx)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



