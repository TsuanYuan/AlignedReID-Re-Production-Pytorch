"""
utils to load binary strut format for crops data
Quan Yuan, converted from Zhu Qi and Renliang's code
2018-09-11
"""
import pickle
import collections
import argparse
import random
import os
import cv2
import struct
import numpy
import glob


def load_index(index_file):
    index = pickle.load(open(index_file, 'rb'))
    return index


def get_int(f):
    tmp_str = f.read(4)
    if tmp_str == '':
        return None
    else:
        number = struct.unpack('i', tmp_str)
        number = number[0]
        return number


def get_part_name(part_prefix, cur_part_idx):
    part_name = part_prefix + '_' + str(cur_part_idx)
    return part_name


def load_list_to_pid(list_file, prefix):
    pid_index = collections.defaultdict(list)
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            label = int(line.split()[0])+prefix
            groups = line.strip().split()[1:]
            num_imgs = len(groups) / 2
            # if label in pid_index:
            #     print "pid label {} already in pid_index. might have a conflict!".format(str(label))

            for i in xrange(num_imgs):
                part_name = groups[2 * i]
                within_idx = int(groups[2 * i + 1])
                pid_index[label].append((part_name, within_idx))
    return pid_index


def convert_to_pid_index(tracklet_index):
    pid_index = collections.defaultdict(list)
    for crop_name in tracklet_index:
        data_file_name, start, pid = tracklet_index[crop_name]
        pid_index[pid].append((data_file_name, start))
    return pid_index


def read_one_image(data_file_path, place):
    with open(data_file_path, 'rb') as f:
        f.seek(place + 4)
        name_len = struct.unpack('i', f.read(4))[0]
        f.seek(name_len, 1)
        img_len = struct.unpack('i', f.read(4))[0]
        img_bgr = cv2.imdecode(numpy.asarray(bytearray(f.read(img_len)), dtype="uint8"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img


def load_images_of_one_pid(pid_index, pid, data_folder):
    if pid not in pid_index:
        return None
    else:
        images = []
        for data_file_name, place in pid_index[pid]:
            data_file = os.path.join(data_folder, data_file_name)
            one_image = read_one_image(data_file, place)
            images.append(one_image)
            # cv2.imshow('w',one_image)
            # cv2.waitKey()
    return images


class SingleFileCrops(object):
    def __init__(self, index_folder, index_ext='.pickl'):
        index_files = glob.glob(os.path.join(index_folder, '*'+index_ext))
        self.data_folder = index_folder
        self.index = {}
        self.load_index_files(index_files)
        self.pid_index = convert_to_pid_index(self.index)
        self.pid_pos = collections.defaultdict(int)
        self.pid_list = self.pid_index.keys()


    def load_index_files(self, index_files):
        for index_file in index_files:
            single_index = load_index(index_file)
            self.index.update(single_index)

    def load_fixed_count_images_of_one_pid(self, pid, count):
        pos = self.pid_pos[pid]
        images = []
        if pos + count > len(self.pid_index[pid]):
            random.shuffle(self.pid_index[pid])
        for i in range(pos, pos + count):
            k = i%len(self.pid_index[pid])
            data_file_name, place = self.pid_index[pid][k]
            data_file = os.path.join(self.data_folder, data_file_name)
            one_image = read_one_image(data_file, place)
            images.append(one_image)
        self.pid_pos[pid] = (pos+count)%len(self.pid_index[pid])
        return images

    def get_pid_list(self):
        return self.pid_list

class MultiFileCrops(object):
    def __init__(self, data_folder, prefix, index_ext='.list'):
        self.prefix = prefix
        index_files = glob.glob(os.path.join(data_folder, '*'+index_ext))
        self.data_folder = data_folder
        self.tracklet_index = {}
        self.pid_index = {}
        self.load_index_files(index_files)
        self.pid_pos = collections.defaultdict(int)
        self.pid_list = self.pid_index.keys()


    def load_index_files(self, list_files):
        for list_file in list_files:
            single_index = load_list_to_pid(list_file, self.prefix)
            self.pid_index.update(single_index)

    def load_fixed_count_images_of_one_pid(self, pid, count, path_tail_len=2):
        pos = self.pid_pos[pid]
        images = []
        if pos + count > len(self.pid_index[pid]):
            random.shuffle(self.pid_index[pid])
        for i in range(pos, pos + count):
            k = i%len(self.pid_index[pid])
            data_file_name, place = self.pid_index[pid][k]
            path_parts = os.path.normcase(data_file_name).split('/')[-path_tail_len:]
            path_tail = os.path.join(*path_parts)
            data_file = os.path.join(self.data_folder, path_tail)
            one_image = read_one_image(data_file, place)
            images.append(one_image)
        self.pid_pos[pid] = (pos+count)%len(self.pid_index[pid])
        return images

    def get_pid_list(self):
        return self.pid_list

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file_folder", type=str, help="path to input folder index files")
    args = ap.parse_args()

    import glob
    index_files = glob.glob(os.path.join(args.index_file_folder, "*.list"))
    # pid_index = load_list_to_pid(index_files[3], prefix=0)
    mfc = MultiFileCrops(args.index_file_folder, prefix=0)
    mfc.load_fixed_count_images_of_one_pid(5, 300)
    # sfc = SingleFileCrops(index_files)
    # index = SingleFileCrops.load_index(args.index_file)
    # pid_index = SingleFileCrops.convert_to_pid_index(index)
    # data_folder = os.path.split(args.index_file)[0]
    # images = SingleFileCrops.load_images_of_one_pid(pid_index, -1, data_folder)
