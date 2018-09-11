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


class SingleFileCrops(object):
    def __init__(self, index_file):
        self.index = SingleFileCrops.load_index(index_file)
        self.pid_index = SingleFileCrops.convert_to_pid_index(self.index)
        self.pid_pos = collections.defaultdict(int)
        self.pid_list = self.pid_index.keys()

    def load_index_files(self, index_files):
        self.index = {}
        for index_file in index_files:
            single_index = SingleFileCrops.load_index(index_file)
            self.index.update(single_index)

    def load_fixed_count_images_of_one_pid(self, pid, data_folder, count):
        pos = self.pid_pos[pid]
        images = []
        if pos + count > len(self.pid_index[pid]):
            random.shuffle(self.pid_index[pid])
        for i in range(pos, pos + count):
            k = i%len(self.pid_index[pid])
            data_file_name, place = self.pid_index[pid][k]
            data_file = os.path.join(data_folder, data_file_name)
            one_image = SingleFileCrops.read_one_image(data_file, place)
            images.append(one_image)
        self.pid_pos[pid] = (pos+count)%len(self.pid_index[pid])
        return images

    def get_pid_list(self):
        return self.pid_list

    @staticmethod
    def load_index(index_file):
        index = pickle.load(open(index_file, 'rb'))
        return index

    @staticmethod
    def convert_to_pid_index(tracklet_index):
        pid_index = collections.defaultdict(list)
        for crop_name in tracklet_index:
            data_file_name, start, pid = tracklet_index[crop_name]
            pid_index[pid].append((data_file_name, start))
        return pid_index

    @staticmethod
    def read_one_image(data_file_path, place):
        with open(data_file_path, 'rb') as f:
            f.seek(place + 4)
            name_len = struct.unpack('i', f.read(4))[0]
            f.seek(name_len, 1)
            img_len = struct.unpack('i', f.read(4))[0]
            img = cv2.imdecode(numpy.asarray(bytearray(f.read(img_len)), dtype="uint8"), cv2.IMREAD_COLOR)
        return img

    @staticmethod
    def load_images_of_one_pid(pid_index, pid, data_folder):
        if pid not in pid_index:
            return None
        else:
            images = []
            for data_file_name, place in pid_index[pid]:
                data_file = os.path.join(data_folder, data_file_name)
                one_image = SingleFileCrops.read_one_image(data_file, place)
                images.append(one_image)
                # cv2.imshow('w',one_image)
                # cv2.waitKey()
        return images


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file", type=str, help="path to input folder sets")
    args = ap.parse_args()

    index = SingleFileCrops.load_index(args.index_file)
    pid_index = SingleFileCrops.convert_to_pid_index(index)
    data_folder = os.path.split(args.index_file)[0]
    images = SingleFileCrops.load_images_of_one_pid(pid_index, -1, data_folder)
