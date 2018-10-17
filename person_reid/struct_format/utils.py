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


def load_list_to_pid(list_file, data_folder, prefix, path_tail_len=2):
    pid_index = collections.defaultdict(list)
    with open(list_file) as f:
        for line in f:
            label = int(line.split()[0])+prefix
            groups = line.strip().split()[1:]
            num_imgs = len(groups) / 2
            # if label in pid_index:
            #     print "pid label {} already in pid_index. might have a conflict!".format(str(label))

            for i in xrange(num_imgs):
                part_name = groups[2 * i]
                within_idx = int(groups[2 * i + 1])
                if len(data_folder) > 0:
                    path_parts = os.path.normcase(part_name).split('/')[-path_tail_len:]
                    path_tail = os.path.join(*path_parts)
                    data_file = os.path.join(data_folder, path_tail)
                else:
                    data_file = part_name
                if os.path.isfile(data_file):
                    pid_index[label].append((data_file, within_idx))
    return pid_index


def load_list_of_unknown_tracks(list_file):
    # assume format of each row "ch03_2018089123232-00023231 data_path file_offset"
    video_track_index = collections.defaultdict(list)
    with open(list_file) as f:
        for line in f:
            video_track = line.split()[0]
            groups = line.strip().split()[1:]
            num_imgs = len(groups) / 2
            for i in xrange(num_imgs):
                data_file = groups[2 * i]
                within_idx = int(groups[2 * i + 1])
                if os.path.isfile(data_file):
                    video_track_index[video_track].append((data_file, within_idx))
        return video_track_index

def load_list_of_unknown_tracks_split(list_file, start_line, final_line, sample_size):
    # assume format of each row "ch03_2018089123232-00023231 data_path file_offset"
    video_track_index = collections.defaultdict(list)
    line_count = 0
    with open(list_file) as f:
        for line in f:
            if line_count>final_line:
                break
            elif line_count < start_line:
                line_count += 1
                continue
            else:
                video_track = line.split()[0]
                groups = line.strip().split()[1:]
                num_imgs = len(groups) / 2
                if num_imgs < sample_size:
                    line_count += 1
                    continue
                sample_ids = range(num_imgs)
                random.shuffle(sample_ids)
                for i in sample_ids[:sample_size]:
                    data_file = groups[2 * i]
                    within_idx = int(groups[2 * i + 1])
                    video_track_index[video_track].append((data_file, within_idx))
            line_count+=1
    return video_track_index


def convert_to_pid_index(tracklet_index):
    pid_index = collections.defaultdict(list)
    for crop_name in tracklet_index:
        data_file_name, start, pid = tracklet_index[crop_name]
        pid_index[pid].append((data_file_name, start))
    return pid_index


def read_one_image(data_file_path, place, bgr_flag=False):
    with open(data_file_path, 'rb') as f:
        f.seek(place + 4)
        name_len = struct.unpack('i', f.read(4))[0]
        f.seek(name_len, 1)
        img_len = struct.unpack('i', f.read(4))[0]
        img_bgr = cv2.imdecode(numpy.asarray(bytearray(f.read(img_len)), dtype="uint8"), cv2.IMREAD_COLOR)
        if bgr_flag:
            return img_bgr
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

def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128), head_top=False):
    color = [0, 0, 0] # zero padding
    aspect_ratio = desired_size[0]/float(desired_size[1])
    current_ar = im.shape[0]/float(im.shape[1])
    if current_ar > aspect_ratio: # current height is too high, pad width
      delta_w = int(round(im.shape[0]/aspect_ratio - im.shape[1]))
      left, right = delta_w / 2, delta_w - (delta_w / 2)
      new_im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT,
                                  value=color)
    else: # current width is too wide, pad height
      delta_h = int(round(im.shape[1]*aspect_ratio - im.shape[0]))
      if head_top:
        top, bottom = 0, delta_h
      else:
        top, bottom = delta_h/2, delta_h - (delta_h / 2)
      new_im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                  value=color)
    return new_im

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
        i = pos
        while i < pos + count:
            k = i % len(self.pid_index[pid])
            data_file, place = self.pid_index[pid][k]
            try:
                one_image = read_one_image(data_file, place)
                images.append(one_image)
                i += 1
            except:
                print "failed to read one image from path {}".format(data_file)

        self.pid_pos[pid] = (pos + count) % len(self.pid_index[pid])
        return images

    def get_pid_list(self):
        return self.pid_list


class MultiFileCrops(object):
    def __init__(self, data_folder, list_file, ignore_pids=None, same_day_camera=False):
        self.prefix = 0
        #index_files = glob.glob(os.path.join(data_folder, '*'+index_ext))
        self.data_folder = data_folder
        self.tracklet_index = {}
        self.pid_index = {}
        self.load_index_files(list_file, ignore_pids)
        if same_day_camera:
            self.convert_channel_time_records()  # put crops of same video together

        self.pid_pos = collections.defaultdict(int)
        self.pid_list = self.pid_index.keys()
        self.quality = {'w_h_max': 0.9, 'min_h':  96}
        self.same_day_camera = same_day_camera
        self.pids_no_good_qualities = set()
        self.pids_few_good_qualities = set()
        print 'crop qualities are w_h_max={}, min_h={}'.format(str(self.quality['w_h_max']), str(self.quality['min_h']))

    def convert_channel_time_records(self):
        # indexed by pid and data_file, which encodes channel and time
        self.pids_ch_time = {}
        for pid in self.pid_index:
            if pid not in self.pids_ch_time:
                self.pids_ch_time[pid] = {}
            for video_file, offset in self.pid_index[pid]:
                if video_file not in self.pids_ch_time[pid]:
                    self.pids_ch_time[pid][video_file] = []
                self.pids_ch_time[pid][video_file].append(offset)

    def load_index_files(self, list_file, ignore_pids=None):
        single_index = load_list_to_pid(list_file, self.data_folder, self.prefix)
        count = 0
        if ignore_pids is not None:
            for pid in ignore_pids:
                if pid in single_index:
                    count += 1
                    single_index.pop(pid, None)
        print 'removed {} ignore pids from training pid list'.format(str(count))
        self.pid_index.update(single_index)

    def prepare_im(self, one_image):
        im = crop_pad_fixed_aspect_ratio(one_image)
        im = cv2.resize(im, (128, 256))
        return im

    def decode_wanda_file(self, wanda_data_file):
        # decode ch16016_20180821120938_0
        # decode ch06002_20180918123747.mp4.short_0
        parts = wanda_data_file.split('_')
        ch = parts[0]
        date_time = parts[1].split('.')[0]
        date = int(date_time[:8])
        time = int(date_time[8:])
        return ch, date, time

    def save_no_good_file(self, no_good_file, file_list):
        # dump bad pids to files for ignore pids
        if not os.path.isfile(no_good_file):
            open(no_good_file, 'a').close()
        with open(no_good_file, 'r') as fp:
            n = len(fp.readlines())
            if len(file_list) > n:
                print "number of pids of {} are {}".format(os.path.basename(no_good_file), str(len(file_list)))
                with open(no_good_file, 'w') as fw:
                    for pid in file_list:
                        fw.write('{}\n'.format(str(pid)))

    def sort_data_by_file(self, data_file_place_pairs, count):
        # put same data file together, for better io performance
        new_data_file_place_pairs = []
        for i in range(0, len(data_file_place_pairs), count):
            current_batch = data_file_place_pairs[i:i+count]
            d = {}
            for data_file, offset in current_batch:
                if data_file not in d:
                    d[data_file] = []
                d[data_file].append(offset)
            for data_file in d:
                for offset in d[data_file]:
                    new_data_file_place_pairs.append((data_file, offset))
        return new_data_file_place_pairs


    def load_fixed_count_images_of_one_pid(self, pid, count):
        pos = self.pid_pos[pid]
        if self.same_day_camera:
            # select crops of same video file (ch, date, time)
            video_file_list = self.pids_ch_time[pid].keys()
            video_file_select = random.choice(video_file_list)
            places = self.pids_ch_time[pid][video_file_select]
            data_file_place_pairs = zip([video_file_select]*len(places), places)
        else:
            data_file_place_pairs = self.pid_index[pid]
        # shuffle the list of (file, offset) pairs
        random.shuffle(data_file_place_pairs)
        data_file_place_pairs = self.sort_data_by_file(data_file_place_pairs, count)
        i = 0
        visit_count = 0
        images = []
        low_quality_ones = []
        n = len(data_file_place_pairs)
        while i < count and visit_count < n:
            k = i%n
            data_file, place = self.pid_index[pid][k]
            visit_count += 1
            one_image = read_one_image(data_file, place)
            im = self.prepare_im(one_image)
            if one_image.shape[0] < self.quality['min_h']:
                low_quality_ones.append(im)
                continue
            if one_image.shape[1]/float(one_image.shape[0]) > self.quality['w_h_max']:
                low_quality_ones.append(im)
                continue
            images.append(im)
            i+=1

        if len(images) < count and len(images) > 0:
            images = [images[k%len(images)] for k in range(count)]
            if (not self.same_day_camera) and (pid not in self.pids_few_good_qualities) :
                self.pids_few_good_qualities.add(pid)
                if len(self.pids_few_good_qualities)%10 == 0:
                    few_good_file = '/tmp/few_good_quality.txt'
                    self.save_no_good_file(few_good_file, self.pids_few_good_qualities)

        elif len(images) == 0:
            images = [low_quality_ones[k%len(low_quality_ones)] for k in range(count)]
            if (not self.same_day_camera) and (pid not in self.pids_no_good_qualities):
                self.pids_no_good_qualities.add(pid)
                if len(self.pids_no_good_qualities)%10 == 0:
                    no_good_file = '/tmp/no_good_quality.txt'
                    self.save_no_good_file(no_good_file, self.pids_no_good_qualities)

        self.pid_pos[pid] = (pos+count)%len(self.pid_index[pid])
        return images

    def get_pid_list(self):
        return self.pid_list


class NoPidFileCrops(object):
    def __init__(self, index_file):
        self.track_index = {}
        self.load_index_file(index_file)
        self.pid_pos = collections.defaultdict(int)
        self.track_list = self.track_index.keys()

    def load_index_file(self, index_file):
        self.track_index = load_list_of_unknown_tracks(index_file)

    def load_fixed_count_images_of_one_pid(self, track_id, count):
        random.shuffle(self.track_index[track_id])
        images = []
        for i in range(count):
            k = i%len(self.track_index[track_id])
            data_file, place = self.track_index[track_id][k]
            one_image = read_one_image(data_file, place)
            images.append(one_image)
        return images

    def get_track_list(self):
        return self.track_list

    def save_index_file(self, output_file):
        with open(output_file, 'wb') as fp:
            pickle.dump(self.track_index, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print 'saved pid_index to {}'.format(output_file)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file_folder", type=str, help="path to input folder index files")
    args = ap.parse_args()

    index_files = glob.glob(os.path.join(args.index_file_folder, "*.list"))
    # pid_index = load_list_to_pid(index_files[3], prefix=0)


    #mfc = MultiFileCrops(args.index_file_folder, )
    #mfc.load_fixed_count_images_of_one_pid(5, 300)
    # sfc = SingleFileCrops(index_files)
    # index = SingleFileCrops.load_index(args.index_file)
    # pid_index = SingleFileCrops.convert_to_pid_index(index)
    # data_folder = os.path.split(args.index_file)[0]
    # images = SingleFileCrops.load_images_of_one_pid(pid_index, -1, data_folder)
