
from __future__ import print_function, division
import os, glob
import torch
from skimage import io
import cv2
import numpy
from evaluate import feature_compute, misc
import random
import collections
from torch.utils.data import Dataset,ConcatDataset
import json
from struct_format.utils import SingleFileCrops, MultiFileCrops
import pickle


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


def resize_original_aspect_ratio(im, desired_size=(256, 128)):

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


class ConcatDayDataset(ConcatDataset):
    """
    random select samples through
    """
    def __init__(self, datasets, batch_size, data_size_factor=4):
        """
        :param datasets: list of ReIDSameDayDataset objects
        :param batch_size: num of ids per batch
        :param data_size_factor: as the total number of pids are huge, each epoch will iterate len(datasets)*data_size_factor iterations.
        Not necessarily all pids
        """
        datasets_valid = [dataset for dataset in datasets if len(dataset) >= batch_size]
        print("{} out of {} datasets are with at least {} pids.".format(str(len(datasets_valid)), str(len(datasets)), str(batch_size)))
        super(ConcatDayDataset, self).__init__(datasets_valid)
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.batch_size = batch_size
        self.data_size_factor = data_size_factor

    @staticmethod
    def item_sum(sequence):
        r = []
        for e in sequence:
            l = len(e)
            r.append(l)
        return r

    def __len__(self):
        return len(self.datasets)*self.data_size_factor

    def __getitem__(self, idx):
        dataset_idx = idx%len(self.datasets)

        sample_indices = random.sample(range(len(self.datasets[dataset_idx])), self.batch_size)
        batch = [self.datasets[dataset_idx][sample_idx] for sample_idx in sample_indices]
        return batch


class ReIDSingleFileCropsDataset(Dataset):
    """ReID data set with single file crops format"""
    def __init__(self, data_folder, index_file, transform=None, sample_size=8, desired_size=(256, 128), ignore_pid_list=None,
                 index_format='pickle', same_day_camera=False, min_crop_height=96, w_h_max=0.9):
        """
        Args:
            root_dir (string): Directory with all the index files and binary data files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_folder
        if index_format=='pickle':
            self.single_file_data = SingleFileCrops(data_folder)
        elif index_format=='list':
            self.single_file_data = MultiFileCrops(data_folder, index_file, ignore_pids=ignore_pid_list,
                                                   same_day_camera=same_day_camera, min_crop_height=min_crop_height, w_h_max=w_h_max)
        else:
            raise Exception('unknonw binary data index format {}'.format(index_format))
        self.person_ids = self.single_file_data.get_pid_list()
        self.sample_size = sample_size
        self.transform = transform
        self.desired_size = desired_size

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_ids[set_id]
        visit_count = 0
        while person_id in self.single_file_data.pids_no_good_qualities and visit_count<len(self.person_ids):
            set_id = (set_id+1)%len(self.person_ids)
            person_id = self.person_ids[set_id]
            visit_count += 1
        if visit_count == len(self.person_ids):
            raise Exception('no person_ids are of good quality!')

        ims = self.single_file_data.load_fixed_count_images_of_one_pid(person_id, self.sample_size)
        for i, im in enumerate(ims):
            ims[i], w_h_ratio = crop_pad_fixed_aspect_ratio(im, desired_size=self.desired_size)

        sample = {'images': ims, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([set_id]))
        return sample


def create_list_of_days_datasets(root_dir, transform=None, crops_per_id=8):
    datasets = []
    sub_folders = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if subfolder.isdigit()]
    person_id_im_paths = {}
    skip_count = 0
    for sub_folder in sub_folders:
        jpgs = glob.glob(os.path.join(root_dir, sub_folder, '*.jpg'))
        if len(jpgs) >= crops_per_id:
            for jpg_file in jpgs:
                channel, date, time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(jpg_file))
                if date not in person_id_im_paths:
                    person_id_im_paths[date] = collections.defaultdict(list)
                person_id_im_paths[date][pid].append(jpg_file)
        else:
            skip_count += 1

    for date in person_id_im_paths:
        dataset = ReIDSameDayDataset(person_id_im_paths[date], transform=transform, crops_per_id=crops_per_id)
        print("Total of {} classes are in the data set of date {}".format(str(len(dataset)), str(date)))
        datasets.append(dataset)

    return datasets


class ReIDSameDayDataset(Dataset):  # ch00002_20180816102633_00005504_00052119.jpg
    """ReID dataset each batch coming from the same day."""

    def __init__(self, person_id_data, transform=None, crops_per_id=8):
        """
        Args:
            person_id_data (string): dict with key of person pids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        person_id_data_valid = {}
        for pid in person_id_data:
            if len(person_id_data[pid]) < crops_per_id:
                continue
            else:
                person_id_data_valid[pid] = person_id_data[pid]
        print("keep {} pids > {} in {} pids".format(str(len(person_id_data_valid)), str(crops_per_id), str(len(person_id_data))))
        self.person_id_im_paths = person_id_data_valid
        self.transform = transform
        self.crops_per_id = crops_per_id


    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_im_paths.keys()[set_id]
        im_paths = self.person_id_im_paths[person_id]
        random.shuffle(im_paths)
        im_paths_sample = im_paths[0:min(self.crops_per_id, len(im_paths))]
        ims = []
        for im_path in im_paths_sample:
            im_bgr = cv2.imread(im_path)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            im, w_h_ratio = crop_pad_fixed_aspect_ratio(im_rgb)
            ims.append(im)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)
        channel, date, time, pid, frame_id = misc.decode_wcc_image_name(im_paths_sample[0])
        sample = {'images': ims, 'person_id': person_id, 'date': date}

        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([int(person_id)]))
        return sample



class ReIDSameIDSameDayCameraDataset(Dataset):  # ch00002_20180816102633_00005504_00052119.jpg
    """ReID dataset each batch coming from the same day same camera."""

    def __init__(self, root_dir, transform=None, crops_per_id=8, desired_size=(256, 128)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.person_id_dates = self.create_pid_same_day_camera_data(root_dir)
        self.transform = transform
        self.crops_per_id = crops_per_id
        self.desired_size = desired_size

    def create_pid_same_day_camera_data(self, root_dir):
        sub_folders = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if subfolder.isdigit()]
        person_id_date_cameras = {}
        self.person_id_2_class_id = {}
        class_id_count = 0
        for sub_folder in sub_folders:
            jpgs = glob.glob(os.path.join(root_dir, sub_folder, '*.jpg'))
            for jpg_file in jpgs:
                channel, date, time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(jpg_file))
                if pid not in person_id_date_cameras:
                    person_id_date_cameras[pid] = collections.defaultdict(list)
                    self.person_id_2_class_id[int(pid)] = class_id_count
                    class_id_count += 1
                date_camera = str(channel)+'_'+str(date)
                person_id_date_cameras[pid][date_camera].append(jpg_file)

        return person_id_date_cameras

    def __len__(self):
        return len(self.person_id_dates)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_dates.keys()[set_id]
        date_cameras = self.person_id_dates[person_id].keys()
        random.shuffle(date_cameras)

        im_paths = self.person_id_dates[person_id][date_cameras[0]]
        if len(im_paths) > self.crops_per_id:
            random.shuffle(im_paths)
            im_paths_sample = im_paths[0:min(self.crops_per_id, len(im_paths))]
        else:
            im_paths_sample = [random.choice(im_paths) for _ in range(self.crops_per_id)]
        ims = []
        for im_path in im_paths_sample:
            im_bgr = cv2.imread(im_path)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            im, w_h_ratio = crop_pad_fixed_aspect_ratio(im_rgb, desired_size=self.desired_size)
            ims.append(im)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)
        channel, date, time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(im_paths_sample[0]))

        sample = {'images': ims}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        class_id = self.person_id_2_class_id[person_id]
        sample['person_id'] = torch.from_numpy(numpy.array([int(class_id)]))
        sample['date'] = torch.from_numpy(numpy.array([int(date)]))
        sample['channel'] = torch.from_numpy(numpy.array([int(channel[2:])]))
        return sample


class ReIDSameIDOneDayDataset(Dataset):  # ch00002_20180816102633_00005504_00052119.jpg
    """ReID dataset each batch coming from the same day."""

    def __init__(self, root_dir, transform=None, crops_per_id=8, desired_size=(256, 128)):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.person_id_dates = self.create_pid_same_day_data(root_dir)
        self.transform = transform
        self.crops_per_id = crops_per_id
        self.desired_size = desired_size

    def create_pid_same_day_data(self, root_dir):
        sub_folders = [os.path.join(root_dir, subfolder) for subfolder in os.listdir(root_dir) if subfolder.isdigit()]
        person_id_dates = {}
        self.person_id_2_class_id = {}
        class_id_count = 0
        for sub_folder in sub_folders:
            jpgs = glob.glob(os.path.join(root_dir, sub_folder, '*.jpg'))
            for jpg_file in jpgs:
                channel, date, time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(jpg_file))
                if pid not in person_id_dates:
                    person_id_dates[pid] = collections.defaultdict(list)
                    self.person_id_2_class_id[int(pid)] = class_id_count
                    class_id_count += 1
                person_id_dates[pid][date].append(jpg_file)

        return person_id_dates

    def __len__(self):
        return len(self.person_id_dates)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_dates.keys()[set_id]
        dates = self.person_id_dates[person_id].keys()
        random.shuffle(dates)

        im_paths = self.person_id_dates[person_id][dates[0]]
        if len(im_paths) > self.crops_per_id:
            random.shuffle(im_paths)
            im_paths_sample = im_paths[0:min(self.crops_per_id, len(im_paths))]
        else:
            im_paths_sample = [random.choice(im_paths) for _ in range(self.crops_per_id)]
        ims = []
        for im_path in im_paths_sample:
            im_bgr = cv2.imread(im_path)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            im, w_h_ratio = crop_pad_fixed_aspect_ratio(im_rgb, desired_size=self.desired_size)
            ims.append(im)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)
        channel, date, time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(im_paths_sample[0]))

        sample = {'images': ims}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        class_id = self.person_id_2_class_id[person_id]
        sample['person_id'] = torch.from_numpy(numpy.array([int(class_id)]))
        sample['date'] = torch.from_numpy(numpy.array([int(date)]))
        return sample


class ReIDKeypointsDataset(Dataset):
    """ReID dataset."""

    def __init__(self, root_dir, transform=None, crops_per_id=8, normalized_shape=(128, 256)):
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
        self.person_id_keypoints = {}
        skip_count = 0
        for subfolder in subfolders:
            pid_folder = os.path.join(root_dir, subfolder)
            keypoints_pkl = os.path.join(root_dir, subfolder, 'keypoints.pkl')
            if not os.path.isfile(keypoints_pkl):
                skip_count += 1
                continue
            with open(keypoints_pkl, 'rb') as fp:
                keypoints = pickle.load(fp)
            if os.path.isdir(pid_folder) and subfolder.isdigit():
                person_id = int(subfolder)
                jpgs_with_kps = [os.path.join(pid_folder, p) for p in keypoints.keys()]
                kps = [keypoints[os.path.split(jpg)[1]]  for jpg in jpgs_with_kps]

                selected_kps, sids = self.keypoints_quality_selection(kps)
                # normalize kps to padded crop
                if len(selected_kps) >= 1:
                    self.person_id_im_paths[person_id] = numpy.array(jpgs_with_kps)[sids]
                    self.person_id_keypoints[person_id] = numpy.array(selected_kps)
                else:
                    skip_count += 1
        print('skipped {0} out of {1} sets for the size are smaller than 1, or without keypoints.pkl'.format(str(skip_count),
                                                                                                     str(len(subfolders))))

        self.transform = transform
        self.crop_per_id = crops_per_id

    def keypoints_quality_selection(self, kps):
        selected_ids = []
        selected_kp = []
        for i, kp in enumerate(kps):
            if len(kp) >= 1:
                mean_visible = numpy.array([numpy.mean(x, axis=0)[3] for x in kp])
                area_ratio = numpy.array([numpy.prod(numpy.max(x, axis=0)[0:2]-numpy.min(x, axis=0)[0:2]) for x in kp])
                quality_scores = mean_visible*10+area_ratio
                best_ids = numpy.where(quality_scores>0.75)[0]
                if len(best_ids) > 1 or len(best_ids) ==0:
                    continue
                else:
                    selected_kp.append(kp[best_ids[0]])
                    selected_ids.append(i)

        return selected_kp, numpy.array(selected_ids)

    def __len__(self):
        return len(self.person_id_im_paths)

    def __getitem__(self, set_id):
        # get personID
        person_id = self.person_id_im_paths.keys()[set_id]
        im_paths = self.person_id_im_paths[person_id]
        keypoints = self.person_id_keypoints[person_id]
        if len(im_paths) < self.crop_per_id:
            # with replacement
            permute_ids = numpy.random.choice(len(im_paths), self.crop_per_id)
        else:
            # no replacement
            permute_ids = numpy.random.permutation(range(len(keypoints)))[:min(self.crop_per_id, len(im_paths))]

        im_paths_sample = im_paths[permute_ids]
        keypoints_sample = keypoints[permute_ids]
        ims = []
        kps = []
        for im_path, kp in zip(im_paths_sample, keypoints_sample):
            im, w_h_ratio = crop_pad_fixed_aspect_ratio(io.imread(im_path))
            kp = feature_compute.adjust_keypoints_to_normalized_shape(kp, w_h_ratio)
            ims.append(im)
            kps.append(kp)

        sample = {'images': ims}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        # debug
        #for im, kp in zip(sample['images'], kps):
        #    parts_utils.visualize_keypoints_on_im(im.astype(numpy.uint8), [kp], 'sample')

        sample['person_id'] = torch.from_numpy(numpy.array([person_id]))
        sample['keypoints'] = torch.from_numpy(numpy.array(kps))

        return sample


class ReIDAppearanceDataset(Dataset):
    """ReID dataset."""

    def __init__(self, root_dir, transform=None, crops_per_id=8, desired_size=(256, 128)):
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
        self.person_id_2_class_id = {}
        skip_count = 0
        class_id_count = 0
        for item in subfolders:
            path = os.path.join(root_dir, item)
            if os.path.isdir(path) and item.isdigit():
                person_id = int(item)
                jpgs = glob.glob(os.path.join(path, '*.jpg'))
                if len(jpgs) >= crops_per_id:
                    self.person_id_im_paths[person_id] = jpgs
                    self.person_id_2_class_id[person_id] = class_id_count
                    class_id_count += 1
                else:
                    skip_count+=1
        print('skipped {0} out of {1} sets for the size are smaller than the sample_size={2}'.format(str(skip_count),str(len(subfolders)), str(crops_per_id)))

        self.transform = transform
        self.crop_per_id = crops_per_id
        self.desired_size = desired_size

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

            im, w_h_ratio = crop_pad_fixed_aspect_ratio(io.imread(im_path), desired_size=self.desired_size)
            w_h_ratios.append(w_h_ratio)
            ims.append(im)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)
        sample = {'images': ims, 'person_id': person_id}
        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([self.person_id_2_class_id[person_id]]))
        return sample


class ReIDMultiFolderAppearanceDataset(Dataset):
    """ReID dataset of multiple folders, triplets only comes from one folder each iter"""

    def __init__(self, root_dir, transform=None, crops_per_id=8, with_roi=False,
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
        print('skipped {0} out of {1} sets for the size are smaller than the sample_size={2}'.format(str(skip_count),str(len(subfolders)), str(crops_per_id)))

        self.transform = transform
        self.crop_per_id = crops_per_id
        self.original_ar = original_ar # whether to use fixed aspect ratio
        self.with_roi = with_roi

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
            im, w_h_ratio = crop_pad_fixed_aspect_ratio(io.imread(im_path))
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



class ReIDHeadAppearanceDataset(Dataset):  # ch00002_20180816102633_00005504_00052119.jpg
    """ReID dataset each batch coming from the same day."""

    def __init__(self, root_dir, transform=None, crops_per_id=8, head_score_threshold=0.65, desired_size=(64, 64), head_box_extension=1.2):
        """
        Args:
            person_id_data (string): dict with key of person pids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.head_box_extension = head_box_extension
        self.desired_size = desired_size
        self.crops_per_id = crops_per_id
        self.head_score_threshold = head_score_threshold
        self.create_head_pid_data(root_dir)
        self.transform = transform


    def create_head_pid_data(self, root_folder):
        subfolders = os.listdir(root_folder)
        self.head_pid_image_records = collections.defaultdict(list)
        skip_count = 0
        for pid_folder in subfolders:
            pid_path = os.path.join(root_folder, pid_folder)
            if os.path.isdir(pid_path) and pid_folder.isdigit():
                person_id = int(pid_folder)
                jpgs = glob.glob(os.path.join(pid_path, '*.jpg'))
                jpgs_with_good_head = []
                for jpg in jpgs:
                    head_json_file = os.path.splitext(jpg)[0]+'.jhd'
                    if os.path.isfile(head_json_file):
                        with open(head_json_file, 'r') as fp:
                            head_info = json.load(fp)
                        head_boxes = head_info['head_boxes']
                        head_scores = head_info['scores']
                        n = len(head_boxes)
                        if n > 0:
                            valid_heads = [head_boxes[k] for k in range(n) if head_scores[k] > self.head_score_threshold]
                            if len(valid_heads) > 0:
                                # debug only
                                if len(valid_heads) > 1:
                                    head_debug = 0
                                best_head_corner_box = sorted(valid_heads, key=lambda x: x[1])[0]  # find the smallest Y value for the highest head
                                jpgs_with_good_head.append((jpg, numpy.array(best_head_corner_box)))
                if len(jpgs_with_good_head) >= self.crops_per_id:
                    self.head_pid_image_records[person_id] = jpgs_with_good_head
                else:
                    skip_count += 1
        print('skipped {0} out of {1} sets for the size are smaller than the sample_size={2}'.format(str(skip_count),
                                                                                                     str(len(
                                                                                                         subfolders)),
                                                                                                     str(self.crops_per_id)))

    def __len__(self):
        return len(self.head_pid_image_records)

    def enforce_boundary(self, corner_box, im_w, im_h):
        corner_box[0] = min(max(int(round(corner_box[0])), 0), im_w - 1)
        corner_box[2] = min(max(int(round(corner_box[2])), 0), im_w - 1)
        corner_box[1] = min(max(int(round(corner_box[1])), 0), im_h - 1)
        corner_box[3] = min(max(int(round(corner_box[3])), 0), im_h - 1)
        return corner_box.astype(int)

    def extend_box(self, corner_box):
        box_w = corner_box[2] - corner_box[0]
        box_h = corner_box[3] - corner_box[1]
        w_ext = box_w/2*self.head_box_extension
        h_ext = box_h/2*self.head_box_extension
        box_center = numpy.array([(corner_box[0]+corner_box[2])/2, (corner_box[1]+corner_box[3])/2])
        extended_corner_box = numpy.zeros(4)
        extended_corner_box[0] = box_center[0] - w_ext
        extended_corner_box[2] = box_center[0] + w_ext
        extended_corner_box[1] = box_center[1] - h_ext
        extended_corner_box[3] = box_center[1] + h_ext
        return extended_corner_box

    def __getitem__(self, set_id):
        # get personID
        person_id = self.head_pid_image_records.keys()[set_id]
        im_paths_with_head = self.head_pid_image_records[person_id]
        random.shuffle(im_paths_with_head)
        im_heads_sample = im_paths_with_head[0:min(self.crops_per_id, len(im_paths_with_head))]
        ims = []
        for im_path in im_heads_sample:
            im_bgr = cv2.imread(im_path[0])
            head_corner_box = self.extend_box(im_path[1])
            head_corner_box = self.enforce_boundary(head_corner_box, im_bgr.shape[1], im_bgr.shape[0])
            im_bgr_head = im_bgr[head_corner_box[1]:head_corner_box[3], head_corner_box[0]:head_corner_box[2], :]
            im_rgb_head = cv2.cvtColor(im_bgr_head, cv2.COLOR_BGR2RGB)
            head_crop = crop_pad_fixed_aspect_ratio(im_rgb_head, self.desired_size)
            # im_rgb_head = cv2.resize(im_rgb_head, (self.desired_size[1], self.desired_size[0]))
            ims.append(im_rgb_head)
            # import scipy.misc
            # scipy.misc.imsave('/tmp/new_im.jpg', im)

        sample = {'images': ims, 'person_id': person_id}

        if self.transform:
            sample['images'] = self.transform(sample['images'])
        sample['person_id'] = torch.from_numpy(numpy.array([int(person_id)]))
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
