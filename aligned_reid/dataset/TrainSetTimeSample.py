from .Dataset import Dataset
# from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict
import random
import glob


class TrainSetTimeSample(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=1,
      ims_per_id=None,
      data_groups = None, # group of ids that are at the same time check point
      frame_interval=None,
      ignore_camera=False,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names

    min_num_ids = max(2, ids_per_batch / 2)
    print 'original num of groups={0}'.format(str(len(data_groups)))
    self.data_groups = self.remove_small_groups(data_groups, min_num_ids)
    print 'after removing of < {0} there are {1}'.format(str(min_num_ids), str(len(data_groups)))
    self.group_ids = self.data_groups.keys()
    # self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id
    self.frame_interval=frame_interval
    self.ignore_camera = ignore_camera
    self.batch_size = 1
    # im_ids = [parse_im_name(name, 'id') for name in im_names]
    # self.ids_to_im_inds = defaultdict(list)
    # for ind, id in enumerate(im_ids):
    #   self.ids_to_im_inds[id].append(ind)
    self.ids = range(len(self.data_groups))

    super(TrainSetTimeSample, self).__init__(
      dataset_size=len(self.data_groups),
      batch_size=self.batch_size,
      **kwargs)

  def remove_small_groups(self, data_groups, min_size):
    rm_keys = [k for k in data_groups if len(data_groups[k]['person_ids']) < min_size]
    for k in rm_keys:
      data_groups.pop(k, None)
    return data_groups

  def decode_im_file_name(self, im_filename):
    # assume file format "00000175_0000_00000006.jpg" as "personID_cameraID_frameIndex"
    no_ext, _ = osp.splitext(im_filename)
    parts = no_ext.split('_')
    camera_id = parts[0]
    frame_index = int(parts[-1])
    return camera_id, frame_index

  def get_sample_within_interval(self, im_paths):
    im_paths_sorted = sorted(im_paths)
    n = len(im_paths_sorted)
    im_names_valid = []
    if n <= self.ims_per_id:
      start_ind_local = 0
    else:
      start_ind_local = np.random.choice(n-self.ims_per_id, 1)[0]
    max_ind_local = min(n, start_ind_local+self.frame_interval)
    start_cid, start_fid = self.decode_im_file_name(im_paths_sorted[start_ind_local])
    # get all valid im names within a time interval
    for i in range(start_ind_local, max_ind_local):
      im_name = im_paths_sorted[i]
      camera_id, frame_index = self.decode_im_file_name(im_name)
      if camera_id != start_cid and (not self.ignore_camera):
        break
      if abs(frame_index-start_fid)>self.frame_interval:
        break
      im_names_valid.append(im_paths_sorted[i]) # im_path actually

    if len(im_names_valid) < self.ims_per_id:
      im_names = np.random.choice(im_names_valid, self.ims_per_id, replace=True)
    else:
      im_names = np.random.choice(im_names_valid, self.ims_per_id, replace=False)

    return im_names


  def get_sample(self, ptr):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    one_group = self.data_groups[self.group_ids[ptr]]
    im_folders = [osp.join(self.im_dir, folder_path) for folder_path in one_group['folder_paths']]
    random_pids = range(len(im_folders))
    if self.ids_per_batch < len(im_folders):
      random_pids = list(random.sample(set(range(len(im_folders))), self.ids_per_batch))
      im_folders = [im_folders[i] for i in random_pids]
    im_paths, labels = [], []
    for i, im_folder in enumerate(im_folders):
      all_ims = np.array(glob.glob(osp.join(im_folder, '*.jpg')))
      #print(str(all_ims))
      if self.frame_interval is None or self.frame_interval < 0:
        if len(all_ims) < self.ims_per_id:
          ims_one = np.random.choice(all_ims, self.ims_per_id, replace=True)
        else:
          ims_one = np.random.choice(all_ims, self.ims_per_id, replace=False)
      else:
        ims_one = self.get_sample_within_interval(all_ims)
      im_paths += ims_one.tolist()
      labels += [random_pids[i]]*len(ims_one)

    ims = [np.asarray(Image.open(im_path)) for im_path in im_paths]
    imgs, mirrored = zip(*[self.pre_process_im(im) for im in ims])

    return imgs, ims, labels, mirrored


  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done and self.shuffle:
      np.random.shuffle(self.ids)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, im_names, labels, mirrored = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(np.concatenate(im_list))
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    mirrored = np.concatenate(mirrored)
    return ims, im_names, labels, mirrored, self.epoch_done
