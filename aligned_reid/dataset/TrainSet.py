from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict


class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=None,
      ims_per_id=None,
      frame_interval=None,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id
    self.frame_interval=frame_interval

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    self.ids = self.ids_to_im_inds.keys()

    super(TrainSet, self).__init__(
      dataset_size=len(self.ids),
      batch_size=ids_per_batch,
      **kwargs)


  def decode_im_file_name(self, im_filename):
    # assume file format "00000175_0000_00000006.jpg" as "personID_cameraID_frameIndex"
    no_ext, _ = osp.splitext(im_filename)
    parts = no_ext.split('_')
    person_id = int(parts[0])
    camera_id = int(parts[1])
    frame_index = int(parts[2])
    return person_id, camera_id, frame_index

  def get_sample_within_interval(self, im_inds):
    im_names_class = sorted(np.array(self.im_names)[im_inds].tolist())
    im_names_valid = []
    start_ind_local = np.random.choice(len(im_inds), 1)[0]
    max_ind_local = min(len(im_inds), start_ind_local+self.frame_interval)
    _ ,start_cid, start_fid = self.decode_im_file_name(im_names_class[start_ind_local])
    # get all valid im names within a time interval
    for i in range(start_ind_local, max_ind_local):
      im_name = osp.basename(im_names_class[i])
      _, camera_id, frame_index = self.decode_im_file_name(im_name)
      if camera_id != start_cid or abs(frame_index-start_fid)>self.frame_interval:
        break
      im_names_valid.append(im_name)

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
    inds = self.ids_to_im_inds[self.ids[ptr]]
    if self.frame_interval is None or self.frame_interval<0:
      if len(inds) < self.ims_per_id:
        inds = np.random.choice(inds, self.ims_per_id, replace=True)
      else:
        inds = np.random.choice(inds, self.ims_per_id, replace=False)
      im_names = [self.im_names[ind] for ind in inds]
    else:
      im_names = self.get_sample_within_interval(inds)
    ims = [np.asarray(Image.open(osp.join(self.im_dir, name)))
             for name in im_names]

    ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[self.ids[ptr]] for _ in range(self.ims_per_id)]
    return ims, im_names, labels, mirrored


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
