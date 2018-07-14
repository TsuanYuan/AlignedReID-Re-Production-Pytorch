"""
transform folder id dataset for alignedReID training
"""
import os
import shutil
import pickle
import numpy
import glob
import random
import re

new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

def partition_train_val_set(im_names, parse_im_name,
                            num_val_ids=None, val_prop=None, seed=1):
  """Partition the trainval set into train and val set.
  Args:
    im_names: trainval image names
    parse_im_name: a function to parse id and camera from image name
    num_val_ids: number of ids for val set. If not set, val_prob is used.
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results. If not to use,
      then set to `None`.
  Returns:
    a dict with keys (`train_im_names`,
                      `val_query_im_names`,
                      `val_gallery_im_names`)
  """
  numpy.random.seed(seed)
  # Transform to numpy array for slicing.
  if not isinstance(im_names, numpy.ndarray):
    im_names = numpy.array(im_names)
  numpy.random.shuffle(im_names)
  ids = numpy.array([parse_im_name(n, 'id') for n in im_names])
  cams = numpy.array([parse_im_name(n, 'cam') for n in im_names])
  unique_ids = numpy.unique(ids)
  numpy.random.shuffle(unique_ids)

  # Query indices and gallery indices
  query_inds = []
  gallery_inds = []

  if num_val_ids is None:
    assert 0 < val_prop < 1
    num_val_ids = int(len(unique_ids) * val_prop)
  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = numpy.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = numpy.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[numpy.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(numpy.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    num_selected_ids += 1
    if num_selected_ids >= num_val_ids:
      break

  if len(im_names) > 0:
      query_inds = numpy.hstack(query_inds)
      gallery_inds = numpy.hstack(gallery_inds)
      val_inds = numpy.hstack([query_inds, gallery_inds])
      trainval_inds = numpy.arange(len(im_names))
      train_inds = numpy.setdiff1d(trainval_inds, val_inds)

      train_inds = numpy.sort(train_inds)
      query_inds = numpy.sort(query_inds)
      gallery_inds = numpy.sort(gallery_inds)
  else:
      train_inds = []
      query_inds = []
      gallery_inds = []
  partitions = dict(train_im_names=im_names[train_inds],
                    val_query_im_names=im_names[query_inds],
                    val_gallery_im_names=im_names[gallery_inds])

  return partitions


def save_pickle(obj, path):
  """Create dir and save file."""
  # may_make_dir(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)

def parse_new_im_name(im_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = int(im_name[:8])
  else:
    parsed = int(im_name[9:13])
  return parsed

def listAllVideoFilesWithCameraKey(rootDir, ext):
    # list all files with ext at a root folder
    # param: rootDir the root folder
    # param: ext the file extension
    # return the list of all found files
    fileList = []
    for subdir, dirs, files in os.walk(rootDir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(ext):
                fileList.append(filepath)
    return fileList

def decode_raw_image_name(im_path):
    # get camera id, person id, frame index
    folder_path, im_file = os.path.split(im_path)
    id_folder = os.path.basename(folder_path)
    person_id = int(id_folder)
    us = im_file.split('_')
    frame_id = int(us[-2])
    camera_id = im_file[0:len(us[0])+len(us[1])+len(us[2])+2]
    return camera_id, person_id, frame_id

def transfer_one_image(image_path, save_dir, id, k, cameraIDs):
    # only work for macong 's format "dfxtd_ch04_20180519_00000083_00000001.jpg" and "dfxtd_ch04_20180519_00002675_00000015_1.jpg'"
    cameraID, person_id, frame_index = decode_raw_image_name(image_path)
    if cameraID not in cameraIDs:
        cameraIDs[cameraID] = len(cameraIDs)
    dest_name = new_im_name_tmpl.format(person_id, cameraIDs[cameraID], frame_index)
    dest_path = os.path.join(save_dir, dest_name)
    statinfo = os.stat(image_path)
    if statinfo.st_size > 0:
        shutil.copy(image_path, dest_path)
        return dest_path
    else:
        print('file might be corrupted {0}'.format(image_path))
        return None

def transfer_one_folder(id_folder, save_dir, id, max_count_per_id):
    jpgList = glob.glob(os.path.join(id_folder, '*.jpg'))
    jpgList += glob.glob(os.path.join(id_folder, '*.jpeg'))
    jpgList += glob.glob(os.path.join(id_folder, '*.png'))
    cameraIDs = {}
    dest_paths = []
    for k, jpgPath in enumerate(jpgList):
        if k > max_count_per_id:  # keep samples per id under limit
            break
        dest_path = transfer_one_image(jpgPath, save_dir, id, k, cameraIDs)
        if dest_path is None:
            continue
        dest_paths.append(dest_path)
    destFileList = [os.path.basename(p) for p in dest_paths]
    return destFileList

def save_partitions(train_images, gallery_images, query_images, partition_file):
    # save training tst
    trainval_im_names = [os.path.basename(p) for p in train_images]
    trainval_ids = list(set([parse_new_im_name(n, 'id')
                             for n in train_images]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    trainval_ids.sort()
    trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))

    partitions = partition_train_val_set(
        trainval_im_names, parse_new_im_name, num_val_ids=100)

    # train_im_names = partitions['train_im_names']
    # train_ids = list(set([parse_new_im_name(n, 'id')
     #                     for n in partitions['train_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    # train_ids.sort()
    # train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    val_marks = [0, ] * len(partitions['val_query_im_names']) \
                + [1, ] * len(partitions['val_gallery_im_names'])
    val_im_names = list(partitions['val_query_im_names']) \
                   + list(partitions['val_gallery_im_names'])

    query_im_names = [os.path.basename(p) for p in query_images]
    gallery_im_names = [os.path.basename(p) for p in gallery_images]
    test_im_names = list(query_im_names) \
                    + list(gallery_im_names)
    test_marks = [0, ] * len(query_im_names) \
                 + [1, ] * len(gallery_im_names)


    partitions = {'trainval_im_names': trainval_im_names,
                  'trainval_ids2labels': trainval_ids2labels,
                  'train_im_names': trainval_im_names,
                  'train_ids2labels': trainval_ids2labels,
                  'val_im_names': val_im_names,
                  'val_marks': val_marks,
                  'test_im_names': test_im_names,
                  'test_marks': test_marks}
    save_pickle(partitions, partition_file)
    print('Partition file saved to {}'.format(partition_file))

    train_query_ims, train_gallery_ims= split_test_query(trainval_im_names)
    testtrain_marks =  [0, ] * len(train_query_ims) \
                 + [1, ] * len(train_gallery_ims)
    partitions_testtrain = {'trainval_im_names': trainval_im_names,
                  'trainval_ids2labels': trainval_ids2labels,
                  'train_im_names': trainval_im_names,
                  'train_ids2labels': trainval_ids2labels,
                  'val_im_names': val_im_names,
                  'val_marks': val_marks,
                  'test_im_names': trainval_im_names,
                  'test_marks': testtrain_marks}
    partitions_testtrain_file = partition_file + ".testtrain.pkl"
    save_pickle(partitions_testtrain, partitions_testtrain_file)
    print('Partition train test file saved to {}'.format(partition_file))

def split_test_query(full_list):
    test_list = []
    query_list = []
    for p in full_list:
        n = numpy.random.choice(4)
        if n == 0:
            query_list.append(p)
        else:
            test_list.append(p)

    return test_list, query_list


def transform_train_test(folder, save_dir, file_range, id_prefix, max_count_per_id):
    id_folder_list = os.listdir(folder)
    n = len(id_folder_list)
    start_idx, end_idx = file_range.split(',')
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx < 0 or end_idx > n:
        end_idx = n
    print "transfer start={0}, end={1} of total {2} ID folders to {3}".format(str(start_idx), str(end_idx), str(n), save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dest_image_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(dest_image_dir):
        os.makedirs(dest_image_dir)
    dest_image_list = []
    for k in range(start_idx, end_idx):
        subfolder = os.path.join(folder, id_folder_list[k])
        folder_only = os.path.basename(subfolder)
        folder_predix = folder_only
        if folder_only.find("head") >= 0:
            print('{0} are skipped'.format(folder_only))
            continue
        if folder_only.find("upper_body") >=0:
            p = re.compile("(.*)_upper")
            folder_predix = p.search(folder_only).group(1)
            print('{0} are merged into {1}'.format(folder_predix, folder_only))
        if folder_only.find("full_body") >=0:
            p = re.compile("(.*)_full_body")
            folder_predix = p.search(folder_only).group(1)
            print('{0} are merged into {1}'.format(folder_predix, folder_only))

        if (folder_predix.isdigit() is False and folder_predix[0:8].isdigit() is False) or int(folder_predix) == 0:  # ignore junk/distractor folder
            continue
        id = id_prefix+int(folder_predix)
        dest_image_paths = transfer_one_folder(subfolder, dest_image_dir, id, max_count_per_id)
        dest_image_list = dest_image_list + dest_image_paths
    return dest_image_list

def transform(input_folder, save_dir, file_range, id_prefix, max_count_per_id):

    train_folder = os.path.join(input_folder, 'train')
    train_dest_images = transform_train_test(train_folder, save_dir, file_range, id_prefix, max_count_per_id)
    test_folder = os.path.join(input_folder, 'test')
    test_query_dest_images = transform_train_test(test_folder, save_dir, file_range, id_prefix, max_count_per_id)
    gallery_dest_images, query_dest_images = split_test_query(test_query_dest_images)
    partition_file = os.path.join(save_dir, 'partitions.pkl')
    save_partitions(train_dest_images, gallery_dest_images, query_dest_images, partition_file)

def randome_sample_train_test(id_dict, num_test, partition_id, save_dir, input_folder):
    person_ids = id_dict.keys()
    random.shuffle(person_ids)
    random_ids = person_ids
    test_ids = random_ids[0:num_test]
    train_ids = random_ids[num_test:]
    train_ims = []
    test_ims = []
    for id in train_ids:
        for im in id_dict[id]:

            train_ims.append(im)

    for id in test_ids:
        for im in id_dict[id]:
            test_ims.append(im)
    gallery_dest_images, query_dest_images = split_test_query(test_ims)
    partition_file = os.path.join(save_dir, 'partitions_{0}.pkl'.format(str(partition_id)))
    save_partitions(train_ims, gallery_dest_images, query_dest_images, partition_file)


def split_train_test(all_images_list, num_test, num_folds, save_dir, input_folder):
    id_dict = {}
    for image_path in all_images_list:
        file_only = os.path.basename(image_path)
        person_id = file_only[0:8]
        if person_id not in id_dict.keys():
            id_dict[person_id] = []
        id_dict[person_id].append(image_path)
    print("split {0} tests in {1} folds out of all {2} ids".format(str(num_test),str(num_folds), str(len(id_dict))))
    for k in range(num_folds):
        randome_sample_train_test(id_dict, num_test, k, save_dir, input_folder)


def transform_original(input_folder, save_dir, num_test, num_folds, id_prefix, max_count_per_id):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    all_images_list = transform_train_test(input_folder, save_dir, "0,-1", id_prefix, max_count_per_id)
    split_train_test(all_images_list, num_test, num_folds, save_dir, input_folder)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
  parser.add_argument('raw_folder', type=str, help="dataset_original_folder")
  parser.add_argument('save_dir', type=str, help="save_folder")
  parser.add_argument('--folder_range', type=str, help="range of folders to process in one batch", required=False,
                      default='0,-1')
  parser.add_argument('--id_prefix', type=int, help="prefix to add on an ID", required=False,
                      default=0)
  parser.add_argument('--max_count_per_id', type=int, help="max count to sample in one id", required=False,
                      default=100000)
  parser.add_argument('--num_test',  type=int, help="num ids in test set", required=False,
                      default=0)
  parser.add_argument('--num_folds', type=int, help="num folds in cross validation tests", required=False,
                      default=5)

  args = parser.parse_args()
  image_folder = os.path.abspath(os.path.expanduser(args.raw_folder))
  save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
  transform_original(image_folder, save_dir, args.num_test, args.num_folds, args.id_prefix, args.max_count_per_id)

  #transform(image_folder, save_dir, args.folder_range, args.id_prefix, args.max_count_per_id)

