"""
transform mars dataset for alignedReID training
"""
import os
import shutil
import pickle
# from aligned_reid.utils.dataset_utils import partition_train_val_set
# from aligned_reid.utils.dataset_utils import parse_im_name as parse_new_im_name
#from aligned_reid.utils.utils import save_pickle
new_im_name_tmpl = '{:08d}_{:04d}_{:08d}.jpg'

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

def transfer_one_image(image_path, save_dir, id_prefix, k, cameraIDs):
    fileonly = os.path.basename(image_path)
    id = int(fileonly[0:4]) + id_prefix
    cameraID = fileonly[4:6]
    if cameraID not in cameraIDs:
        cameraIDs[cameraID] = len(cameraIDs)
    dest_name = new_im_name_tmpl.format(id, cameraIDs[cameraID], k)
    dest_path = os.path.join(save_dir, dest_name)
    shutil.copy(image_path, dest_path)
    return dest_path

def transfer_one_folder(id_folder, save_dir, id_prefix, max_count_per_id):
    jpgList = listAllVideoFilesWithCameraKey(id_folder, '.jpg')
    jpgList += listAllVideoFilesWithCameraKey(id_folder, '.jpeg')
    cameraIDs = {}
    dest_paths = []
    for k, jpgPath in enumerate(jpgList):
        if k > max_count_per_id:  # keep samples per id under limit
            break
        dest_path = transfer_one_image(jpgPath, save_dir, id_prefix, k, cameraIDs)
        dest_paths.append(dest_path)
    destFileList = [os.path.basename(p) for p in dest_paths]
    return destFileList

def save_partitions(trainval_im_names, partition_file):
    # save all to training
    trainval_ids = list(set([parse_new_im_name(n, 'id')
                             for n in trainval_im_names]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    trainval_ids.sort()
    trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
    partitions = {'trainval_im_names': trainval_im_names,
                  'trainval_ids2labels': trainval_ids2labels,
                  'train_im_names': trainval_im_names,
                  'train_ids2labels': trainval_ids2labels,
                  'val_im_names': [],
                  'val_marks': [],
                  'test_im_names': [],
                  'test_marks': []}
    save_pickle(partitions, partition_file)
    print('Partition file saved to {}'.format(partition_file))
    # partition train/val/test set

    # trainval_ids = list(set([parse_new_im_name(n, 'id')
    #                          for n in train_test_split['trainval_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    # trainval_ids.sort()
    # trainval_ids2labels = dict(zip(trainval_ids, range(len(trainval_ids))))
    # partitions = partition_train_val_set(
    #     train_test_split['trainval_im_names'], parse_new_im_name, num_val_ids=100)
    # train_im_names = partitions['train_im_names']
    # train_ids = list(set([parse_new_im_name(n, 'id')
    #                       for n in partitions['train_im_names']]))
    # Sort ids, so that id-to-label mapping remains the same when running
    # the code on different machines.
    # train_ids.sort()
    # train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

    # A mark is used to denote whether the image is from
    #   query (mark == 0), or
    #   gallery (mark == 1), or
    #   multi query (mark == 2) set

    # val_marks = [0, ] * len(partitions['val_query_im_names']) \
    #             + [1, ] * len(partitions['val_gallery_im_names'])
    # val_im_names = list(partitions['val_query_im_names']) \
    #                + list(partitions['val_gallery_im_names'])
    #
    # test_im_names = list(train_test_split['q_im_names']) \
    #                 + list(train_test_split['mq_im_names']) \
    #                 + list(train_test_split['gallery_im_names'])
    # test_marks = [0, ] * len(train_test_split['q_im_names']) \
    #              + [2, ] * len(train_test_split['mq_im_names']) \
    #              + [1, ] * len(train_test_split['gallery_im_names'])
    #
    # partitions = {'trainval_im_names': train_test_split['trainval_im_names'],
    #               'trainval_ids2labels': trainval_ids2labels,
    #               'train_im_names': train_im_names,
    #               'train_ids2labels': train_ids2labels,
    #               'val_im_names': val_im_names,
    #               'val_marks': val_marks,
    #               'test_im_names': test_im_names,
    #               'test_marks': test_marks}
    # partition_file = osp.join(save_dir, 'partitions.pkl')
    # save_pickle(partitions, partition_file)
    # print('Partition file saved to {}'.format(partition_file))

def transform(input_folder, save_dir, file_range, id_prefix, max_count_per_id):
    #jpgList = listAllVideoFilesWithCameraKey(input_folder, '.jpg')
    id_folder_list = os.listdir(input_folder)
    n = len(id_folder_list)
    start_idx, end_idx = file_range.split(',')
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx < 0 or end_idx>n:
        end_idx = n
    print "transfer start={0}, end={1} of total {2} ID folders".format(str(start_idx), str(end_idx), str(n))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    dest_image_dir = os.path.join(save_dir, 'images')
    if not os.path.isdir(dest_image_dir):
        os.makedirs(dest_image_dir)

    dest_image_list = []
    for k in range(start_idx, end_idx):
        folder = os.path.join(input_folder, id_folder_list[k])
        folder_only = os.path.basename(folder)
        if folder_only.isdigit() == False or int(folder_only) == 0:  # ignore junk/distractor folder
            continue
        dest_image_paths = transfer_one_folder(folder, dest_image_dir, id_prefix, max_count_per_id)
        dest_image_list = dest_image_list + dest_image_paths
    partition_file = os.path.join(save_dir, 'partitions.pkl')
    save_partitions(dest_image_list, partition_file)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform Mars Dataset")
  parser.add_argument('--image_folder', type=str, help="dataset_original_folder", required=True)
  parser.add_argument('--save_dir', type=str, help="save_folder", required=True)
  parser.add_argument('--folder_range', type=str, help="range of folders to process in one batch", required=False,
                      default='0,-1')
  parser.add_argument('--id_prefix', type=int, help="prefix to add on an ID", required=False,
                      default=10000)
  parser.add_argument('--max_count_per_id', type=int, help="max count to sample in one id", required=False,
                      default=1000)
  args = parser.parse_args()
  image_folder = os.path.abspath(os.path.expanduser(args.image_folder))
  save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
  transform(image_folder, save_dir, args.folder_range, args.id_prefix, args.max_count_per_id)