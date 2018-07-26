import numpy as np
import os.path as osp
ospj = osp.join
ospeu = osp.expanduser
import re
from ..utils.utils import load_pickle
#from ..utils.dataset_utils import parse_im_name
from .TrainSetTimeSample import TrainSetTimeSample
from .TrainSet import TrainSet
from .TestSet import TestSet


def create_dataset(
    name='market1501',
    part='trainval',
    **kwargs):
  assert name in ['customized','zeros','market1501', 'cuhk03', 'duke', 'public3','public4','folder_all','folder0', 'folder1','folder2', 'folder3','folder4', 'combined4'], \
    "Unsupported Dataset {}".format(name)

  assert part in ['trainval', 'train', 'val', 'test'], \
    "Unsupported Dataset Part {}".format(part)

  ########################################
  # Specify Directory and Partition File #
  ########################################
  dir = osp.dirname(__file__)
  if name == 'market1501':
    im_dir = ospeu(osp.join(dir,'../../Dataset/market1501/images'))
    partition_file = ospeu(osp.join(dir,'../../Dataset/market1501/partitions.pkl'))
  elif name == 'zeros':
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/zeros/images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/zeros/partitions_0.pkl')
  elif name == 'cuhk03':
    im_type = ['detected', 'labeled'][0]
    im_dir = ospeu(ospj(dir,'../../Dataset/cuhk03', im_type, 'images'))
    partition_file = ospeu(ospj(dir, '../../Dataset/cuhk03', im_type, 'partitions.pkl'))
  elif name == 'duke':
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/duke/images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/duke/partitions.pkl')
  elif name == 'public3':
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/market1501_cuhk03_duke/trainval_images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/market1501_cuhk03_duke/partitions.pkl')
  elif name == 'public4':
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/public4/trainval_images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/public4/partitions.pkl')
  elif name.find('combined')>=0:
    assert part in ['trainval'], \
      "Only trainval part of the combined dataset is available now."
    p = re.compile("combined(.*)")
    combine_num = p.search(name).group(1)
    im_dir = ospeu('/ssd/qyuan/combined_annotated_{0}/trainval_images'.format(combine_num))
    partition_file = ospeu('/ssd/qyuan/combined_annotated_{0}/partitions_{0}.pkl'.format(combine_num))
  elif name == 'folder_all':
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/folder_all_train/images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/folder_all_train/partitions_0.pkl')
  elif name.find('folder') >=0:
    p = re.compile("folder(.*)")
    folder_num = p.search(name).group(1)
    im_dir = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/folder_ready/images')
    partition_file = ospeu('/mnt/soulfs/qyuan/code/AlignedReID-Re-Production-Pytorch/Dataset/folder_ready/partitions_{0}.pkl'.format(folder_num))
  elif name == 'customized':
    im_dir = osp.join(kwargs['customized_id_folder_path'],'images')
    partition_file = osp.join(kwargs['customized_id_folder_path'],'partitions_{0}.pkl'.format(kwargs['partition_number']))



  ##################
  # Create Dataset #
  ##################

  # Use standard Market1501 CMC settings for all datasets here.
  # cmc_kwargs = dict(separate_camera_set=False,
  #                   single_gallery_shot=False,
  #                   first_match_break=True)
    
  # print partition_file
  # partitions = load_pickle(partition_file)
  # im_names = partitions['{}_im_names'.format(part)]

  if len(kwargs['customized_id_folder_path']) > 0:
    print partition_file
    partitions = load_pickle(partition_file)
    im_names = partitions['{}_im_names'.format(part)]

    if part == 'trainval':
      ids2labels = partitions['trainval_ids2labels']

      ret_set = TrainSet(
        im_dir=im_dir,
        im_names=im_names,
        ids2labels=ids2labels,
        **kwargs)

    elif part == 'train':
      ids2labels = partitions['train_ids2labels']

      ret_set = TrainSet(
        im_dir=im_dir,
        im_names=im_names,
        ids2labels=ids2labels,
        **kwargs)
    else:
      raise Exception('unknown create data set option')
  else:
    data_dir = osp.join(kwargs['customized_noid_folder_path'])
    group_file = kwargs['group_file']
    tracklet_groups = load_pickle(group_file)
    ret_set = TrainSetTimeSample(
      im_dir=data_dir,
      im_names=None,
      data_groups=tracklet_groups,
      **kwargs)

  return ret_set

  # if part == 'trainval':
  #   ids2labels = partitions['trainval_ids2labels']
  #
  #   ret_set = TrainSet(
  #     im_dir=im_dir,
  #     im_names=im_names,
  #     ids2labels=ids2labels,
  #     **kwargs)
  #
  # elif part == 'train':
  #   ids2labels = partitions['train_ids2labels']
  #
  #   ret_set = TrainSet(
  #     im_dir=im_dir,
  #     im_names=im_names,
  #     ids2labels=ids2labels,
  #     **kwargs)
  #
  # elif part == 'val':
  #   marks = partitions['val_marks']
  #   kwargs.update(cmc_kwargs)
  #
  #   ret_set = TestSet(
  #     im_dir=im_dir,
  #     im_names=im_names,
  #     marks=marks,
  #     **kwargs)
  #
  # elif part == 'test':
  #   marks = partitions['test_marks']
  #   kwargs.update(cmc_kwargs)
  #
  #   ret_set = TestSet(
  #     im_dir=im_dir,
  #     im_names=im_names,
  #     marks=marks,
  #     **kwargs)
  #
  # if part in ['trainval', 'train']:
  #   num_ids = len(ids2labels)
  # elif part in ['val', 'test']:
  #   ids = [parse_im_name(n, 'id') for n in im_names]
  #   num_ids = len(list(set(ids)))
  #   num_query = np.sum(np.array(marks) == 0)
  #   num_gallery = np.sum(np.array(marks) == 1)
  #   num_multi_query = np.sum(np.array(marks) == 2)
  #
  # # Print dataset information
  # print('-' * 40)
  # print('{} {} set'.format(name, part))
  # print('-' * 40)
  # print('NO. Images: {}'.format(len(im_names)))
  # print('NO. IDs: {}'.format(num_ids))
  #
  # try:
  #   print('NO. Query Images: {}'.format(num_query))
  #   print('NO. Gallery Images: {}'.format(num_gallery))
  #   print('NO. Multi-query Images: {}'.format(num_multi_query))
  # except:
  #   pass
  #
  # print('-' * 40)
  #
  # return ret_set
