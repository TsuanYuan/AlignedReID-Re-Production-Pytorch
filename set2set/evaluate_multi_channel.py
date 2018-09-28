"""
evaluate multi-camera reid accuracies
Quan Yuan
2018-09-29
"""

import os
import sys
import argparse
import logging
import numpy
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from evaluate import feature_compute, metric_compute, misc
from evaluate.metric_compute import Same_Pair_Requirements
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def process(data_folder, model_list, exts, force_compute, dump_folder, ignore_ids, same_dist_requirements):

    sub_folders = os.listdir(data_folder)
    features_per_person, file_seq_list, person_id_list,crops_file_list = [], [], [], []

    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit() and (int(sub_folder) not in ignore_ids):
            descriptors, crop_files = feature_compute.load_descriptor_list(os.path.join(data_folder,sub_folder),model_list, exts, frame_interval=-1, force_compute=force_compute)
            if len(descriptors) > 1:
                features_per_person.append(descriptors)
                crops_file_list.append(crop_files)

    # avoid bias towards person of long tracks
    mean_len = sum([len(crop_files) for crop_files in crops_file_list])/max(1,len(crops_file_list))
    len_limit = int(mean_len*1.5)
    for i, crop_files in enumerate(crops_file_list):
        if len(crop_files) > len_limit:
            crops_file_list[i] = crop_files[:len_limit]
            features_per_person[i] = features_per_person[i][:len_limit]

    same_pair_dist, same_pair_files = metric_compute.compute_same_pair_dist(features_per_person, crops_file_list, same_dist_requirements)
    diff_pair_dist, diff_pair_files = metric_compute.compute_diff_pair_dist(features_per_person, crops_file_list)
    # dump difficult files
    data_tag = os.path.basename(os.path.normpath(data_folder))
    time_tag = str(time.time())
    model_tag = os.path.basename(model_list[0])
    dump_folder_with_tag = os.path.join(dump_folder, data_tag+'_'+model_tag + '_'+time_tag)
    misc.dump_difficult_pair_files(same_pair_dist, same_pair_files, diff_pair_dist, diff_pair_files, output_folder=dump_folder_with_tag)
    # report true postives at different false positives
    same_pair_dist = numpy.array(same_pair_dist)
    diff_pair_dist = numpy.array(diff_pair_dist)
    tpr2, fpr2, th2 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.01)
    tpr3, fpr3, th3 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.001)
    tpr4, fpr4, th4 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.0001)
    tpr5, fpr5, th5 = metric_compute.report_TP_at_FP(same_pair_dist, diff_pair_dist, fp_th=0.00001)

    mlog.info('same_pairs are {0}, diff_pairs are {1}'.format(str(same_pair_dist.size), str(diff_pair_dist.size)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
            .format('%.3f'%tpr2, '%.6f'%th2, '%.5f'%fpr2, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
            .format('%.3f'%tpr3, '%.6f'%th3, '%.5f'%fpr3, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
              .format('%.3f' % tpr4, '%.6f'%th4, '%.5f' % fpr4, data_folder, str(exts)))
    mlog.info('tpr={0}, dist_th={1}, fpr={2} on data {3} with model extension {4}'
              .format('%.3f' % tpr5, '%.6f' % th5, '%.5f' % fpr5, data_folder, str(exts)))
    return tpr2, tpr3, tpr4, tpr5, th2, th3, th4, th4

def process_all(folder, frame_interval, experts, exts, force_compute, dump_folder, ignore_ids):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    tps = []
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        tp3 = process(sub_folder_full,frame_interval, experts, exts, force_compute, dump_folder, ignore_ids)
        tps.append(tp3)
    tps = numpy.array(tps)
    mean_tps = numpy.mean(tps, axis=0)
    row_len = mean_tps.size/2
    mlog.info('average of true positive rates are:   {0}'.format(str(mean_tps[:row_len])))
    mlog.info('corresponding thresholds are:         {0}'.format(str(mean_tps[row_len:])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of ground truth crops with computed features')

    parser.add_argument('experts_file', type=str,
                        help='the file of list of expert model paths')

    parser.add_argument('--frame_interval', type=int, default=-1,
                        help='the num of samples from each ID')

    parser.add_argument('--dump_folder', type=str, default='/tmp/difficult',
                        help='whether to dump tough pairs')

    parser.add_argument('--force_compute', action='store_true', default=False,
                        help='whether to force compute features')

    parser.add_argument('--device_id', type=int, default=0,
                        help='device id to run model')

    parser.add_argument('--single_folder', action='store_true', default=False,
                        help='process only current folder')

    parser.add_argument('--must_different_days', action='store_true', default=False,
                        help='crop attach at top')

    parser.add_argument('--must_same_camera', action='store_true', default=False,
                        help='crop attach at top')

    args = parser.parse_args()
    print 'frame interval={0}'.format(args.frame_interval)

    same_day_requirements = Same_Pair_Requirements(frame_interval=args.frame_interval, must_different_days=args.must_different_days, must_same_camera=args.must_same_camera)
    experts, exts = feature_compute.load_experts(args.experts_file, args.device_id)
    import time
    HEAD_TOP = args.head_top
    if HEAD_TOP:
        print 'put partial head crop at top'
    if len(args.ignore_ids) > 0:
        print 'ignore ids {0}'.format(str(args.ignore_ids))

    start_time = time.time()
    if args.single_folder:
        process(args.test_folder, args.frame_interval, experts, exts, args.force_compute, args.dump_folder, args.ignore_ids)
    else:
        process_all(args.test_folder, args.frame_interval, experts, exts, args.force_compute, args.dump_folder,args.ignore_ids)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'total time = {0}'.format(str(elapsed))
