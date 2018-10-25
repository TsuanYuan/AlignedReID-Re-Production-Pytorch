"""
evaluate appearance model as classifier top k
Quan Yuan
2018-10-20
"""

import os
import glob
import numpy
import argparse
import sklearn.metrics
import time
from evaluate import feature_compute
import collections

def decode_track_id(image_file):
    # decode ch00002_20181006161838_474_328.jpg
    file_only = os.path.basename(image_file)
    parts = file_only.split('_')
    tracklet_id = parts[0] + '_' + parts[1] + '_' + parts[2]
    return tracklet_id

def process_folder(data_folder, model, force_compute, ext, sample_size, batch_max, down_sample=False):
    sub_folders = os.listdir(data_folder)
    features_per_person, file_seq_list, person_id_list, crops_file_list = [], [], [], []

    for sub_folder in sub_folders:
        pid_folder = os.path.join(data_folder, sub_folder)
        if os.path.isdir(pid_folder) and sub_folder.isdigit() :
            descriptors, crop_files = feature_compute.load_descriptor_list(pid_folder, model, ext,
                                                                           force_compute=force_compute, batch_max=batch_max,
                                                                           load_keypoints=False, keypoints_score_th=0,
                                                                           same_sampel_size=sample_size)
            if len(descriptors) > 1:
                features_per_person.append(descriptors)
                crops_file_list.append(crop_files)
                if (len(crops_file_list) + 1) % 100 == 0:
                    print "finished {} pid folders".format(str(len(crops_file_list) + 1))

    print "finish feature computing on {}".format(data_folder)
    return features_per_person, crops_file_list


def group_tracklet_files(data_folder):
    tracklet_crops = collections.defaultdict(list)
    crop_files = glob.glob(os.path.join(data_folder, '*.jpg'))
    for crop_file in crop_files:
        tracklet_id = decode_track_id(crop_file)
        tracklet_crops[tracklet_id].append(crop_file)
    return tracklet_crops


def process_test(data_folder, model, force_compute, ext, sample_size, batch_max):
    sub_folders = os.listdir(data_folder)
    features_per_track, crops_file_list = collections.defaultdict(list), collections.defaultdict(list)
    tracklet_to_pid = {}
    tracklet_length = {}
    for sub_folder in sub_folders:
        pid_folder = os.path.join(data_folder, sub_folder)
        pid = int(sub_folder)
        if os.path.isdir(pid_folder) and sub_folder.isdigit() :
            tracklet_files = group_tracklet_files(pid_folder)

            for track_id in tracklet_files:
                tracklet_length[track_id] = len(tracklet_files[track_id])
                descriptors, crop_files = feature_compute.load_descriptor_list_on_files(tracklet_files[track_id], model, ext,
                                                                           force_compute=force_compute, batch_max=batch_max,
                                                                           keypoint_file=None, keypoints_score_th=0,
                                                                           same_sampel_size=sample_size, w_h_quality_th=100, min_crop_h=1)
                if len(crop_files) > 1:
                    features_per_track[track_id].append(descriptors)
                    crops_file_list[track_id].append(crop_files)
                    if (len(crops_file_list) + 1) % 100 == 0:
                        print "finished {} track ids".format(str(len(crops_file_list) + 1))
                if track_id not in tracklet_to_pid:
                    tracklet_to_pid[track_id] = pid
    print "finish feature computing on {}".format(data_folder)
    return features_per_track, crops_file_list, tracklet_to_pid, tracklet_length


def compute_top_k(tracklet_features, tracklet_to_pid, train_features, match_option, tracklet_length, dump_folder='/tmp/online/'):
    tracklet_ids = tracklet_features.keys()
    tracklet_to_pid_dists = {}
    for tracklet_id in tracklet_ids:
        tracklet_feature = numpy.squeeze(numpy.array(tracklet_features[tracklet_id]))
        tracklet_to_pid_dists[tracklet_id] = {}
        for pid in train_features:
            pid_feature = numpy.array(train_features[pid])
            if len(pid_feature)<=0:
                continue
            if match_option == 'ten_percent':
                dist_matrix = sklearn.metrics.pairwise.pairwise_distances(pid_feature, tracklet_feature, metric='cosine')
                tracklet_to_pid_dists[tracklet_id][pid] = numpy.percentile(dist_matrix, 10)
            elif match_option == 'median':
                tracklet_median = feature_compute.median_feature(numpy.squeeze(tracklet_feature))
                pid_median = feature_compute.median_feature(pid_feature)
                tracklet_to_pid_dists[tracklet_id][pid] =  1 - numpy.dot(tracklet_median, pid_median)

    top1, top5, top1_wl, top5_wl, total_len = 0, 0, 0, 0, 0
    top1_missed_list = []
    for tracklet_id in tracklet_to_pid_dists:
        pid_list = numpy.array(tracklet_to_pid_dists[tracklet_id].keys())
        dist_list = numpy.array(tracklet_to_pid_dists[tracklet_id].values())
        sort_ids = numpy.argsort(dist_list)
        if pid_list[sort_ids[0]] == tracklet_to_pid[tracklet_id]:
            top1 += 1
            top1_wl += tracklet_length[tracklet_id]
        else:
            top1_missed_list.append((tracklet_id, pid_list[sort_ids[0]], tracklet_to_pid[tracklet_id]))
        total_len += tracklet_length[tracklet_id]
        if tracklet_to_pid[tracklet_id] in pid_list[sort_ids[0:5]].tolist():
            top5 += 1
            top5_wl += tracklet_length[tracklet_id]

    n = len(tracklet_to_pid_dists)
    top1 /= float(n)
    top5 /= float(n)
    top1_wl /= float(total_len)
    top5_wl /= float(total_len)

    print "top 1 of tracklet pid matching with option {} is {}".format(match_option, str(top1))
    print "top 5 of tracklet pid matching with option {} is {}".format(match_option, str(top5))
    print "top 1 of crop count weighted pid matching with option {} is {}".format(match_option, str(top1_wl))
    print "top 5 of crop count weighted pid matching with option {} is {}".format(match_option, str(top5_wl))

    if os.path.isdir(dump_folder) == False:
        os.makedirs(dump_folder)
    dump_file = os.path.join(dump_folder, 'missed_top1.txt')
    with open(dump_file, 'w') as fp:
        for missed_tracklet, pid_output, ground_truth_pid in top1_missed_list:
            fp.write('{} {} {}\n'.format(missed_tracklet, str(pid_output), str(ground_truth_pid)))
    print "missed top1 tracklets were dumped to {}".format(dump_file)

def tracklet_train_features(train_features, train_files):
    tracklet_features = collections.defaultdict(list)
    for train_feature_per_person, train_file_per_person in zip(train_features, train_files):
        for train_file, train_feature in zip(train_file_per_person, train_feature_per_person):
            # decode ch00002_20181006161838_474_328.jpg
            pid = int(train_file.split('/')[-2])
            if len(train_feature.shape) > 0:
                tracklet_features[pid].append(train_feature)
    return tracklet_features

def tracklet_test_features(test_features, test_files):
    tracklet_features = {}
    tracklet_to_pid = {}
    for test_feature_per_person, test_file_per_person in zip(test_features, test_files):
        for test_file, test_feature in zip(test_file_per_person, test_feature_per_person):
            # decode ch00002_20181006161838_474_328.jpg
            file_only = os.path.basename(test_file)
            pid = int(test_file.split('/')[-2])
            parts = file_only.split('_')
            tracklet_id = parts[0]+'_'+parts[1]+'_'+parts[2]
            if tracklet_id not in tracklet_features:
                tracklet_features[tracklet_id] = []
                tracklet_to_pid[tracklet_id] = pid
            if pid != tracklet_to_pid[tracklet_id]:
                print 'tracket {} mapped to two different pids {} and {}'.format(str(tracklet_id), str(pid), str(tracklet_to_pid[tracklet_id]))
            tracklet_features[tracklet_id].append(test_feature)
    return tracklet_features, tracklet_to_pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of test crops with computed features')

    parser.add_argument('train_folder', type=str,
                        help='folder of train crops with computed features')

    parser.add_argument('model_path', type=str,
                        help='the model path')

    parser.add_argument('--ext', type=str, default='dsc',
                        help='the ext to dump descirptor files')

    parser.add_argument('--force_compute', action='store_true', default=False,
                        help='whether to force compute features')

    parser.add_argument('--device_ids', nargs='+', type=int, default=(0, ),
                        help='device ids to run model')

    parser.add_argument('--batch_max', type=int, default=128,
                        help='batch size to compute reid features')

    parser.add_argument('--sample_size', type=int, default=32,
                        help='down sample tracklet size to compute reid features')

    parser.add_argument('--match_option', type=str, default='ten_percent',
                        help='options to get top k result')

    args = parser.parse_args()
    print "options are ext={}, force_compute={}, batch_max={}, sample_size={}, match_option={}".format(args.ext, str(args.force_compute), str(args.batch_max),
                                                                                                       str(args.sample_size), args.match_option)
    start_time = time.time()
    model = feature_compute.AppearanceModelForward(args.model_path, device_ids=args.device_ids)

    test_features, test_files, tracklet_to_pid, tracklet_length = process_test(args.test_folder, model, args.force_compute, args.ext, args.sample_size, args.batch_max)
    train_features, train_files = process_folder(args.train_folder, model, args.force_compute, args.ext, args.sample_size, args.batch_max)

    train_pid_features = tracklet_train_features(train_features, train_files)
    #tracklet_features, tracklet_to_pid = tracklet_test_features(test_features, test_files)
    compute_top_k(test_features, tracklet_to_pid, train_pid_features, args.match_option, tracklet_length)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'total time = {0}'.format(str(elapsed))
