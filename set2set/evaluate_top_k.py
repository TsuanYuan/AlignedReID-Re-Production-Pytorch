"""
evaluate appearance model as classifier top k
Quan Yuan
2018-10-20
"""

import os
import numpy
import argparse
import sklearn.metrics
from evaluate import feature_compute


def process_folder(data_folder, model, force_compute, ext, sample_size, batch_max, down_sample=False):
    sub_folders = os.listdir(data_folder)
    features_per_person, file_seq_list, person_id_list, crops_file_list = [], [], [], []

    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder, sub_folder)) and sub_folder.isdigit() :
            descriptors, crop_files = feature_compute.load_descriptor_list(os.path.join(data_folder, sub_folder), model, ext,
                                                                           force_compute=force_compute, batch_max=batch_max,
                                                                           load_keypoints=False, keypoints_score_th=0,
                                                                           same_sampel_size=sample_size)
            if len(descriptors) > 1:
                features_per_person.append(descriptors)
                crops_file_list.append(crop_files)
                if (len(crops_file_list) + 1) % 100 == 0:
                    print "finished {} pid folders".format(str(len(crops_file_list) + 1))
    # avoid bias towards person of long tracks
    if down_sample:
        mean_len = sum([len(crop_files) for crop_files in crops_file_list]) / max(1, len(crops_file_list))
        len_limit = int(mean_len * 1.5)
        for i, crop_files in enumerate(crops_file_list):
            if len(crop_files) > len_limit:
                sample_ids = numpy.round(numpy.linspace(0, len(crop_files) - 1, len_limit)).astype(int)
                crops_file_list[i] = numpy.array(crop_files)[sample_ids]
                features_per_person[i] = numpy.array(features_per_person[i])[sample_ids, :]
            else:
                crops_file_list[i] = numpy.array(crop_files)
                features_per_person[i] = numpy.array(features_per_person[i])
        return
    print "finish feature computing on {}".format(data_folder)
    return features_per_person, crops_file_list

def compute_top_k(tracklet_features, tracklet_to_pid, train_features, match_option):
    tracklet_ids = tracklet_features.keys()
    tracklet_to_pid_dists = {}
    for tracklet_id in tracklet_ids:
        tracklet_feature = numpy.array(tracklet_features[tracklet_id])
        tracklet_to_pid_dists[tracklet_id] = {}
        for pid in train_features:
            pid_feature = numpy.array(train_features[pid])
            if match_option == 'ten_percent':
                dist_matrix = sklearn.metrics.pairwise.pairwise_distances(pid_feature, tracklet_feature, metric='cosine')
                tracklet_to_pid_dists[tracklet_id][pid] = numpy.percentile(dist_matrix, 10)
            elif match_option == 'median':
                tracklet_median = feature_compute.median_feature(tracklet_feature)
                pid_median = feature_compute.median_feature(pid_feature)
                tracklet_to_pid_dists[tracklet_id][pid] =  1 - numpy.dot(tracklet_median, pid_median)

    top1, top5 = 0, 0
    for tracklet_id in tracklet_to_pid_dists:
        pid_list = numpy.array(tracklet_to_pid_dists[tracklet_id].keys())
        dist_list = numpy.array(tracklet_to_pid_dists[tracklet_id].values())
        sort_ids = numpy.argsort(dist_list)
        if pid_list[sort_ids[0]] == tracklet_to_pid[tracklet_id]:
            top1 += 1
        if tracklet_to_pid[tracklet_id] in pid_list[sort_ids[0:5]].tolist():
            top5 += 1
    n = len(tracklet_to_pid_dists)
    top1 /= float(n)
    top5 /= float(n)
    print "top 1 of tracklet pid matching with option {} is {}".format(match_option, str(top1))
    print "top 5 of tracklet pid matching with option {} is {}".format(match_option, str(top5))


def tracklet_train_features(train_features, train_files):
    tracklet_features = {}
    for train_feature_per_person, train_file_per_person in zip(train_features, train_files):
        for train_file, train_feature in zip(train_file_per_person, train_feature_per_person):
            # decode ch00002_20181006161838_474_328.jpg
            pid = int(train_file.split('/')[-2])
            if pid not in tracklet_features:
                tracklet_features[pid] = []
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

    parser.add_argument('--match_option', type=str, default='ten_percent',
                        help='options to get top k result')

    args = parser.parse_args()

    model = feature_compute.AppearanceModelForward(args.model_path, device_ids=args.device_ids)

    test_features, test_files = process_folder(args.test_folder, model, args.force_compute, args.ext, -1, args.batch_max)
    train_features, train_files = process_folder(args.train_folder, model, args.force_compute, args.ext, -1, args.batch_max)

    train_pid_features = tracklet_train_features(train_features, train_files)
    tracklet_features, tracklet_to_pid = tracklet_test_features(test_features, test_files)
    compute_top_k(tracklet_features, tracklet_to_pid, train_pid_features, args.match_option)
