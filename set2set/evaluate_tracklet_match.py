"""
tracklet to tracklet matching
"""

"""
evaluate tracklet merge roc
Quan Yuan
2018-07-31
"""

import argparse
import cv2
import os, glob, shutil
import json, pickle
import numpy
import sklearn.metrics
import pickle
from evaluate import tracklet_utils

def load_features(folder, exts):
    features = {}
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    for ext in exts:
        features[ext] = {}
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        if sub_folder.isdigit():
            track_index = int(sub_folder)
        else:
            print 'skipped non digit subfolder {0}'.format(sub_folder)
            continue
        for ext in exts:
            feature_files = glob.glob(os.path.join(sub_folder_full, '*.'+ext))
            feature_rows = []
            for feature_file in feature_files:
                feature_rows.append(numpy.squeeze(numpy.fromfile(feature_file, dtype=numpy.float32)))
            feature_rows = numpy.array(feature_rows)
            features[ext][track_index] = feature_rows

    return features

def update_tracklet_with_path(folder, tracklet):
    updated_tracklet = []
    for frame_id, crop_file in tracklet:
        parts = crop_file.split('/')
        updated_crop_file = os.path.join(folder, parts[-2], parts[-1])
        updated_tracklet.append((frame_id, updated_crop_file))
    return updated_tracklet

def load_track_ground_truth(folder, track_gt_file):
    tid_2_pid = {}
    tid_count = 0
    tracklets_data = {}
    with open(track_gt_file, 'r') as fp:
        d = pickle.load(fp)
    for k in d:
        tracklets = d[k]
        for tracklet in tracklets:
            updated_tracklet = update_tracklet_with_path(folder, tracklet)
            tracklets_data[tid_count] = updated_tracklet
            tid_2_pid[tid_count] = int(k)
            tid_count += 1
    return tracklets_data, tid_2_pid


def get_track_representations(tracklets_data, feature_extractor):
    """
    representation of each tracklet before comparison, e.g., one feature vector per track, key feature vector per track, etc.
    :param tracklets_data:
    :return: tracklets_representation
    """
    tracklets_representation = {}
    for track_id in tracklets_data:
        images = [cv2.cvtColor(cv2.imread(d[1]), cv2.COLOR_BGR2RGB) for d in tracklets_data[track_id]]
        tracklets_representation[track_id] = feature_extractor(images)

    return tracklets_representation


def thresholdAt999(fpr,tpr, thresholds, fp_th=0.001, tp_th=0.999):
    idx = numpy.argmax(fpr > fp_th)  # fp lower than fp_th
    if idx == 0:  # no points with fpr<=0.05
        thTN = 0
    else:
        thTN = 1 - thresholds[idx]

    idx = numpy.argmax(tpr > tp_th)
    if idx >= len(tpr)-1:  # no points with fpr<=0.05
        thTP = 1.0
    else:
        thTP = 1-thresholds[idx]
    return thTN, thTP


def evaluate_tracks(track_features, tid_2_pid, evaluator):
    distance_dict = {}
    tids = track_features.keys()
    distances = []
    track_pairs = []
    labels = []
    for i, tid in enumerate(tids):
        for tid2 in tids[i + 1:]:
            if (tid not in tid_2_pid) or (tid2 not in tid_2_pid):
                continue
            d = evaluator(track_features[tid], track_features[tid2])
            if tid not in distance_dict:
                distance_dict[tid] = {}
            distance_dict[tid][tid2] = d
            distances.append(d)
            track_pairs.append((tid, tid2))
            if tid_2_pid[tid] == tid_2_pid[tid2]:
                labels.append(1)
            else:
                labels.append(0)

    return distances, labels, track_pairs

def evaluate_roc(features, tid_2_pid, track_compare):
    distance_dict = {}
    tids = features.keys()
    distances = []
    track_pairs = []
    labels = []
    for i, tid in enumerate(tids):
        for tid2 in tids[i+1:]:
            if (tid not in tid_2_pid) or (tid2 not in tid_2_pid):
                continue
            d = track_compare(features[tid], features[tid2])
            if tid not in distance_dict:
                distance_dict[tid] = {}
            distance_dict[tid][tid2] = d
            distances.append(d)
            track_pairs.append((tid, tid2))
            if tid_2_pid[tid] == tid_2_pid[tid2]:
                labels.append(1)
            else:
                labels.append(0)
    scores = 1-numpy.array(distances)
    labels = numpy.array(labels)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores)
    thTN999, thTP999 = thresholdAt999(fpr, tpr, thresholds)
    thTN99, thTP99 = thresholdAt999(fpr, tpr, thresholds, fp_th=0.01, tp_th=0.99)
    thTN95, thTP95 = thresholdAt999(fpr, tpr, thresholds, fp_th=0.05, tp_th=0.95)
    return fpr, tpr, thTN999, thTP999,thTN99, thTP99,thTN95, thTP95


def evaluate_merges(track_features, tid_2_pid, output_curve_file):
    curves = {}
    for option_key in track_features.keys():
        track_comparison = tracklet_utils.TrackletComparison(option_key)
        fpr, tpr, thTN999, thTP999,thTN99, thTP99,thTN95, thTP95 = evaluate_roc(track_features[option_key], tid_2_pid, track_comparison)
        curves[option_key] = {}
        curves[option_key]['fpr'] = fpr
        curves[option_key]['tpr'] = tpr
        curves[option_key]['thTP999'] = thTP999
        curves[option_key]['thTN999'] = thTN999
        curves[option_key]['thTP99'] = thTP99
        curves[option_key]['thTN99'] = thTN99
        curves[option_key]['thTP95'] = thTP95
        curves[option_key]['thTN95'] = thTN95
        print "0.999 true postive threshold of {0} is at {1}".format(str(option_key), str(thTP999))
        print "0.999 true negative threshold of {0} is at {1}".format(str(option_key), str(thTN999))
        print "0.99 true postive threshold of {0} is at {1}".format(str(option_key), str(thTP99))
        print "0.99 true negative threshold of {0} is at {1}".format(str(option_key), str(thTN99))
        print "0.95 true postive threshold of {0} is at {1}".format(str(option_key), str(thTP95))
        print "0.95 true negative threshold of {0} is at {1}".format(str(option_key), str(thTN95))
    with open(output_curve_file, 'wb') as handle:
        pickle.dump(curves, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print "output track roc curves to {0}".format(output_curve_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='top folder of tracklet crops')

    parser.add_argument('model_list_file', type=str,
                        help='path of models to compare')

    parser.add_argument('--gpu_ids', nargs='+', type=int, help="gpu ids to use")

    parser.add_argument('--output_file', type=str, default='/tmp/track_merge_curves.pkl',
                        help='output of track merge roc')

    args = parser.parse_args()
    print('parameters:')
    print('  data_folder={0}'.format(args.folder))
    print('  aggregate={}'.format(args.aggregate))

    track_ground_truth_file = os.path.join(args.folder, 'tracklets.pkl')
    tracklet_data, tid2pid = load_track_ground_truth(args.folder, track_ground_truth_file)

    model_files, exts, options = [], [], []
    with open(args.model_list_file, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            model_files.append(parts[0])
            exts.append(parts[1])
            options.append(parts[2])

    tracklets_by_options = {}
    for model_file, ext, option in zip(model_files, exts, options):
        k = ext+'_'+option
        track_feature_file = os.path.join(args.folder, 'track_features.'+k)
        if os.path.isfile(track_feature_file):
            print 'track feature file {} exist. load existing features'.format(track_feature_file)
            with open(track_feature_file, 'rb') as fp:
                tracklet_representations = pickle.load(fp)
        else:
            feature_extractor = tracklet_utils.FeatureExtractor(model_path=args.model_path, device_ids=args.gpu_ids,
                                                                aggregate=option)
            tracklets_by_options[k] = get_track_representations(tracklet_data, feature_extractor)
            with open(track_feature_file, 'wb') as fp:
                pickle.dump(tracklets_by_options[k], fp, pickle.HIGHEST_PROTOCOL)

    evaluate_merges(tracklets_by_options, tid2pid, args.output_file)
