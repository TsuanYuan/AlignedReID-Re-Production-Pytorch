"""
evaluation protocol of crop to crop matching
Quan Yuan
2018-05-14
"""
import os, glob
import numpy
import sklearn.metrics
import sklearn.preprocessing
import logging
import argparse
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)


def load_person_id_descriptors(person_folder, ext, sample_size=64):
    desc_files = glob.glob(os.path.join(person_folder, '*.'+ext))
    # load all descriptors
    interval = max(len(desc_files)/sample_size, 1)
    descriptors = [numpy.fromfile(desc_file, dtype=numpy.float32) for i, desc_file in enumerate(desc_files) if i%interval==0]
    return descriptors, desc_files


def softmax(x, theta=1.0):
    ps = numpy.exp(x * theta)
    ps /= numpy.sum(ps+1e-8)
    return ps


def compute_distance_matrix(feature_list):
    feature_arr = numpy.array(feature_list)
    distance_matrix = sklearn.metrics.pairwise.cosine_distances(feature_arr)
    return distance_matrix


def report_MeanAP_over_ids(distance_matrix, id_matrix):
    aps = []
    n = distance_matrix.shape[0]
    scores = 1-distance_matrix
    same_mask = id_matrix == 0
    diff_mask = id_matrix != 0
    for i in range(n):
        score_row = scores[i,:]
        label_row = same_mask[i,:]
        ap = sklearn.metrics.average_precision_score(label_row, score_row)
        aps.append(ap)
    mean_ap = numpy.mean(numpy.array(aps))
    return mean_ap


def report_AUC95(same_distances, diff_distances):
    # AUC with true negative rate >= 95
    n_same = same_distances.size
    n_diff = diff_distances.size
    scores = 1-numpy.concatenate((same_distances, diff_distances))
    labels = numpy.concatenate((numpy.ones(n_same), -numpy.ones(n_diff)))
    fpr, tpr, thresholds =sklearn.metrics.roc_curve(labels, scores)
    fp_th = 0.05
    idx = numpy.argmax(fpr > fp_th)  # fp lower than 0.05 for auc 95
    if idx == 0:  # no points with fpr<=0.05
        return 0
    fpr005 = fpr[0:idx]
    tpr005 = tpr[0:idx]
    auc95 = sklearn.metrics.auc(fpr005, tpr005)/fp_th

    return auc95, 1-thresholds[idx]


def compute_metrics(distance_matrix, person_id_list, file_list, output_folder='/tmp/s2s_results/', file_tag=''):
    person_ids = numpy.array(person_id_list)

    id_dm = numpy.subtract(person_ids, person_ids.reshape(1,-1).transpose())
    same_mask = id_dm == 0
    diff_mask = id_dm != 0
    same_pair_indices = numpy.argwhere(same_mask)
    diff_pair_indices = numpy.argwhere(diff_mask)
    same_distances = distance_matrix[same_mask]
    diff_distances = distance_matrix[diff_mask]
    auc95, dist_th = report_AUC95(same_distances, diff_distances)
    mAP = report_MeanAP_over_ids(distance_matrix, id_dm)
    # top_error = 20
    # highest_same_indices = numpy.argsort(same_distances)[-top_error:]
    # lowest_diff_indices = numpy.argsort(diff_distances)[0:top_error]
    # highest_same_distances = same_distances[highest_same_indices]
    # lowest_diff_distances = diff_distances[lowest_diff_indices]
    # top_same_pairs = same_pair_indices[highest_same_indices]
    # top_diff_pairs = diff_pair_indices[lowest_diff_indices]

    return auc95, dist_th, mAP


def process(data_folder, ext, sample_size):

    sub_folders = os.listdir(data_folder)
    feature_list, file_seq_list, person_id_list = [], [], []
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit():
            person_id = int(sub_folder)
            descriptors, desc_files = load_person_id_descriptors(os.path.join(data_folder,sub_folder), ext, sample_size=sample_size)
            person_id_seqs = [person_id]*len(descriptors)
            feature_list += descriptors
            person_id_list += person_id_seqs

    _, tail = os.path.split(data_folder)
    distance_matrix = compute_distance_matrix(feature_list)
    auc95, dist_th,mAP = compute_metrics(distance_matrix, person_id_list, file_seq_list, file_tag=tail)
    mlog.info('AUC95={0} at dist_th={1}, mAP={2} on data set {3} with model extension {4}'
            .format('%.3f'%auc95, '%.6f'%dist_th, '%.3f'%mAP, data_folder, ext))


def process_all(folder, ext, sample_size):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        process(sub_folder_full, ext, sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of ground truth crops with computed features')
    parser.add_argument('ext', type=str,
                        help='the extension of feature files')
    parser.add_argument('--sample_size', type=int, default=16,
                        help='the num of samples from each ID')

    args = parser.parse_args()

    process_all(args.test_folder, args.ext, args.sample_size)
