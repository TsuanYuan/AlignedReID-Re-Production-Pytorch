"""
compute metrics for reid
Quan Yuan
2018-09-29
"""
from collections import namedtuple
import os
import numpy
from misc import decode_wcc_image_name
import sklearn.metrics.pairwise as pairwise
import sklearn.metrics

Same_Pair_Requirements = namedtuple("Same_Pair_Requirements", ['min_frame_interval', 'max_frame_interval','must_different_days', 'must_same_day', 'must_same_camera',
                                                               'must_diff_camera','must_same_video', 'must_diff_video', 'same_sample_size'])

def make_string_matrix_from_arr(string_arr, k):
    return numpy.tile(string_arr.reshape((string_arr.size, 1)), [1, k])


def string_distance_array(string_arr_0, string_arr_1):
    # boolean compare of two one d string array in a distance matrix
    matrix_0 = make_string_matrix_from_arr(string_arr_0, string_arr_1.size)
    matrix_1 = make_string_matrix_from_arr(string_arr_1, string_arr_0.size).transpose()
    string_dist = matrix_0==matrix_1
    return string_dist


def compute_same_pair_dist_per_person(features, crop_files, requirements):
    days = numpy.array([decode_wcc_image_name(os.path.basename(file_name))[1] for file_name in crop_files ])
    video_times = numpy.array([decode_wcc_image_name(os.path.basename(file_name))[2] for file_name in crop_files ])
    frame_indices = numpy.array([decode_wcc_image_name(os.path.basename(file_name))[-1] for file_name in crop_files ])
    camera_ids = numpy.array([decode_wcc_image_name(os.path.basename(file_name))[0] for file_name in crop_files ])

    # assume cosine distance
    features_dist = pairwise.cosine_distances(features)

    # boolean matrix item ids from requirements
    satisfied = numpy.ones(features_dist.shape, dtype=numpy.uint8)
    # remove upper triangular comparison including diagonal
    upper_ids = numpy.tril_indices(features_dist.shape[0])
    satisfied[upper_ids] = 0
    satisfied = satisfied > 0
    n = days.size

    assert (requirements.must_diff_camera and requirements.must_same_camera) == False
    assert (requirements.must_different_days and requirements.must_same_day) == False

    same_camera = string_distance_array(camera_ids, camera_ids)
    same_video_times = string_distance_array(video_times, video_times)
    same_days = pairwise.euclidean_distances(days.reshape((n, 1))) == 0

    if requirements.must_same_camera:
        satisfied = numpy.logical_and(satisfied, same_camera)
    elif requirements.must_diff_camera:
        satisfied = numpy.logical_and(satisfied, numpy.logical_not(same_camera))

    if requirements.must_same_video:
        satisfied = numpy.logical_and(satisfied, numpy.logical_and(same_camera, same_video_times))
    elif requirements.must_diff_video:
        satisfied = numpy.logical_not(numpy.logical_and(same_camera, same_video_times))

    if requirements.must_same_day:
        satisfied = numpy.logical_and(satisfied, same_days)
    elif requirements.must_different_days:
        satisfied = numpy.logical_and(satisfied, numpy.logical_not(same_days))

    # frame interval > 0 could be different days, different video_time, different camera or frame diff > frame_interval
    diff_days = numpy.logical_not(same_days)
    diff_videos = numpy.logical_not(numpy.logical_and(same_camera, same_video_times))
    frame_diff =  pairwise.euclidean_distances(frame_indices.reshape((n, 1)))
    frame_interval_min_satisfied = frame_diff >= requirements.min_frame_interval
    frame_interval_max_satisfied = frame_diff <= requirements.max_frame_interval
    frame_requirement = numpy.logical_or(numpy.logical_or(diff_days, diff_videos), numpy.logical_or(frame_interval_min_satisfied, frame_interval_max_satisfied))
    satisfied = numpy.logical_and(satisfied, frame_requirement)

    satisfied_dist = features_dist[satisfied]
    crops_file_array = numpy.tile(numpy.array(crop_files).reshape((n, 1)), [1, n])
    crops_file_array_t = numpy.tile(numpy.array(crop_files).reshape((1, n)), [n, 1])
    satisfied_file_pairs = zip(crops_file_array[satisfied].tolist(), crops_file_array_t[satisfied].tolist())
    return satisfied_dist, satisfied_file_pairs


def compute_same_pair_dist(features_per_person, crop_files, requirements):
    """
    :param features_per_person: list of features, each item is an array of features from the same person.
    :param crop_files: list of arrays of corresponding file path of each feature
    :param requirements: the Same_Pair_Requirements tuple that encodes matching requirements
    :return: distances between same pairs that satisfies the requirements
    """
    dists_all = None
    file_pairs_all = []
    for features, file_paths in zip(features_per_person, crop_files):
        dists, file_pairs = compute_same_pair_dist_per_person(features, file_paths, requirements)
        file_pairs_all += file_pairs
        if dists_all is None:
            dists_all = dists
        else:
            dists_all = numpy.concatenate((dists_all, dists))
    return dists_all, file_pairs_all


def sub_sample_feature_files(files_dict, features_dict, features_per_person, crop_files_per_person, j, sub_sample_size):
    if j in features_dict:
        features_j = features_dict[j]
        files_j = files_dict[j]
    else:
        sub_ids = numpy.round(numpy.linspace(0, features_per_person[j].shape[0] - 1, sub_sample_size)).astype(int)
        features_j = features_per_person[j][sub_ids, :]
        files_j = numpy.array(crop_files_per_person[j])[sub_ids]
    return features_j, files_j


def compute_diff_pair_dist(features_per_person, crop_files_per_person, sub_sample_size=16, folder_sample_interval=1):
    """
    :param features_per_person: list of features, each item is an array of features from the same person.
    :param crop_files_per_person: list of arrays of corresponding file path of each feature
    :param sub_sample_size: number of samples per person to compute negative pairs
    :return: distances between diff pairs
    """
    dists_all = None
    file_pairs_all = []
    n = len(features_per_person)
    files_dict = {}
    features_dict = {}
    for i in range(n):
        features_i, files_i = sub_sample_feature_files(files_dict, features_dict, features_per_person, crop_files_per_person, i, sub_sample_size)
        for j in range(i+1, n,folder_sample_interval):
            features_j, files_j = sub_sample_feature_files(files_dict, features_dict, features_per_person, crop_files_per_person, j, sub_sample_size)
            dists = pairwise.cosine_distances(features_i, features_j).ravel()
            crop_files_matrix_i = make_string_matrix_from_arr(files_i, files_j.size).ravel()
            crop_files_matrix_j = make_string_matrix_from_arr(files_j, files_i.size).transpose().ravel()
            if dists_all is None:
                dists_all = dists
            else:
                dists_all = numpy.concatenate((dists_all, dists))
            file_pairs = zip(crop_files_matrix_i.tolist(), crop_files_matrix_j.tolist())
            file_pairs_all += file_pairs
    return dists_all, file_pairs_all


def report_TP_at_FP(same_distances, diff_distances, fp_th=0.001):
    # report true positive rate at a false positive rate
    n_same = same_distances.size
    n_diff = diff_distances.size
    scores = 1-numpy.concatenate((same_distances, diff_distances))
    labels = numpy.concatenate((numpy.ones(n_same), -numpy.ones(n_diff)))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, drop_intermediate=False)
    idx = numpy.argmin(numpy.abs(fpr - fp_th))
    fpr = fpr[idx]
    tpr = tpr[idx]
    th = 1-thresholds[idx]

    return tpr, fpr, th