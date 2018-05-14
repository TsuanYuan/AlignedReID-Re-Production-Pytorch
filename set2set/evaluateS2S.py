"""
evaluation protocol of set to set matching
Quan Yuan
2018-05-14
"""
import os, glob
import numpy
import sklearn.metrics
import logging
import argparse
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',)
mlog = logging.getLogger('myLogger')
level = logging.getLevelName('INFO')
mlog.setLevel(level)

def partition_subsequences(feature_list, descriptor_files, min_size):
    n = len(feature_list)
    if n < min_size*2:
        return None
    seq_ends = [min_size, min_size*3, min_size*6, min_size*10, min_size*15, min_size*21, min_size*28]  # e.g. when min_size=4, lengths are 4, 8, 12, 16 and the rest
    feature_seq_list = []
    file_list = []
    seq_start = 0
    for se in seq_ends:
        if n-se>=min_size:
            feature_seq = feature_list[seq_start:se]
            file_seq = descriptor_files[seq_start:se]
            seq_start = se
            feature_seq_list.append(feature_seq)
            file_list.append(file_seq)
    feature_seq_list.append(feature_list[seq_start:])
    file_list.append(descriptor_files[seq_start:])

    return feature_seq_list, file_list


def load_person_id_descriptors(person_folder, ext):
    desc_files = glob.glob(os.path.join(person_folder, '*.'+ext))
    # load all descriptors
    descriptors = [numpy.fromfile(desc_file) for desc_file in desc_files]
    return descriptors, desc_files


def compute_sequence_matching(descriptors_1, descriptors_2, aggregation_type='min'):
    desc1 = numpy.array(descriptors_1)
    desc2 = numpy.array(descriptors_2)
    if aggregation_type == 'min' or aggregation_type == 'percent_10':
        dist_matrix = sklearn.metrics.pairwise_distances(desc1, desc2, metric="cosine")
        if aggregation_type == 'min':
            return numpy.min(dist_matrix)
        elif aggregation_type == 'percent_10':
            return numpy.percentile(dist_matrix, 10)
        else:
            raise Exception('undefined matching option for crops!')
    else:
        raise Exception('undefined matching method!')


def compute_distance_matrix(feature_seq_list, aggregation_type):
    n = len(feature_seq_list)
    dist_matrix = numpy.ones((n,n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i,j] = compute_sequence_matching(feature_seq_list[i], feature_seq_list[j], aggregation_type)

    return dist_matrix


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

    return auc95, thresholds[idx]


def dump_error_results(file_list, distances, id_pairs, output_path):
    image_size = (256, 128)
    id_pair_set = set()
    for count, id_pair in enumerate(id_pairs):
        p = tuple(sorted(id_pair))
        if p in id_pair_set:
            continue
        else:
            id_pair_set.add(p)

    n_rows = len(id_pair_set)
    images_per_row = 32
    canvas = numpy.zeros((image_size[0]*n_rows*2, image_size[1]*images_per_row, 3), dtype=numpy.uint8)

    for count, p in enumerate(id_pair_set):
        for pid in range(2):
            for imi, file_path in enumerate(file_list[p[pid]]):
                image_path = os.path.splitext(file_path)[0]+'.jpg'
                if os.path.isfile(image_path) and imi < images_per_row:
                    im = cv2.imread(image_path)
                    imr = cv2.resize(im, (image_size[1],image_size[0]))
                    y_start = count*image_size[0]*2+pid*image_size[0]
                    x_start = imi*image_size[1]
                    canvas[y_start:y_start+image_size[0], x_start:x_start+image_size[1],:] = imr
        dist = '%.6f'%distances[count]
        box_color = (0, 255, 0)
        cv2.rectangle(canvas, (0, count*image_size[0]*2), (images_per_row*image_size[1] - 4, (count+1)*image_size[0]*2), box_color, 4)
        cv2.putText(canvas, "d={0}".format(str(dist)), (10, count*image_size[0]*2+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imwrite(output_path, canvas)


def compute_metrics(distance_matrix, person_id_list, file_list, output_folder='/tmp/s2s_results/', file_tag=''):
    person_ids = numpy.array(person_id_list)
    #id_dm = sklearn.metrics.pairwise_distances(person_ids.reshape(1,-1))
    id_dm = numpy.subtract(person_ids, person_ids.reshape(1,-1).transpose())
    same_mask = id_dm == 0
    diff_mask = id_dm != 0
    same_pair_indices = numpy.argwhere(same_mask)
    diff_pair_indices = numpy.argwhere(diff_mask)
    same_distances = distance_matrix[same_mask]
    diff_distances = distance_matrix[diff_mask]
    auc95, dist_th = report_AUC95(same_distances, diff_distances)
    top_error = 20
    highest_same_indices = numpy.argsort(same_distances)[-top_error:]
    lowest_diff_indices = numpy.argsort(diff_distances)[0:top_error]
    highest_same_distances = same_distances[highest_same_indices]
    lowest_diff_distances = diff_distances[lowest_diff_indices]
    top_same_pairs = same_pair_indices[highest_same_indices]
    top_diff_pairs = diff_pair_indices[lowest_diff_indices]

    # dump results
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    canvas_path = os.path.join(output_folder, 'same_pairs_{0}.jpg'.format(file_tag))
    dump_error_results(file_list, highest_same_distances, top_same_pairs, canvas_path)
    mlog.info('same pair top cases are at {0}'.format(canvas_path))
    canvas_path = os.path.join(output_folder, 'diff_pairs_{0}.jpg'.format(file_tag))
    dump_error_results(file_list, lowest_diff_distances, top_diff_pairs, canvas_path)
    mlog.info('diff pair top cases are at {0}'.format(canvas_path))
    return auc95, dist_th


def process(data_folder, ext, min_seq_size, aggregation_type):

    sub_folders = os.listdir(data_folder)
    feature_seq_list, file_seq_list, person_id_list = [], [], []
    for sub_folder in sub_folders:
        if os.path.isdir(os.path.join(data_folder,sub_folder)) and sub_folder.isdigit():
            person_id = int(sub_folder)
            descriptors, desc_files = load_person_id_descriptors(os.path.join(data_folder,sub_folder), ext)
            feature_seqs, file_seqs = partition_subsequences(descriptors, desc_files, min_seq_size)
            person_id_seqs = [person_id]*len(feature_seqs)
            feature_seq_list += feature_seqs
            file_seq_list += file_seqs
            person_id_list += person_id_seqs

    distance_matrix = compute_distance_matrix(feature_seq_list, aggregation_type)
    auc95, dist_th = compute_metrics(distance_matrix, person_id_list, file_seq_list, file_tag=aggregation_type)
    mlog.info('AUC95={0} at dist_th={1} on data set {2} with model extension {3} and compare option {4}'
            .format('%.3f'%auc95, '%.3f'%dist_th, data_folder, ext, aggregation_type))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('test_folder', type=str,
                        help='folder of ground truth crops with computed features')
    parser.add_argument('ext', type=str,
                        help='the extension of feature files')
    parser.add_argument('--min_seq_size', type=int, default=4,
                        help='the min number of crops to in an appearance feature sequence')
    parser.add_argument('--aggregation_type', type=str, default='min',
                        help='the aggregation type')

    args = parser.parse_args()

    process(args.test_folder, args.ext, args.min_seq_size, args.aggregation_type)