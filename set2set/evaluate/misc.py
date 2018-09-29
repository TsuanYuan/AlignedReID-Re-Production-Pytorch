"""
extra utils functions for feature evaluation
Quan Yuan
2018-09-29
"""

import numpy
import os
import json
import cv2
import shutil
import feature_compute


def get_filename_for_display(file_path):
    p1, _ = os.path.split(file_path)
    folder_name = os.path.basename(p1)
    bn = os.path.basename(file_path)
    bn, _ = os.path.splitext(bn)
    parts = bn.split('_')
    return parts[-2]+'_'+parts[-1], folder_name


def dump_pair_in_folder(file_pairs, pair_dist, output_path):
    json0 = os.path.splitext(file_pairs[0])[0]+'.json'
    json1 = os.path.splitext(file_pairs[1])[0] + '.json'
    with open(json0, 'r') as fp:
        d0 = json.load(fp)
    with open(json1,'r') as fp:
        d1 = json.load(fp)
    box0 = d0['box'][0:4]
    box1 = d1['box'][0:4]
    import cv2

    im0 = cv2.imread(file_pairs[0])
    im1 = cv2.imread(file_pairs[1])
    im0 = cv2.resize(im0, (256, 512))
    im1 = cv2.resize(im1, (256, 512))
    canvas = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
    canvas[:,:256,:] = im0
    canvas[:,256:,:] = im1

    top_name, folder_name = get_filename_for_display(file_pairs[0])
    cv2.putText(canvas, str(top_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.putText(canvas, str(box0), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    top_name, folder_name = get_filename_for_display(file_pairs[1])
    cv2.putText(canvas, str(top_name), (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.putText(canvas, str(box1), (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.putText(canvas, str(pair_dist), (120, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.imwrite(output_path, canvas)


def plot_error_spatial(canvas, tough_pair_files):
    # plot tough boxes on a frame
    for tough_pair in tough_pair_files:
        for tough_file in tough_pair:
            no_ext, _ = os.path.splitext(tough_file)
            json_file = no_ext+'.json'
            if not os.path.isfile(json_file):
                continue
            with open(json_file, 'r') as fp:
                data = json.load(fp)
                box_br = (data['box'][0] +  data['box'][2], data['box'][1] +  data['box'][3])
                cv2.rectangle(canvas, tuple(data['box'][0:2]), box_br, (0,255,0))
    return canvas


def dump_difficult_pair_files(same_pair_dist, same_pair_files, diff_pair_dist, diff_pair_files, tough_diff_th=0.1, tough_same_th = 0.2,
                              output_folder='/tmp/difficult/',frame_shape=(1920, 1080)):
    same_sort_ids = numpy.argsort(same_pair_dist)
    tough_same_ids = [i for i in same_sort_ids if same_pair_dist[i]>tough_same_th]
    if len(tough_same_ids) < 8:
        tough_num = min(max(int(round(len(same_sort_ids)*0.1)), 32), 128)
        tough_same_ids = same_sort_ids[-tough_num:]
    same_select_files, same_select_dist, same_all_files = [],[],[]
    same_dict = {}
    for id in tough_same_ids:
        p = same_pair_files[id]
        d = same_pair_dist[id]
        pid = feature_compute.decode_wcc_image_name(os.path.basename(p[0]))[3]
        if pid not in same_dict:
            same_dict[pid] = 1
        elif same_dict[pid] >= 3:
            same_all_files.append(p)
            continue
        else:
            same_dict[pid] += 1
        same_select_files.append(p)
        same_select_dist.append(d)

    tough_same_pairs = numpy.array(same_select_files)
    tough_same_dist = numpy.array(same_select_dist)

    canvas = numpy.zeros((frame_shape[1], frame_shape[0], 3))
    canvas = plot_error_spatial(canvas, numpy.array(same_all_files))

    diff_sort_ids = numpy.argsort(diff_pair_dist)
    tough_diff_ids = [i for i in diff_sort_ids if diff_pair_dist[i] < tough_diff_th]
    if len(tough_diff_ids) < 8:
        tough_num = min(max(int(round(len(diff_sort_ids)*0.1)), 32), 128)
        tough_diff_ids = diff_sort_ids[0:tough_num]
    diff_select_files, diff_select_dist, diff_all_files =[], [], []
    diff_dict = {}
    for id in tough_diff_ids:
        p = diff_pair_files[id]
        d = diff_pair_dist[id]
        pid0 = feature_compute.decode_wcc_image_name(os.path.basename(p[0]))[3]
        pid1 = feature_compute.decode_wcc_image_name(os.path.basename(p[1]))[3]
        sorted_pids = tuple(sorted((pid0, pid1)))
        if sorted_pids not in diff_dict:
            diff_dict[sorted_pids] = 1
        elif diff_dict[sorted_pids] >= 3:
            diff_all_files.append(p)
            continue
        else:
            diff_dict[sorted_pids] += 1
        diff_select_files.append(p)
        diff_select_dist.append(d)

    tough_diff_pairs = numpy.array(diff_select_files)
    tough_diff_dist = numpy.array(diff_select_dist)

    canvas = plot_error_spatial(canvas, numpy.array(diff_all_files))
    output_tough_image_file = os.path.join(output_folder,'spatial.jpg')

    if os.path.isdir(output_folder):
        print 'remove existing {0} for difficult pairs output'.format(output_folder)
        shutil.rmtree(output_folder)

    same_folder = os.path.join(output_folder, 'same')
    if not os.path.isdir(same_folder):
        os.makedirs(same_folder)
    count = 0
    for dist, same_pair in zip(tough_same_dist, tough_same_pairs):
        file_path = os.path.join(same_folder, '{0}.jpg'.format(str(count)))
        dump_pair_in_folder(same_pair,dist, file_path)
        count+=1

    diff_folder = os.path.join(output_folder, 'diff')
    if not os.path.isdir(diff_folder):
        os.makedirs(diff_folder)
    count = 0
    for dist, file_pair in zip(tough_diff_dist, tough_diff_pairs):
        file_path = os.path.join(diff_folder, '{0}.jpg'.format(str(count)))
        dump_pair_in_folder(file_pair,dist, file_path)
        count+=1

    cv2.imwrite(output_tough_image_file, canvas)
    print 'output tough image map to {0}'.format(output_tough_image_file)
    print 'difficult pairs were dumped to {0}'.format(output_folder)