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
import pickle


def decode_wcc_image_name(image_name):
    # decode ch00002_20180816102633_00005504_00052119.jpg
    # or ch02001_20180917143702_pano_00034574_00001536
    image_base, _ = os.path.splitext(image_name)
    parts = image_base.split('_')
    channel = parts[0]
    date = parts[1][:8]
    video_time = parts[1][8:]
    if video_time.isdigit():
        video_time = int(video_time)
    pid = parts[-2]
    frame_id = parts[-1]
    return channel, int(date), video_time, int(pid), int(frame_id)

def get_filename_for_display(file_path):
    p1, _ = os.path.split(file_path)
    folder_name = os.path.basename(p1)
    bn = os.path.basename(file_path)
    bn, _ = os.path.splitext(bn)
    parts = bn.split('_')
    return parts[-2]+'_'+parts[-1], folder_name


def plot_key_points(im_rgb, xs, ys, radius=4, put_index=True):
    color = (0, 255, 255)
    count = 0
    for x, y in zip(xs, ys):
        cv2.circle(im_rgb, (int(x), int(y)),
                   radius, color, thickness=2)
        if put_index:
            cv2.putText(im_rgb, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), 1)
        count += 1
    return im_rgb


def keypoints_quality(normalized_keypoints):
    mins = numpy.min(normalized_keypoints, axis=0)[0:2]
    maxs = numpy.max(normalized_keypoints, axis=0)[0:2]
    occupy = (maxs[1] - mins[1])*(maxs[0] - mins[0])
    mean_score = numpy.mean(normalized_keypoints[:, 3])
    score = mean_score*10+occupy
    return score


def dump_pair_in_folder(file_pairs, pair_dist, output_path, load_keypoints=True):
    # json0 = os.path.splitext(file_pairs[0])[0]+'.json'
    # json1 = os.path.splitext(file_pairs[1])[0] + '.json'
    # with open(json0, 'r') as fp:
    #     d0 = json.load(fp)
    # with open(json1,'r') as fp:
    #     d1 = json.load(fp)
    #box0 = d0['box'][0:4]
    #box1 = d1['box'][0:4]
    w = 256
    h = 512
    im0 = cv2.imread(file_pairs[0])
    im1 = cv2.imread(file_pairs[1])
    im_shape_0 = im0.shape
    im_shape_1 = im1.shape
    im0 = cv2.resize(im0, (w, h))
    im1 = cv2.resize(im1, (w, h))
    canvas = numpy.zeros((h, h, 3), dtype=numpy.uint8)
    canvas[:,:w,:] = im0
    canvas[:,w:,:] = im1

    top_name, folder_name = get_filename_for_display(file_pairs[0])
    if load_keypoints:
        for ki in range(2):
            keypoints_path = os.path.join(os.path.split(file_pairs[ki])[0], 'keypoints.pkl')
            if os.path.isfile(keypoints_path):
                with open(keypoints_path, 'rb') as fp:
                    keypoints_dict = pickle.load(fp)
                if os.path.split(file_pairs[ki])[1] in keypoints_dict:
                    kps = keypoints_dict[os.path.split(file_pairs[ki])[1]][0]
                    kpsx = kps[:, 0]*w+w*ki
                    kpsy = kps[:, 1]*h
                    canvas = plot_key_points(canvas, kpsx, kpsy, radius=4, put_index=False)

    cv2.putText(canvas, str(top_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.putText(canvas, str(im_shape_0), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    top_name, folder_name = get_filename_for_display(file_pairs[1])
    cv2.putText(canvas, str(top_name), (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2)
    cv2.putText(canvas, str(folder_name), (w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.putText(canvas, str(im_shape_1), (270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
               (0, 255, 0), 2)
    cv2.putText(canvas, str(pair_dist), (w/2, h-w/2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
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

def dump_difficult_pair_files(same_pair_dist, same_pair_files, diff_pair_dist, diff_pair_files, tough_diff_count=64, tough_same_count=64,
                              output_folder='/tmp/difficult/',frame_shape=(1920, 1080), load_keypoints=False):
    same_sort_ids = numpy.argsort(same_pair_dist)[::-1]  # descending argsort
    tough_same_ids = same_sort_ids[0:tough_same_count*100]
    same_select_files, same_select_dist, same_all_files = [],[],[]
    same_dict = {}
    valid_count = 0
    for sid in tough_same_ids:
        p = same_pair_files[sid]
        d = same_pair_dist[sid]
        pid = decode_wcc_image_name(os.path.basename(p[0]))[3]
        if pid not in same_dict:
            same_dict[pid] = 1
        elif same_dict[pid] >= 3:
            same_all_files.append(p)
            continue
        else:
            same_dict[pid] += 1
            valid_count += 1
        same_select_files.append(p)
        same_select_dist.append(d)
        if valid_count >= tough_same_count:
            break

    tough_same_pairs = numpy.array(same_select_files)
    tough_same_dist = numpy.array(same_select_dist)

    canvas = numpy.zeros((frame_shape[1], frame_shape[0], 3))
    canvas = plot_error_spatial(canvas, numpy.array(same_all_files))

    diff_sort_ids = numpy.argsort(diff_pair_dist)
    tough_diff_ids = diff_sort_ids[0:tough_diff_count*100]
    diff_select_files, diff_select_dist, diff_all_files =[], [], []
    diff_dict = {}
    valid_count = 0
    for id in tough_diff_ids:
        p = diff_pair_files[id]
        d = diff_pair_dist[id]
        pid0 = decode_wcc_image_name(os.path.basename(p[0]))[3]
        pid1 = decode_wcc_image_name(os.path.basename(p[1]))[3]
        sorted_pids = tuple(sorted((pid0, pid1)))
        if sorted_pids not in diff_dict:
            diff_dict[sorted_pids] = 1
        elif diff_dict[sorted_pids] >= 3:
            diff_all_files.append(p)
            continue
        else:
            diff_dict[sorted_pids] += 1
            valid_count += 1
        if valid_count >= tough_diff_count:
            break
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
