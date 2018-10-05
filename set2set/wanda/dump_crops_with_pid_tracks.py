"""
dump wanda data to crops
Quan Yuan
2018-10-04
"""

import argparse
import json
import pickle
import os
from collections import defaultdict
import cv2
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from struct_format import utils

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_list_file", type=str, help="files of list of pids")
    ap.add_argument("track_folder", type=str, help="path to track crop folder")
    ap.add_argument("output_folder", type=str, help="path to output file")
    args = ap.parse_args()

    start_time = time.time()
    data_by_video_name = {}
    with open(args.pid_list_file, 'r') as fp:
        pid_to_tracks = json.load(fp)
        for pid in pid_to_tracks:
            track_list = pid_to_tracks[pid]
            for track in track_list:
                parts = track.split('-')
                video_name = parts[0]
                track_id = parts[1]
                if video_name not in data_by_video_name:
                    data_by_video_name[video_name] = []
                data_by_video_name[video_name].append(track_id+'-'+pid.zfill(8))

    crops_per_pid = defaultdict(int)
    tracklets_per_pid = defaultdict(int)
    for video_name in data_by_video_name:
        track_list = data_by_video_name[video_name]
        track_data_index_file = os.path.join(args.track_folder, video_name + '_part_idx_map.pickl')
        if not os.path.isfile(track_data_index_file):
            print 'cannot find index file {}'.format(track_data_index_file)
            continue
        with open(track_data_index_file, 'rb') as fp:
            data_offsets = pickle.load(fp)
        print 'dumping crops on video name {}'.format(video_name)
        for track_pid in track_list:
            track_id, pid = track_pid.split('-')
            track_key = video_name + '-' + track_id
            matched_keys = [k for k in data_offsets if k.find(track_key)>=0]
            if len(matched_keys) ==0:
                print 'cannot find key {} in binary data offsets'.format(track_key)
                continue
            output_pid_folder = os.path.join(args.output_folder, str(pid))
            if not os.path.isdir(output_pid_folder):
                os.makedirs(output_pid_folder)
            for key in matched_keys:
                video_offset = data_offsets[key]
                part_file = os.path.join(args.track_folder, video_offset[0])
                if os.path.isfile(part_file) == False:
                    print "cannot find binary data file {}".format(part_file)
                    continue
                crops_per_pid[pid]+=1
                image_bgr = utils.read_one_image(part_file, int(video_offset[1]), bgr_flag=True)
                image_output_name = key.split('-')[0]+'_'+pid+'_'+key.split('-')[-1]+'.jpg'
                image_output_path = os.path.join(output_pid_folder, image_output_name)
                cv2.imwrite(image_output_path, image_bgr)
            tracklets_per_pid[pid] += 1

    mean_crops = sum([crops_per_pid[k] for k in crops_per_pid])/float(len(crops_per_pid))
    mean_track_count = sum([tracklets_per_pid[k] for k in tracklets_per_pid])/float(len(tracklets_per_pid))
    print "mean crops per pid is {}, mean tracks per pid is {}".format(str(mean_crops), str(mean_track_count))
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'all dump crops finished in {} for pids in {}, output to {}'.format(elapsed, args.pid_list_file, args.output_folder)

