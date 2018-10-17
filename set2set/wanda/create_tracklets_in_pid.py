"""
dump wanda data to tracklets for tracklet level evaluation
Quan Yuan
2018-10-
"""

import argparse
import json
import pickle
import os
from collections import defaultdict
import cv2
import glob
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from struct_format import utils
from evaluate import misc


def split_tracklets_of_one_pid(pid_folder_path, frame_interval_gap_th=200, tracklet_length_max=100):
    image_files = glob.glob(os.path.join(pid_folder_path, "*.jpg"))
    # 1. separate by camera and video
    crops_by_camera_video = {}
    for image_file in image_files:
        channel, date, video_time, pid, frame_id = misc.decode_wcc_image_name(os.path.basename(image_file))
        cdv = str(channel)+'_'+str(date)+'_'+str(video_time)+'_'+str(pid)
        if cdv not in crops_by_camera_video:
            crops_by_camera_video[cdv] = []
        crops_by_camera_video[cdv].append((frame_id, image_file))
    # 2. sort by time (frame_id) and split by frame interval gap
    tracklets_collections = {}
    for cdv in crops_by_camera_video:
        frame_data = crops_by_camera_video[cdv]
        frame_data = sorted(frame_data, key=lambda k: k[0])
        current_frame_id = None
        current_track = []
        pid = int(cdv.split('_')[-1])
        for frame_id, image_file in frame_data:
            if current_frame_id is None:
                current_frame_id = frame_id
            else:
                if frame_id - current_frame_id <= frame_interval_gap_th and len(current_track) < tracklet_length_max:
                    current_track.append((frame_id, image_file))
                else:
                    if pid not in tracklets_collections:
                        tracklets_collections[pid] = []
                    tracklets_collections[pid].append(list(current_track))
                    current_track = []
    return tracklets_collections


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_folder", type=str, help="parent folder of pid folders")
    ap.add_argument("output_file", type=str, help="path to output record file")
    args = ap.parse_args()

    start_time = time.time()
    sub_folders = os.listdir(args.pid_folder)

    tracklets_splits={}
    tracklet_count = 0
    for sub_folder in sub_folders:
        pid_folder_path = os.path.join(args.pid_folder, sub_folder)
        if os.path.isdir(pid_folder_path) and sub_folder.isdigit():
            pid = int(sub_folder)
            tracklets_splits[pid] = split_tracklets_of_one_pid(pid_folder_path)
            tracklet_count += len(tracklets_splits[pid])

    with open(args.output_file, 'wb') as fp:
        pickle.dump(tracklets_splits, fp, protocol=pickle.HIGHEST_PROTOCOL)

    finish_time = time.time()
    elapsed = finish_time - start_time
    print "all tracks were dumped to {} in {}".format(args.output_file, elapsed)
