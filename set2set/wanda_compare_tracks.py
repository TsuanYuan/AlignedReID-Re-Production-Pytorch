"""
compare pid tracklets with unknown id tracklets
Quan Yuan
2018-09-04
"""
import cv2
import os, glob
import argparse, logging
import numpy
import time
import multiprocessing
import pickle
from load_model import AppearanceModelForward
import compute_feature_alignedReid


def get_pid_descriptors(pid_folder, model_path, ext, device_id=0):
    model = AppearanceModelForward(model_path, sys_device_ids=((device_id,),))

    descriptors_per_file = compute_feature_alignedReid.get_descriptors(pid_folder, model, force_compute=False, ext=ext)
    pid_descriptors = {}

    for pid_folder in descriptors_per_file:
        pid_str = os.path.basename(pid_folder)
        descriptors = []
        for descriptor_item in descriptors_per_file[pid_folder]:
            descriptors.append(descriptor_item['descriptor'])
        # m = numpy.expand_dims(numpy.mean(numpy.array(descriptors), axis=0),0)
        # l2_norm = numpy.sqrt((m * m + 1e-10).sum(axis=1))
        # m = m / (l2_norm[:, numpy.newaxis])
        pid_descriptors[pid_str] = numpy.squeeze(descriptors)
        print "finished pid folder {0}".format(pid_folder)
    return pid_descriptors

def distance(single_set_descriptor, multi_set_descriptors):
    # cosine
    d = []
    for set_descriptors in multi_set_descriptors:
        dm = (1-numpy.dot(single_set_descriptor, set_descriptors.transpose()))
        d.append(numpy.median(dm))
    #d0 = (1-numpy.dot(a,b))
    # # euclidean
    # d1 = numpy.linalg.norm(a-b)
    # if abs(d0*2-d1)>0.0001:
    #     raise Exception('cosine and euclidean distance not equal')
    return numpy.array(d)


def compare_one_video_folder(video_folder, model, pid_descriptor_array, pid_descriptor_names, output_folder, ext, max_id=100):
    track_folders = os.listdir(video_folder)
    track_match_results = {}
    # for track_folder in track_folders:
    #     track_full_folder = os.path.join(video_folder, track_folder)

    track_descritpors = compute_feature_alignedReid.get_descriptors(video_folder, model, force_compute=False, ext=ext)
    for track_id_str in track_descritpors:
        descriptors = []
        for descriptor_item in track_descritpors[track_id_str]:
            descriptors.append(descriptor_item['descriptor'])
        if len(descriptors) == 0:
            continue
        # m = numpy.expand_dims(numpy.mean(numpy.array(descriptors), axis=0),0)
        # l2_norm = numpy.sqrt((m * m + 1e-10).sum(axis=1))
        # m = m / (l2_norm[:, numpy.newaxis])
        distances = distance(descriptors, pid_descriptor_array)
        sort_ids = numpy.argsort(distances)
        top_ids = sort_ids[:100]
        matching_names = pid_descriptor_names[top_ids]
        track_path = os.path.join(video_folder, track_id_str)
        if track_path not in track_match_results:
            track_match_results[track_path]={}
        track_match_results[track_path]['pids'] = matching_names
        track_match_results[track_path]['distances'] = distances[top_ids]
    video_name = os.path.basename(video_folder)
    output_file = os.path.join(output_folder, video_name+'.match')
    with open(output_file, 'wb') as fp:
        pickle.dump(track_match_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print "matching results are dumped to {0}".format(output_file)


def batch_run_match(inputs):
    video_folder = inputs[0]
    model = inputs[1]
    pid_descriptor_array = inputs[2]
    pid_descriptor_name = inputs[3]
    output_folder = inputs[4]
    ext = inputs[5]
    compare_one_video_folder(video_folder, model, pid_descriptor_array, pid_descriptor_name, output_folder, ext)


def compare_unknown_tracks(folder, model_path, output_folder, ext, pid_descriptors, device_id, start_index=0, num_to_run=-1, num_gpus=8):
    video_folders = os.listdir(folder)
    # models = []
    # n = len(video_folders)
    # for i in range(num_gpus):
    #     models.append(AppearanceModelForward(model_path, ((i,),)))
    model = AppearanceModelForward(model_path, ((device_id,),))
    pid_descriptor_array = []
    pid_descriptor_names = []
    for pid_str in pid_descriptors:
        pid_descriptor_array.append(pid_descriptors[pid_str])
        pid_descriptor_names.append(pid_str)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # for debug run
    if num_to_run < 0:
        num_to_run = len(video_folders[start_index:])
    for video_folder in video_folders[start_index:start_index+num_to_run]:
        #model = models[device_id]
        full_folder = os.path.join(folder,video_folder)
        compare_one_video_folder(full_folder, model, pid_descriptor_array, numpy.array(pid_descriptor_names), output_folder, ext)
    """
    assgined_models = [models[i%num_gpus] for i, _ in enumerate(video_folders)]
    pid_descriptor_names = [numpy.array(pid_descriptor_names)]*n
    pid_descriptor_arrays = [numpy.array(pid_descriptor_array).transpose()]*n

    exts = [ext]*n
    output_folders = [ext]*n
    p = multiprocessing.Pool(processes=num_gpus)
    p.map(batch_run_match, zip(video_folders, assgined_models, pid_descriptor_arrays, pid_descriptor_names, output_folders, exts))
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('pid_folder', type=str,
                        help='the path to the crops')

    parser.add_argument('tracklet_folder', type=str,
                        help='the path to parent folder of video + tracklet folders')

    parser.add_argument('model_path', type=str,
                        help='the path to appearance model file')

    parser.add_argument('output_folder', type=str,
                        help='folder of output matching results')

    parser.add_argument('ext', type=str,
                        help='the ext to appearance descriptor file')

    parser.add_argument('--device_id', type=int, default=0, required=False,
                        help='the gpu id')

    parser.add_argument('--start_video_index', type=int, default=0,
                        help='the start video folder index')

    parser.add_argument('--num_videos', type=int, default=150,
                        help='the num of videos to process')

    args = parser.parse_args()
    start_time = time.time()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    pid_file = os.path.join(args.output_folder, 'pids'+'_'+args.ext+'.pkl')
    if os.path.isfile(pid_file):
        with open(pid_file, 'rb') as fp:
            pid_descriptors = pickle.load(fp)
    else:
        pid_descriptors = get_pid_descriptors(args.pid_folder, args.model_path, args.ext, args.device_id)
        with open(pid_file, 'wb') as fp:
            pickle.dump(pid_descriptors, fp, protocol=pickle.HIGHEST_PROTOCOL)
    compare_unknown_tracks(args.tracklet_folder,args.model_path, args.output_folder, args.ext, pid_descriptors,
                           args.device_id, start_index=args.start_video_index, num_to_run=args.num_videos,
                           num_gpus=8)
