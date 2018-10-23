"""
compare pid tracklets with unknown id tracklets
Quan Yuan
2018-09-04
"""

import os
import argparse
import numpy
import time
import json
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from load_model import AppearanceModelForward
import compute_feature_alignedReid
from struct_format import utils


def get_descriptors_in_split(model, split_data, data_folder, batch_max=128):
    descriptors = {}
    #images_batch = []
    #track_batch = []
    for i, video_track in enumerate(split_data.keys()):
        crop_pairs = split_data[video_track]
        images = []
        count = 0
        for p in crop_pairs:
            file_name, offset = p
            data_file = os.path.join(data_folder, file_name)
            crop_ready = True
            if not os.path.isfile(data_file):
                print 'fail to load data file {}'.format(data_file)
                crop_ready = False
            if crop_ready:
                image = utils.read_one_image(data_file, offset)
                images.append(image)
            if len(images) >= batch_max or count == len(crop_pairs)-1:
                images_normalized = model.normalize_images(images)
                descriptor_batch = model.compute_features_on_batch(images_normalized)
                if video_track not in descriptors:
                    descriptors[video_track] = descriptor_batch
                else:
                    descriptors[video_track] = numpy.concatenate((descriptor_batch, descriptors[video_track]), axis=0)
                images = []
            count += 1

        if (i+1) % 100 == 0:
            print "finished computing descriptor of pid/track_id {} out of {}".format(str(i+1), str(len(split_data)))
    return descriptors


def get_descriptors_in_binary(model_path, list_file, data_folder, device_id, sample_size, pid_flag, batch_max=128):
    if pid_flag:
        file_loader = utils.MultiFileCrops(data_folder, list_file)
        id_list = file_loader.get_pid_list()
    else:
        file_loader = utils.NoPidFileCrops(list_file)
        id_list = file_loader.get_track_list()

    model = AppearanceModelForward(model_path, sys_device_ids=((device_id,),))
    descriptors = {}
    images_batch = []
    descriptor_batch = []
    ids_batch = []
    for k,id in enumerate(id_list):
        images = file_loader.load_fixed_count_images_of_one_pid(id, sample_size)
        images_batch = images_batch + images
        ids_batch.append(id)
        if len(images_batch) >= batch_max or k == len(id_list)-1:
            if len(images_batch) > 0:
                descriptor_batch = model.compute_features_on_batch(numpy.array(images_batch))
            for j, kid in enumerate(ids_batch):
                descriptors[kid] = descriptor_batch[j*sample_size:(j+1)*sample_size,:]
            descriptor_batch = []
            images_batch = []
            ids_batch = []
        if (k+1)%100 == 0:
            print "finished computing descriptor of pid/track_id {} out of {}".format(str(k), str(len(id_list)))

    return descriptors

def get_pid_descriptors(pid_folder, model_path, ext, sample_size, device_id=0):
    model = AppearanceModelForward(model_path, sys_device_ids=((device_id,),))

    descriptors_per_file = compute_feature_alignedReid.get_descriptors(pid_folder, model, sample_size=sample_size, force_compute=False, ext=ext)
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

def distance(single_set_descriptor, multi_set_descriptors, sample_size):
    # cosine
    # d = []
    dm = (1-numpy.dot(single_set_descriptor, multi_set_descriptors.transpose()))
    # start_count = 0
    n_multi = multi_set_descriptors.shape[0]
    n_sets = n_multi/sample_size
    n_single = single_set_descriptor.shape[0]
    dm = dm.reshape((n_single, n_sets, sample_size))
    dm = numpy.moveaxis(dm, 0, -1).reshape((n_sets, sample_size*n_single))
    d = numpy.median(dm, axis=1)
    return d
    # for set_size in multi_set_sizes:
    #     #dm = (1-numpy.dot(single_set_descriptor, set_descriptors.transpose()))
    #     dx = dm[:, start_count:start_count+set_size]
    #     d.append(numpy.median(dx))
    #     start_count += set_size
    # d0 = (1-numpy.dot(a,b))
    # # euclidean
    # d1 = numpy.linalg.norm(a-b)
    # if abs(d0*2-d1)>0.0001:
    #     raise Exception('cosine and euclidean distance not equal')
    # return numpy.array(d)


def compare_one_video_folder(video_folder, model, pid_descriptor_list, pid_descriptor_names, output_folder, ext, sample_size, max_id=100):
    track_folders = os.listdir(video_folder)
    track_match_results = {}
    # for track_folder in track_folders:
    #     track_full_folder = os.path.join(video_folder, track_folder)
    pid_descriptor_sizes = [s.shape[0] for s in pid_descriptor_list]
    pid_descriptor_array = numpy.vstack(pid_descriptor_list)
    #pid_descriptor_array = [numpy.concatenate((pid_descriptor_array )) for pid_array in pid_descriptor_list]

    track_descritptors = compute_feature_alignedReid.get_descriptors(video_folder, model, force_compute=False, ext=ext)
    for track_id_str in track_descritptors:
        descriptors = []
        for descriptor_item in track_descritptors[track_id_str]:
            descriptors.append(descriptor_item['descriptor'])
        if len(descriptors) == 0:
            continue
        # m = numpy.expand_dims(numpy.mean(numpy.array(descriptors), axis=0),0)
        # l2_norm = numpy.sqrt((m * m + 1e-10).sum(axis=1))
        # m = m / (l2_norm[:, numpy.newaxis])
        distances = distance(numpy.array(descriptors), pid_descriptor_array, sample_size)
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
    sample_size = inputs[6]
    compare_one_video_folder(video_folder, model, pid_descriptor_array, pid_descriptor_name, output_folder, ext, sample_size)


def compare_unknown_tracks(folder, model_path, output_folder, ext, pid_descriptors, device_id, sample_size, start_index=0, num_to_run=-1, num_gpus=8):
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
        compare_one_video_folder(full_folder, model, pid_descriptor_array, numpy.array(pid_descriptor_names), output_folder, ext, sample_size)
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

    parser.add_argument('pid_path', type=str,
                        help='the path to the crops')

    parser.add_argument('tracklet_path', type=str,
                        help='the path to parent folder of video + tracklet folders')

    parser.add_argument('model_path', type=str,
                        help='the path to appearance model file')

    parser.add_argument('output_folder', type=str,
                        help='folder of output matching results')

    parser.add_argument('ext', type=str,
                        help='the ext to appearance descriptor file')

    parser.add_argument('--pid_data_folder', type=str, default='',
                        help='actual folder of pid data')

    parser.add_argument('--pid_id_matching_file', type=str, default='',
                        help='conversion between classifier id to pid')

    parser.add_argument('--tracklet_data_folder', type=str, default='',
                        help='actual folder of track data')

    parser.add_argument('--device_id', type=int, default=0, required=False,
                        help='the gpu id')

    parser.add_argument('--start_video_index', type=int, default=0,
                        help='the start video folder index')

    parser.add_argument('--num_videos', type=int, default=150,
                        help='the num of videos to process')

    parser.add_argument('--sample_size', type=int, default=16,
                        help='the num of crops to sample per id')

    parser.add_argument('--folder_format', action='store_true', default=False,
                        help='whether to load pid and tracks from folder format')

    args = parser.parse_args()
    start_time = time.time()
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    if args.folder_format:
        pid_file = os.path.join(args.output_folder, 'pids'+'_'+args.ext+'.pkl')
        if os.path.isfile(pid_file):
            with open(pid_file, 'rb') as fp:
                pid_descriptors = pickle.load(fp)
        else:
            pid_descriptors = get_pid_descriptors(args.pid_path, args.model_path, args.ext, args.sample_size, device_id=args.device_id)
            with open(pid_file, 'wb') as fp:
                pickle.dump(pid_descriptors, fp, protocol=pickle.HIGHEST_PROTOCOL)


        compare_unknown_tracks(args.tracklet_path,args.model_path, args.output_folder, args.ext, pid_descriptors,
                           args.device_id, args.sample_size, start_index=args.start_video_index, num_to_run=args.num_videos,
                           num_gpus=8)
    else:
        with open(args.pid_id_matching_file, 'rb') as fp:
            pid_id_matching = json.load(fp)
        id_pid_matching = {v: '%08d'%int(k) for k, v in pid_id_matching.iteritems()}

        pid_desc_file = os.path.join(args.output_folder, args.ext+'_pid_descriptors.pkl')
        if os.path.isfile(pid_desc_file):
            with open(pid_desc_file, 'rb') as fp:
                pid_descriptors=pickle.load(fp)
        else:
            pid_descriptors = get_descriptors_in_binary(args.model_path, args.pid_path, args.pid_data_folder, args.device_id,
                                                    sample_size=8, pid_flag=True)
            with open(pid_desc_file, 'wb') as fp:
                pickle.dump(pid_descriptors, fp, protocol=pickle.HIGHEST_PROTOCOL)
        vt_desc_file = os.path.join(args.output_folder, args.ext+'_vt_descriptors.pkl')
        if os.path.isfile(vt_desc_file):
            with open(vt_desc_file, 'rb') as fp:
                video_track_descriptors=pickle.load(fp)
        else:
            video_track_descriptors = get_descriptors_in_binary(args.model_path, args.tracklet_path, args.tracklet_data_folder,
                                                            args.device_id, sample_size=4, pid_flag=False)
            with open(vt_desc_file, 'wb') as fp:
                pickle.dump(video_track_descriptors, fp, protocol=pickle.HIGHEST_PROTOCOL)

        vt_descriptors = numpy.array([v for k,v in video_track_descriptors.iteritems()])
        vt_keys = [k for k,v in video_track_descriptors.iteritems()]
        track_match_results = {}
        for id in pid_descriptors:
            pid_dist = distance(numpy.array(pid_descriptors[id]), vt_descriptors, sample_size=args.sample_size)
            sort_ids = numpy.argsort(pid_dist)
            top_ids = sort_ids[:100]
            matching_names = vt_keys[top_ids]
            #track_path = os.path.join(video_folder, track_id_str)
            pid = id_pid_matching[id]
            if pid not in track_match_results:
                track_match_results[pid] = {}
            track_match_results[pid]['tracks'] = matching_names
            track_match_results[pid]['scores'] = 1-pid_dist[top_ids]
            output_file = os.path.join(args.output_folder, str(pid) + '.match')
            with open(output_file, 'wb') as fp:
                pickle.dump(track_match_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print "pid matching results are dumped to {0}".format(output_file)