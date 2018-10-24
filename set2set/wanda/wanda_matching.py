
import os, glob
import pickle
import numpy
import json
import argparse

def distance(single_set_descriptor, multi_set_descriptors, sample_size):
    # cosine
    # d = []
    dm = (1-numpy.dot(single_set_descriptor, multi_set_descriptors.transpose()))
    # start_count = 0
    n_multi = multi_set_descriptors.shape[0]
    n_sets = n_multi/sample_size
    n_single = single_set_descriptor.shape[0]
    n_s = n_single/sample_size
    dm = dm.reshape((n_single, n_sets, sample_size))
    dm1 = numpy.squeeze(numpy.median(dm, axis=2))
    dm2 = dm1.reshape((n_s, sample_size, n_sets))
    d = numpy.squeeze(numpy.median(dm2, axis=1))
    if len(d.shape)==1:
        d = numpy.expand_dims(d,1)
    return d


def pid_track_match(pid_folder, track_folder, cid2pid_file, output_folder, cid_range=(0,100), sample_size=8):
    cid_desc_files = glob.glob(os.path.join(pid_folder, '*.pkl'))
    track_desc_files = glob.glob(os.path.join(track_folder, '*.pkl'))
    with open(cid2pid_file, 'rb') as fp:
        pid_cid_matching = json.load(fp)
    cid_pid_matching = {v: '%08d' % int(k) for k, v in pid_cid_matching.iteritems()}
    pid_top_matches = {}
    for cid_desc_file in cid_desc_files:
        with open(cid_desc_file,'rb') as fp:
            cid_desc = pickle.load(fp)
        cids = cid_desc.keys()
        if len(cids) == 0:
            continue
        #for cid in cids[cid_range[0]:cid_range[1]]:
        cid_desc_n = [cid_desc[cid] for cid in cids[cid_range[0]:cid_range[1]]]
        cids_now = cids[cid_range[0]:cid_range[1]]
        dist_100 = {}
        name_100 = {}

        for track_desc_file in track_desc_files:
            with open(track_desc_file, 'rb') as fp:
                track_desc = pickle.load(fp)
            if len(track_desc) == 0:
                continue
            vt_descriptors = numpy.array([v for k, v in track_desc.iteritems() if v.shape[0]==sample_size])
            vt_descriptors = vt_descriptors.reshape((-1, vt_descriptors.shape[2]))
            vt_keys = numpy.array([k for k, v in track_desc.iteritems() if v.shape[0]==sample_size])
            cid_dist = distance(numpy.array(cid_desc_n).reshape((-1, vt_descriptors.shape[1])), vt_descriptors, sample_size=sample_size)
            for k in range(cid_dist.shape[0]):
                sort_ids = numpy.argsort(cid_dist[k,:])
                top_ids = sort_ids[:100]
                pid = cid_pid_matching[int(cids_now[k])]
                # merge with existing top 100 and pick 100 out of 200
                if pid in name_100:
                    matching_names = numpy.concatenate((vt_keys[top_ids], numpy.array(name_100[pid])))
                    top_dist = numpy.concatenate((cid_dist[k,top_ids], numpy.array(dist_100[pid])))
                else:
                    matching_names = vt_keys[top_ids]
                    top_dist = cid_dist[k,top_ids]
                sort_ids = numpy.argsort(top_dist)
                top_ids = sort_ids[:100]

                name_100[pid] = matching_names[top_ids]
                dist_100[pid] = top_dist[top_ids]
        for pid in name_100:
            if not pid.isdigit():
                continue
            if pid not in pid_top_matches:
                pid_top_matches[pid] = {}
            pid_top_matches[pid]['tracks'] = name_100[pid]
            pid_top_matches[pid]['scores'] = 1 - dist_100[pid]
        if len(pid_top_matches) == 0:
            continue
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(cid_desc_file))[0]+'_'+str(cid_range[0]) + '.match')
        if os.path.isdir(output_folder) == False:
            os.makedirs(output_folder)
        with open(output_file, 'wb') as fp:
            pickle.dump(pid_top_matches, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print "pid matching results are dumped to {0}".format(output_file)

def load_pid_descriptor(pid_folder):
    pid_desc_files = glob.glob(os.path.join(pid_folder, '*.pkl'))
    pid_desc = {}
    for pid_desc_file in pid_desc_files:
        with open(pid_desc_file,'rb') as fp:
            d = pickle.load(fp)
            pid_desc.update(d)
    return pid_desc


def video_track_match(pid_folder, track_folder, output_folder, sample_size=8, track_batch_size=2000, search_vt_names=None):
    cid_desc_files = glob.glob(os.path.join(pid_folder, '*.pkl'))
    track_desc_files = glob.glob(os.path.join(track_folder, '*.pkl'))
    pid_top_matches = {}
    for track_desc_file in track_desc_files:
        with open(track_desc_file, 'rb') as fp:
            track_desc = pickle.load(fp)
        if len(track_desc) == 0:
            continue
        vt_names_in_file = numpy.array(track_desc.keys())
        if search_vt_names is not None:
            vt_names_in_file = numpy.array([search_vt_name for search_vt_name in search_vt_names if search_vt_name in vt_names_in_file])
        nvt = len(vt_names_in_file)
        for vi in range(0, nvt, track_batch_size):
            vtids = numpy.array(range(vi, min(nvt, vi+track_batch_size)))
            vt_names = vt_names_in_file[vtids]
            vt_descriptors = numpy.concatenate(
                [track_desc[k][numpy.round(numpy.linspace(0, track_desc[k].shape[0] - 1, sample_size)).astype(int), :]
                 for k in vt_names], axis=0)
            # vt_names = numpy.array([k for k, v in track_desc.iteritems()])
            # vt_descriptors = vt_descriptors.reshape((-1, vt_descriptors.shape[2]))
            dist_100 = {}
            pid_100 = {}
            for cid_desc_file in cid_desc_files:
                with open(cid_desc_file, 'rb') as fp:
                    cid_desc = pickle.load(fp)
                cids = numpy.array(cid_desc.keys())
                if len(cids) == 0:
                    continue
                cid_desc_n = [cid_desc[cid][numpy.round(numpy.linspace(0, cid_desc[cid].shape[0]-1, sample_size)).astype(int)] for cid in cids]
                cid_dist = distance(numpy.concatenate(cid_desc_n, axis=0), vt_descriptors,
                                    sample_size=sample_size)

                for k in range(cid_dist.shape[1]): # track id
                    sort_ids = numpy.argsort(cid_dist[:,k])
                    top_ids = sort_ids[:100]
                    track_name = vt_names[k]
                    # merge with existing top 100 and pick 100 out of 200
                    if track_name in pid_100:
                        matching_pids = numpy.concatenate((cids[top_ids], numpy.array(pid_100[track_name])))
                        top_dist = numpy.concatenate((cid_dist[top_ids,k], numpy.array(dist_100[track_name])))
                    else:
                        matching_pids = cids[top_ids]
                        top_dist = cid_dist[top_ids,k]
                    sort_ids = numpy.argsort(top_dist)
                    top_ids = sort_ids[:100]

                    pid_100[track_name] = matching_pids[top_ids]
                    dist_100[track_name] = top_dist[top_ids]
                # for cid in cids[cid_range[0]:cid_range[1]]:
                # cid_desc_n = [cid_desc[cid] for cid in cids[cid_range[0]:cid_range[1]]]
                # cids_now = cids[cid_range[0]:cid_range[1]]

            for track_name in pid_100:
                if track_name not in pid_top_matches:
                    pid_top_matches[track_name] = {}
                pid_top_matches[track_name]['pids'] = pid_100[track_name]
                pid_top_matches[track_name]['scores'] = 1 - dist_100[track_name]
            if len(pid_top_matches) == 0:
                continue
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(track_desc_file))[0]+'_pid' + '.match')
        if os.path.isdir(output_folder) == False:
            os.makedirs(output_folder)
        with open(output_file, 'wb') as fp:
            pickle.dump(pid_top_matches, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print "pid matching results are dumped to {0}".format(output_file)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_folder", type=str, help="path to pid feature")
    ap.add_argument("track_folder", type=str, help="path to track features")
    ap.add_argument("cid2pid_file", type=str, help="path cid2pid file")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--start_cid", type=int, help="cid to start", default=0)
    ap.add_argument("--last_cid", type=int, help="last cid to process", default=5000000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=16)
    ap.add_argument("--t2p", action='store_true', help="match from tracklet to pid", default=False)
    ap.add_argument("--search_vt_file", type=str, help="tracklet name to find", default=None)
    args = ap.parse_args()
    if args.t2p:
        print "match from tracklet to pids"
        if args.search_vt_file is not None:
            with open(args.search_vt_file, 'r') as fp:
                search_vts = [k.strip() for k in fp.readlines()]
        else:
            search_vts = None
        video_track_match(args.pid_folder, args.track_folder, args.output_folder, sample_size=args.sample_size, search_vt_names=search_vts)
    else:
        print "match from pids to tracklets"
        pid_track_match(args.pid_folder, args.track_folder, args.cid2pid_file, args.output_folder,
                    cid_range=(args.start_cid, args.last_cid+1), sample_size=args.sample_size)
