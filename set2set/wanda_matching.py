
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
    dm = dm.reshape((n_single, n_sets, sample_size))
    dm = numpy.moveaxis(dm, 0, -1).reshape((n_sets, sample_size*n_single))
    d = numpy.median(dm, axis=1)
    return d


def pid_track_match(pid_folder, track_folder, cid2pid_file, output_folder, cid_range=(0,100), sample_size=8):
    cid_desc_files = glob.glob(os.path.join(pid_folder, '*.pkl'))
    track_desc_files = glob.glob(os.path.join(track_folder, '*.pkl'))
    with open(cid2pid_file, 'rb') as fp:
        pid_cid_matching = json.load(fp)
    cid_pid_matching = {v: '%08d' % int(k) for k, v in pid_cid_matching.iteritems()}
    pid_top_matches = {}
    for cid_desc_file in cid_desc_files[cid_range[0]:cid_range[1]]:
        with open(cid_desc_file,'rb') as fp:
            cid_desc = pickle.load(fp)
        cids = cid_desc.keys()
        for cid in cids:
            cid_desc_one = cid_desc[cid]
            dist_100 = numpy.array([])
            name_100 = []
            for track_desc_file in track_desc_files:
                with open(track_desc_file, 'rb') as fp:
                    track_desc = pickle.load(fp)
                vt_descriptors = numpy.array([v for k, v in track_desc.iteritems()])
                vt_keys = [k for k, v in track_desc.iteritems()]
                cid_dist = distance(numpy.array(cid_desc_one), vt_descriptors, sample_size=sample_size)
                sort_ids = numpy.argsort(cid_dist)
                top_ids = sort_ids[:100]
                # merge with existing top 100 and pick 100 out of 200
                matching_names = vt_keys[top_ids] + name_100
                top_dist = numpy.concatenate((cid_dist[top_ids], dist_100))
                sort_ids = numpy.argsort(top_dist)
                top_ids = sort_ids[:100]
                name_100 = matching_names[top_ids]
                dist_100 = top_dist[top_ids]
            pid = cid_pid_matching[id]
            if pid not in pid_top_matches:
                pid_top_matches[pid] = {}
            pid_top_matches[pid]['tracks'] = name_100
            pid_top_matches[pid]['scores'] = 1 - dist_100
        output_file = os.path.join(output_folder, str(cid_range[0]) + '.match')
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


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_folder", type=str, help="path to pid feature")
    ap.add_argument("track_folder", type=str, help="path to track features")
    ap.add_argument("cid2pid_file", type=str, help="path cid2pid file")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--start_cid", type=int, help="cid to start", default=0)
    ap.add_argument("--last_cid", type=int, help="last cid to process", default=5000000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    args = ap.parse_args()

    pid_track_match(args.pid_folder, args.track_folder, args.cid2pid_file, args.output_folder,
                    cid_range=(args.start_cid, args.last_cid+1), sample_size=args.sample_size)