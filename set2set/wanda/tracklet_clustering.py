"""
clustering tracklets by features
Quan Yuan
2018-10-17
"""
import argparse
import glob
import os
import pickle
import numpy


def match(track_folder, output_folder, sample_size):
    track_desc_files = glob.glob(os.path.join(track_folder, '*.pkl'))
    for track_desc_file in track_desc_files:
        with open(track_desc_file, 'rb') as fp:
            track_desc = pickle.load(fp)
        if len(track_desc) == 0:
            continue
        vt_descriptors = numpy.array([v for k, v in track_desc.iteritems() if v.shape[0] == sample_size])
        vt_descriptors = vt_descriptors.reshape((-1, vt_descriptors.shape[2]))


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("track_folder", type=str, help="path to track features")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    args = ap.parse_args()

    match(args.track_folder, args.output_folder,sample_size=args.sample_size)
