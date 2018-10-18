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
from sklearn.cluster import k_means
import shutil


def match(track_folder, plots_folder, output_folder, num_clusters, force_clustering=False):
    output_feature_file = os.path.join(output_folder, 'all_features.res')
    if os.path.isfile(output_feature_file):
        print 'feature per track file exist {} will skip reading pkl files'.format(output_feature_file)
    else:
        track_desc_files = glob.glob(os.path.join(track_folder, '*.pkl'))
        descriptors = []
        video_tracks = []
        for track_desc_file in track_desc_files:
            with open(track_desc_file, 'rb') as fp:
                track_desc = pickle.load(fp)
            if len(track_desc) == 0:
                continue
            video_track_names = [k for k, v in track_desc.iteritems()]
            descriptors_median = [numpy.median(v, axis=0) for k, v in track_desc.iteritems()]
            descriptors_normed = [x /(numpy.linalg.norm(x)+0.000001) for x in descriptors_median]
            video_tracks += video_track_names
            descriptors += descriptors_normed

        with open(output_feature_file, 'wb') as fp:
            pickle.dump(video_tracks, fp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(descriptors, fp, pickle.HIGHEST_PROTOCOL)
        print "features per track is saved to {}".format(output_feature_file)

    with open(output_feature_file, 'rb') as fp:
        video_tracks = pickle.load(fp)
        descriptors = pickle.load(fp)

    output_cluster_result_file = os.path.join(output_folder, 'cluster_result.res')
    if os.path.isfile(output_cluster_result_file):
        with open(output_cluster_result_file, 'rb') as fp:
            centroids = pickle.load(fp)
            labels = pickle.load(fp)
            inertia = pickle.load(fp)
    else:
        centroids, labels, inertia = k_means(descriptors, num_clusters, max_iter=30000, n_jobs=10)
        with open(output_cluster_result_file, 'wb') as fp:
            pickle.dump(centroids, fp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(labels, fp, pickle.HIGHEST_PROTOCOL)
            pickle.dump(inertia, fp, pickle.HIGHEST_PROTOCOL)

    output_plot_folder = os.path.join(output_folder, 'plots')
    if not os.path.isdir(output_plot_folder):
        os.makedirs(output_plot_folder)
    for video_track, label in zip(video_tracks, labels):
        no_pid_figure_file = os.path.join(plots_folder, video_track+'-nopid.jpg')
        if not os.path.isfile(no_pid_figure_file):
            print 'cannot find figure file {}'.format(no_pid_figure_file)
        else:
            output_cluster_folder = os.path.join(output_plot_folder, str(label))
            if not os.path.isdir(output_cluster_folder):
                os.makedirs(output_cluster_folder)
            dest_plot_path = os.path.join(output_cluster_folder, video_track+'-nopid.jpg')
            shutil.copyfile(no_pid_figure_file, dest_plot_path)
    print 'all plots are copied to cluster folder {}'.format(output_plot_folder)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("track_folder", type=str, help="path to track features")
    ap.add_argument("plot_folder", type=str, help="path to 6 figure plots")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    ap.add_argument("--num_clusters", type=int, help="k  for kmeans", default=0)
    args = ap.parse_args()

    match(args.track_folder, args.plot_folder, args.output_folder, args.num_clusters)
