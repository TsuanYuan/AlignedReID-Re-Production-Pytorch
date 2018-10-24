"""
find matching pids to tracklets
Quan Yuan
2018-10-23
"""
import os
import shutil
import argparse
import glob
import pickle


def match(track_folder, track_plot_folder, pid_plot_folder, output_folder):
    track_match_files = glob.glob(os.path.join(track_folder, '*.match'))
    output_plot_folder = os.path.join(output_folder, 'plots')
    if not os.path.isdir(output_plot_folder):
        os.makedirs(output_plot_folder)

    for track_match_file in track_match_files:
        with open(track_match_file, 'rb') as fp:
            track_match = pickle.load(fp)

        for video_track in track_match:
            no_pid_figure_file = os.path.join(track_plot_folder, video_track.split('-')[0]+'.mp4.short', video_track.split('-')[0]+'.mp4.short-'+video_track.split('-')[1]+'-nopid.jpg')
            if not os.path.isfile(no_pid_figure_file):
                print 'cannot find figure file {}'.format(no_pid_figure_file)
            else:
                dest_plot_path = os.path.join(output_plot_folder, video_track + '-nopid.jpg')
                shutil.copyfile(no_pid_figure_file, dest_plot_path)
            print "{} matched top 10 pids are {}".format(str(video_track), str(track_match[video_track]['pids'][0:10]))
    print 'all plots are copied to cluster folder {}'.format(output_plot_folder)


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("track_folder", type=str, help="path to track features")
    ap.add_argument("track_plot_folder", type=str, help="path to track features")
    ap.add_argument("pid_plot_folder", type=str, help="path to 6 figure plots")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    ap.add_argument("--num_clusters", type=int, help="k  for kmeans", default=0)
    args = ap.parse_args()

    match(args.track_folder, args.track_plot_folder, args.pid_plot_folder, args.output_folder)