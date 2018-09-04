"""
create training sets out of long track id
Quan Yuan
2018-09-03
"""
import os,glob
import json
import load_results
import argparse
import pickle

GROUP_SIZE=8

def load_tracklets_result(result_file, video_folder, id_type):
    if os.path.isdir(result_file): # assume pb folder
        global_data = {}
        pb_files = glob.glob(os.path.join(result_file, '*.pb'))
        for pb_file in pb_files:
            print 'processing {0}'.format(pb_file)
            d = load_results.load_pb(pb_file, video_folder, id_type)
            global_data.update(d)
    else:
        _, ext = os.path.splitext(result_file)
        if ext == '.json':
            fp = open(result_file, 'r')
            tracker_records = json.load(fp)
            global_data = load_results.load_global_data(tracker_records, '')
        elif ext == '.pb':

            global_data = load_results.load_pb(result_file, '', id_type)

        else:
            raise Exception('unknown merged result format {0}'.format(result_file))

    return global_data


def get_id_data_set(global_data):
    tracklet_data = {}
    same_time_tracks = {}
    for video_name in global_data:
        tracklet_data[video_name] = {}
        for frame_index in global_data[video_name]:
            current_frame_tracks = set()
            boxes = global_data[video_name][frame_index]
            for box in boxes:
                track_id = box['track_id']
                if track_id not in tracklet_data[video_name]:
                    tracklet_data[video_name][track_id] = []

                boxA = box['human_box'][0:4]
                tracklet_data[video_name][track_id].append(boxA)
                current_frame_tracks.add(track_id)
            for track_id in current_frame_tracks:
                if track_id not in same_time_tracks:
                    same_time_tracks[track_id] = set()
                same_time_tracks[track_id].union(current_frame_tracks)

    # remove self link
    for track_id in same_time_tracks:
        same_time_tracks[track_id].remove(track_id)

    return tracklet_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('tracker_result_path', type=str,
                        help='the path to tracker result')

    parser.add_argument('output_file', type=str,
                        help='the path to output tracklets')

    parser.add_argument('--id_type', type=str, default='long_tracklet_id',
                        help='type of person ids, like pid, long_tracklet_id, tracklet_index')

    args = parser.parse_args()
    global_data = load_tracklets_result(args.tracker_result_path, '', args.id_type)
    tracklet_data = get_id_data_set(global_data)
    with open(args.output_file, 'wb') as fp:
        pickle.dump(tracklet_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print "tracklet data were dumped to {0}".format(args.output_file)

