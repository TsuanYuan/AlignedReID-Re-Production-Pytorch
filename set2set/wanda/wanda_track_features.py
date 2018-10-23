import os
import pickle
import argparse
import wanda_compare_tracks
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from struct_format import utils


def load_track_index(list_file, output_file):
    file_loader = utils.NoPidFileCrops(list_file)
    file_loader.save_index_file(output_file)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file", type=str, help="path to input index file")
    ap.add_argument("data_folder", type=str, help="path to data_folder")
    ap.add_argument("model_path", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--gpu_id", type=int, help="gpu_device", default=7)
    ap.add_argument("--start_line", type=int, help="line to start", default=0)
    ap.add_argument("--last_line", type=int, help="last line to process", default=300000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    ap.add_argument("--batch_max", type=int, help="num crops per batch", default=128)
    args = ap.parse_args()

    video_track_split = utils.load_list_of_unknown_tracks_split(args.index_file, args.start_line, args.last_line, args.sample_size)
    track_features = wanda_compare_tracks.get_descriptors_in_split(args.model_path, video_track_split, args.data_folder,
                                                                   args.gpu_id, args.batch_max)
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    output_file = os.path.join(args.output_folder, "{}_{}_track_feature.pkl".format(str(args.start_line), str(args.last_line)))
    with open(output_file, 'wb') as fp:
        pickle.dump(track_features, fp, protocol=pickle.HIGHEST_PROTOCOL)

