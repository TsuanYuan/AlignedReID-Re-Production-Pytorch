
import argparse
import os, shutil, glob




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('wanda_pid_folder', type=str,
                        help='the path to tracker result')

    parser.add_argument('pid_file', type=str,
                        help='list of mapping videoname, tracklet id, pid')

    parser.add_argument('output_folder', type=str,
                        help='the path to output')

    args = parser.parse_args()

    path_mapping = {}
    with open(args.pid_file, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split(' ')
            if len(fields) < 3:  # only video file name
                continue
            else:
                video_name = fields[0]
                tracklet_id = fields[1]
                pid = fields[2]
            source_path = os.path.join(args.wanda_pid_folder, video_name, tracklet_id, "*.*")
            source_files = glob.glob(source_path)
            dest_folder = os.path.join(args.output_folder, pid)
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)
            for file in source_files:
                shutil.copyfile(file, dest_folder)
            print "copied {0} to {1}".format(source_path, dest_folder)
            