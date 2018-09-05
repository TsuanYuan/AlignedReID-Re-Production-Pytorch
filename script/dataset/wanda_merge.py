
import argparse
import os, shutil, glob
import numpy



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('wanda_pid_folder', type=str,
                        help='the path to tracker result')

    parser.add_argument('pid_file', type=str,
                        help='list of mapping videoname, tracklet id, pid')

    parser.add_argument('output_folder', type=str,
                        help='the path to output')

    parser.add_argument('--sample_size', type=int, default=16,
                        help='the default number of crops per tracklet')

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
            source_path = os.path.join(args.wanda_pid_folder, video_name, tracklet_id, "*.jpg")
            source_files = glob.glob(source_path)
            dest_folder = os.path.join(args.output_folder, pid)
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)

            n = len(source_files)
            # print "n={0}".format(str(n))
            # print "source_files = {0}".format(str(source_files))
            # step = max(1, int(n/float(args.sample_size)))
            if n==0:
                continue
            sample_files = [source_files[int(round(k))] for k in numpy.linspace(0, n-1, args.sample_size)]
            for jpg_file in sample_files:
                shutil.copy(jpg_file, dest_folder)
                no_ext, _ = os.path.splitext(jpg_file)
                json_file = no_ext+'.json'
                shutil.copy(json_file, dest_folder)

            print "copied {0} to {1}".format(source_path, dest_folder)
