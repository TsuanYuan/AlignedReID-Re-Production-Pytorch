"""
dump wanda bencmark 1000 pids to file
Quan Yuan
2018-10-04
"""

import argparse
import json

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("pid_list_file", type=str, help="files of list of pids")
    ap.add_argument("output_txt_file", type=str, help="path to output file")
    args = ap.parse_args()

    with open(args.pid_list_file, 'r') as fp:
        pid_to_tracks = json.load(fp)
    with open(args.output_txt_file, 'w') as fout:
        for pid in pid_to_tracks:
            fout.write(str(pid)+'\n')
    print "all pids are dumped to {}".format(args.output_txt_file)
