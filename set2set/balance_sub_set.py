"""
remove excessive crops from a id to balance the crop numbers
Quan Yuan
2018-08-29
"""

import os, glob
import argparse

def balance_sub_sets(folder, cap_number):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        jpgs = glob.glob(os.path.join(sub_folder_full,'*.jpg'))
        if len(jpgs) <= cap_number:
            continue
        for jpg in jpgs[cap_number:]:
            basename = os.path.splitext(jpg)[0]
            json_file = basename+'.json'
            if os.path.isfile(jpg):
                print 'remove jpg file {0}'.format(jpg)
                os.remove(jpg)
            if os.path.isfile(json_file):
                print 'remove json file {0}'.format(json_file)
                os.remove(json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='folder of ground truth crops')

    parser.add_argument('cap_number', type=int,
                        help='the cap of max number in each folder')

    args = parser.parse_args()
    balance_sub_sets(args.folder, args.cap_number)