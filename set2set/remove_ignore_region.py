"""
remove boxes from ignore regions
Quan Yuan
2018-08-29
"""

import os, glob, shutil
import argparse, json

def move_crops(folder, ignore_minmax, dest_folder, dry_run):
    sub_folders = next(os.walk(folder))[1]  # [x[0] for x in os.walk(folder)]
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(folder, sub_folder)
        jpgs = glob.glob(os.path.join(sub_folder_full, '*.jpg'))
        for jpg in jpgs:
            basename = os.path.splitext(jpg)[0]
            json_file = basename+'.json'
            if not os.path.isfile(json_file):
                continue
            with open(json_file, 'r') as fp:
                d = json.load(fp)
            box = d['box']
            if box[0]>=ignore_minmax[0] and box[0]<=ignore_minmax[2] and box[1] >= ignore_minmax[1] and box[1] <= ignore_minmax[3]:
                if not os.path.isdir(dest_folder):
                    os.makedirs(dest_folder)
                tail_jpg = os.path.split(jpg)[1]
                tail_json = os.path.split(json_file)[1]
                print 'jpg and json files of {0} are moved to {1}'.format(jpg, dest_folder)
                if dry_run:
                    shutil.copy2(json_file, os.path.join(dest_folder, tail_json))
                    shutil.copy2(jpg, os.path.join(dest_folder, tail_jpg))
                else:
                    shutil.move(json_file, os.path.join(dest_folder, tail_json))
                    shutil.copy2(jpg, os.path.join(dest_folder, tail_jpg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='folder of ground truth crops')

    parser.add_argument('--ignore_minmax', nargs='+', type=int, required=True,
                        help='the [xmin,ymin,xmax,ymax]')

    parser.add_argument('--dest_folder', type=str, default='/tmp/ignore/',
                        help='the dest folder to put ignored crops')

    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='dry run only copy the files to dest folder')

    args = parser.parse_args()
    move_crops(args.folder, args.ignore_minmax, args.dest_folder, args.dry_run)