"""
transform data from aligned reid training set to folders format
"""
import glob, os
import argparse
from shutil import copyfile

def process(input_folder, output_folder):
    jpgs = glob.glob(os.path.join(input_folder, '*.jpg'))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    for jpg_file in jpgs:
        _, file_only = os.path.split(jpg_file)
        us = file_only.find('_')
        if us < 0:
            print('wrong file name in {0}'.format(jpg_file))
            continue
        set_id = file_only[0:us]
        if not set_id.isdigit():
            continue
        set_folder = os.path.join(output_folder, set_id)
        if not os.path.isdir(set_folder):
            os.makedirs(set_folder)
        dest_name = os.path.join(set_folder, file_only)
        copyfile(jpg_file, dest_name)
    print 'all transfered to {0}'.format(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('input_folder', type=str, help="dataset original folder with 8_4_x.jpg format")
    parser.add_argument('output_folder', type=str, help="folder to output")

    args = parser.parse_args()
    process(args.input_folder, args.output_folder)