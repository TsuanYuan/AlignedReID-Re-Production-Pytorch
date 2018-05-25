
import os, glob


def remove_bad(input_folder):
    sub_folders = os.listdir(input_folder)
    for sub_folder in sub_folders:
        full_folder = os.path.join(input_folder,sub_folder)
        jpgs = glob.glob(os.path.join(full_folder, '*.jpg'))
        for jpg_file in jpgs:
            statinfo = os.stat(jpg_file)
            if statinfo.st_size <= 0:
                print 'file size is zero {0}'.format(jpg_file)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
  parser.add_argument('raw_folder', type=str, help="dataset_original_folder")

  args = parser.parse_args()
  remove_bad(args.raw_folder)