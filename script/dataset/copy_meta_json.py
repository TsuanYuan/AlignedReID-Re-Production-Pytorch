
import os, glob, shutil

def copy_meta_json(jpg_folder, src_folder):
    subfolders = os.listdir(jpg_folder)
    print "total number of subfolders is {0}".format(str(len(subfolders)))
    for subfolder in subfolders:
        folder_only = os.path.basename(subfolder)
        digit_folder_only = folder_only[0:8]
        if digit_folder_only.isdigit() is False:  # ignore junk/distractor folder
            continue
        jpg_list = glob.glob(os.path.join(jpg_folder, subfolder, '*.jpg'))
        for jpg_path in jpg_list:
            no_ext, _ = os.path.splitext(jpg_path)
            json_path = no_ext + '.json'
            _, json_file_only = os.path.split(json_path)
            src_path = os.path.join(src_folder, digit_folder_only, json_file_only)
            shutil.copyfile(src_path, json_path)

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
  parser.add_argument('jpg_folder', type=str, help="folder of jpg files to find corresponding jsons")
  parser.add_argument('src_folder', type=str, help="dataset_original_folder")
  args = parser.parse_args()
  copy_meta_json(args.jpg_folder, args.src_folder)