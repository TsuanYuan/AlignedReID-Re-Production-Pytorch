"""
merge sets of folder/ch01_2018072812380120/001,....
Quan Yuan
2018-08-12
"""
import argparse
import os, glob
import shutil

def merge_folders(source_folder, output_folder, id_iterval):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    sub_folders = next(os.walk(source_folder))[1]  # [x[0] for x in os.walk(folder)]
    count, base=0,0

    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(source_folder, sub_folder)
        id_folders = next(os.walk(sub_folder_path))[1]
        for id_folder in id_folders:
            id_folder = os.path.join(sub_folder_path, id_folder)
            new_id = int(id_folder) + base
            id_dest_folder = os.path.join(output_folder, str(new_id))
            shutil.copytree(id_folder, id_dest_folder)
            count+=1
        base+=id_iterval
    print "merged {0} ids in total from {1} to {2}".format(str(count), source_folder, output_folder)
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="merge folders Dataset")
    parser.add_argument('top_folder', type=str, help="dataset_original_folder")
    parser.add_argument('save_dir', type=str, help="save_folder")
    parser.add_argument('--id_interval', type=int, help="prefix to add on an ID", required=True,
                      default=10000)
    args = parser.parse_args()
    print 'id interval = {0}'.format(str(args.id_interval))
    merge_folders(args.top_folder, args.save_dir, args.id_interval)