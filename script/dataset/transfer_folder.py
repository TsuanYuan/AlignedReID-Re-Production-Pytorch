"""
transfer raw folder sets into folder, with non-conflict digit names
"""
import glob
import shutil
import os

def transfer_folder(source_folder, dest_folder, id_prefix, folder_range):
    subfolders = os.listdir(source_folder)
    if folder_range[1] < 0:
        folder_range[1] = len(subfolders)
    for i in range(folder_range[0], folder_range[1]):
        subfolder = subfolders[i]
        folder_only = os.path.basename(subfolder)
        if folder_only.isdigit() is False or int(folder_only) == 0:  # ignore junk/distractor folder
            continue
        target_folder_only = id_prefix + int(folder_only)
        target_folder = os.path.join(dest_folder, str(target_folder_only))
        if os.path.isdir(target_folder):
            print "warning: target folder {0} already exist, will delete the original one".format(target_folder)
            shutil.rmtree(target_folder)
        else:
            shutil.copytree(os.path.join(source_folder,subfolder), target_folder)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('--raw_folder', type=str, help="annotated original folder", required=True)
    parser.add_argument('--save_dir', type=str, help="saved target folder", required=True)
    parser.add_argument('--id_prefix', type=int, help="prefix of folder digits", required=True)
    parser.add_argument('--folder_range', type=str, help="range of folders to process in one batch", required=False,
                        default='0,-1')
    args = parser.parse_args()
    folder_range = args.folder_range.split(',')
    folder_range[0] = int(folder_range[0])
    folder_range[1] = int(folder_range[1])

    transfer_folder(args.raw_folder, args.save_dir, args.id_prefix, folder_range)

