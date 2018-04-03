"""
transfer raw head folder with mask, kp, head folders into folder, with non-conflict digit names
"""

import shutil
import os


def check_body_head_sub_folders(folder, ishead=False):
    subfolders = os.listdir(folder)
    body_exist = False
    head_exist = False
    for subfolder in subfolders:
        basename = os.path.basename(subfolder)
        if basename.find('body') >= 0:
            body_exist = True
        if basename.find('head') >= 0:
            head_exist = True
    if body_exist and ishead==False:
        return os.path.join(folder,'body')
    elif head_exist and ishead:
        return os.path.join(folder, 'head')
    elif ishead==False and body_exist == False and head_exist == False: # no head or body subfolder but all images
        return folder
    else:
        return None

def transfer_folder(source_folder, dest_folder, id_prefix, folder_range):
    subfolders = os.listdir(source_folder)
    folder_range_local = list(folder_range)
    if folder_range_local[1] < 0:
        folder_range_local[1] = len(subfolders)
    print "total number of subfolders is {0}".format(str(len(subfolders)))
    for i in range(folder_range_local[0], folder_range_local[1]):
        subfolder = subfolders[i]
        folder_only = os.path.basename(subfolder)
        if folder_only.isdigit() is False or int(folder_only) == 0:  # ignore junk/distractor folder
            continue
        if int(folder_only) > id_prefix:
            print "warning: folder digit {0} > id_prefix {1}".format(folder_only, str(id_prefix))
        target_folder_only = id_prefix + int(folder_only)
        target_folder = os.path.join(dest_folder, str(target_folder_only))
        if os.path.isdir(target_folder):
            print "warning: target folder {0} already exist, will delete the original one".format(target_folder)

        else:
            source_digit_folder = os.path.join(source_folder, subfolder)

            shutil.copytree(source_digit_folder, target_folder, ignore=shutil.ignore_patterns('*.db'))

def transfer_all_four(raw_folder, save_dir,id_prefix, folder_range):
    transfer_folder(raw_folder, save_dir, id_prefix, folder_range)
    head_folder = raw_folder + '_head_box'
    head_save_dir = save_dir + '_head_box'
    if not os.path.isdir(head_save_dir):
        os.makedirs(head_save_dir)
    transfer_folder(head_folder, head_save_dir, id_prefix, folder_range)
    kp_folder = raw_folder + '_keypoint'
    kp_save_dir = save_dir + '_keypoint'
    if not os.path.isdir(kp_save_dir):
        os.makedirs(kp_save_dir)
    transfer_folder(kp_folder, kp_save_dir, id_prefix, folder_range)
    mask_folder = raw_folder + '_mask'
    mask_save_dir = save_dir + '_mask'
    if not os.path.isdir(mask_save_dir):
        os.makedirs(mask_save_dir)
    transfer_folder(mask_folder, mask_save_dir, id_prefix, folder_range)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('raw_folder', type=str, help="annotated original folder")
    parser.add_argument('save_dir', type=str, help="saved target folder")
    parser.add_argument('--id_prefix', type=int, help="prefix of folder digits", default=0, required=False)
    parser.add_argument('--head', action='store_true', help="whether to use head subfolder", default=False, required=False)
    parser.add_argument('--folder_range', type=str, help="range of folders to process in one batch", required=False,
                        default='0,-1')
    args = parser.parse_args()
    folder_range = args.folder_range.split(',')
    folder_range[0] = int(folder_range[0])
    folder_range[1] = int(folder_range[1])
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    transfer_all_four(args.raw_folder, args.save_dir, args.id_prefix, folder_range)

