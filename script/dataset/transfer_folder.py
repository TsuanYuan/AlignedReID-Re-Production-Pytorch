"""
transfer raw folder sets into folder, with non-conflict digit names
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

def transfer_folder(source_folder, dest_folder, id_prefix, folder_range, ishead):
    subfolders = os.listdir(source_folder)
    if folder_range[1] < 0:
        folder_range[1] = len(subfolders)
    print "total number of subfolders is {0}".format(str(len(subfolders)))
    max_id = 0
    for i in range(folder_range[0], folder_range[1]):
        subfolder = subfolders[i]
        folder_only = os.path.basename(subfolder)
        if folder_only.isdigit() == False:
            s = folder_only.split('_')
            if len(s) == 2 and s[0].isdigit() and s[1].isdigit(): # 00000001_2 format
                if int(s[0]) > max_id:
                    max_id = int(s[0])
        else:
            if int(folder_only) > max_id:
                max_id = int(folder_only)

    extra_count = 1
    for i in range(folder_range[0], folder_range[1]):
        subfolder = subfolders[i]
        folder_only = os.path.basename(subfolder)
        if folder_only.isdigit() is False:
            s = folder_only.split('_')
            if len(s) == 2 and s[0].isdigit() and s[1].isdigit():
                folder_id = max_id + extra_count
                extra_count += 1
            else:
                continue
        else:
            folder_id = int(folder_only)
        if folder_id > id_prefix:
            print "warning: folder digit {0} > id_prefix {1}".format(folder_only, str(id_prefix))
        target_folder_only = id_prefix + folder_id
        target_folder = os.path.join(dest_folder, str(target_folder_only))
        if os.path.isdir(target_folder):
            print "warning: target folder {0} already exist, will delete the original one".format(target_folder)
            # shutil.rmtree(target_folder)
        else:
            source_digit_folder = os.path.join(source_folder, subfolder)
            source_im_folder = check_body_head_sub_folders(source_digit_folder, ishead)
            if source_im_folder is None:
                continue
            # zero_folder = os.path.join(source_folder,subfolder, '0')
            # if os.path.isdir(zero_folder):
            #     shutil.rmtree(zero_folder)

            shutil.copytree(source_im_folder, target_folder, ignore=shutil.ignore_patterns('*.db'))

        print 'all done. max folder id is {0}'.format(str(max_id+extra_count))
        
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

    transfer_folder(args.raw_folder, args.save_dir, args.id_prefix, folder_range, args.head)

