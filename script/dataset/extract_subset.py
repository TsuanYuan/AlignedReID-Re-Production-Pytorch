"""
transform folder id dataset for alignedReID training
"""
import os
import shutil
import transfer_folder

def transfer_train_test(raw_folder_list, save_dir, test_subsets_file, prefix_base=10000):
    test_folder = os.path.join(save_dir,'test')
    train_folder = os.path.join(save_dir, 'train')
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    with open(raw_folder_list,'r') as f:
        raw_folders = [os.path.normpath(x.strip('\n')) for x in f]

    with open(test_subsets_file,'r') as f:
        test_subsets = [os.path.normpath(x.strip('\n')) for x in f]
    for i, raw_folder in enumerate(raw_folders):
        prefix = i*prefix_base # prefix to add to subset to seperate different supersets
        raw_folder_name = os.path.basename(os.path.normpath(raw_folder))
        sub_sets = os.listdir(raw_folder)
        for sub_set in sub_sets:
            sub_set_name = os.path.basename(os.path.normpath(sub_set))
            if sub_set_name.isdigit() is False:  # ignore junk/distractor folder
                continue
            sub_set_folder = transfer_folder.check_body_head_sub_folders(os.path.join(raw_folder,sub_set), ishead=False)
            if sub_set_folder is None:
                continue
            tail2_path = os.path.join(raw_folder_name, sub_set_name)
            dest_sub_set_name = str(int(sub_set_name)+prefix)
            if tail2_path in test_subsets:
                dest_folder = os.path.join(test_folder, dest_sub_set_name)
            else:
                dest_folder = os.path.join(train_folder, dest_sub_set_name)
            if os.path.isdir(dest_folder):
                print "warning: destination {0} exist for source {1}, skipped".format(dest_folder, sub_set_folder)
            else:
                shutil.copytree(sub_set_folder, dest_folder)
    print "data from {0} are transfered to training and test sets in {1}".format(str(raw_folder_list), save_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('folder_list_file', type=str, help="list of raw data folders")
    parser.add_argument('save_dir', type=str, help="saved target folder")
    parser.add_argument('test_subsets_file', type=str, help="list of person id folders for test only")

    args = parser.parse_args()

    transfer_train_test(args.folder_list_file, args.save_dir, args.test_subsets_file)