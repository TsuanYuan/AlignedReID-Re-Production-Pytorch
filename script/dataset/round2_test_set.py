"""
create test folders from a list
"""
import os, glob
import argparse
import shutil

def load_test_sets(set_file):
    test_ids = []
    with open(set_file, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split()
            test_ids.append(fields[0])

    return test_ids

def copy_folders(test_ids, data_folder, target_folder):
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    for test_id in test_ids:
        source_folder = os.path.join(data_folder, test_id)
        if os.path.isdir(source_folder):
            target_id_folder = os.path.join(target_folder, test_id)
            shutil.copytree(source_folder, target_id_folder)
        else:
            print "cannot find test id folder {0}".format(source_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create test folders from list")
    parser.add_argument('data_folder', type=str, help="dataset original folder with subfolders of person id crops")
    parser.add_argument('test_id_file', type=str, help="file of test ids")
    parser.add_argument('target_folder', type=str, help="target folder")
    args = parser.parse_args()
    test_ids = load_test_sets(args.test_id_file)
    copy_folders(test_ids, args.data_folder, args.target_folder)