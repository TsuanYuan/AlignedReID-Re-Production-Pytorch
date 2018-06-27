"""
remove folders from a list
Quan Yuan
2018-06-27
"""
import argparse
import os, glob
import shutil

def remove_folders(ids, data_folder):
    folders = os.listdir(data_folder)
    count = 0
    for folder in folders:
        if folder.isdigit() and int(folder) in ids:
            id_folder = os.path.join(data_folder, folder)
            shutil.rmtree(id_folder)
            count += 1
            print 'removed id folder {0}'.format(id_folder)
    print '{0} folders were removed from {1}'.format(data_folder, str(count))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create test folders from list")
    parser.add_argument('data_folder', type=str, help="dataset original folder with subfolders of person id crops")
    parser.add_argument('folder_id_file', type=str, help="file of test ids")

    args = parser.parse_args()
    folder_ids = []
    with open(args.folder_id_file, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split()
            folder_ids.append(int(fields[0]))

    remove_folders(folder_ids, args.data_folder)