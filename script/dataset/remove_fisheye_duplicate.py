"""remove duplicated pid folders
Quan Yuan
2018-08-12
"""

import argparse
import os, shutil

def remove_duplicates(top_folder, ids):
    for id in ids:
        sub_folder = os.path.join(top_folder, str(id))
        shutil.rmtree(sub_folder)
        print 'deleted duplicated pid folder {0}'.format(sub_folder)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="merge fisheye folders dataset")
    parser.add_argument('top_folder', type=str, help="dataset_original_folder")
    parser.add_argument('dup_file', type=str, help="txt file of duplicated ids")

    args = parser.parse_args()
    ids = []
    with open(args.dup_fille, 'r') as fp:
        for line in fp:
            fields = line.rstrip('\n').rstrip(' ').split(' ')
            ids.append(fields[1])
    print "ids to delete are {0}".format(str(ids))
    remove_duplicates(args.top_folder, ids)