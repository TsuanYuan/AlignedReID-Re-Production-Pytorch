"""
split wanda data into training and test
Quan Yuan
2018-09-14
"""

import os
import argparse

def dump_training_test(list_file, output_folder, metric_learn_count, class_count, test_ratio=0.5, min_test_per_class=2):
    """
    separate training test cases for classifiers and metric learning. first class_count
    :param list_file: input list file of binary data
    :param output_folder: output folder of three index files
    :param metric_learn_count: count of pids in metric learning training. no overlap with classification pids
    :param class_count: count of pids in classification task. separate into training and test
    :param test_ratio: ratio of crops in test vs total
    :return: None
    """

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    metric_learn_file = os.path.join(output_folder, 'metric_learn.list')
    classification_train_file = os.path.join(output_folder, 'classification_train.list')
    classification_test_file = os.path.join(output_folder, 'classification_test.list')
    metric_learn_fp = open(metric_learn_file, 'w')
    classification_train_fp = open(classification_train_file, 'w')
    classification_test_fp = open(classification_test_file, 'w')
    with open(list_file) as f:
        lines = f.readlines()
        valid_count = 0
        for i, line in enumerate(lines):
            if valid_count < class_count:  # for classification, separate training and testing
                line_split = line.strip().split()
                class_id = line_split[0]
                classification_train_fp.write(str(class_id))
                classification_test_fp.write(str(class_id))
                groups = line_split[1:]
                num_images = len(groups) / 2
                test_num = int(round(num_images*test_ratio))
                if test_num < min_test_per_class:
                    continue
                else:
                    train_num = num_images - test_num
                    for j in range(num_images):
                        data_file = groups[2 * j]
                        within_idx = groups[2 * j + 1]
                        if j < train_num:
                            classification_train_fp.write(' '+data_file)
                            classification_train_fp.write(' ' + within_idx)
                        else:
                            classification_test_fp.write(' ' + data_file)
                            classification_test_fp.write(' ' + within_idx)
                    valid_count+=1
                    classification_train_fp.write('\n')
                    classification_test_fp.write('\n')
            elif i-valid_count < metric_learn_count :
                metric_learn_fp.write(line)
            else:
                break

    classification_test_fp.close()
    classification_train_fp.close()
    metric_learn_fp.close()
    print "index files are dumped to \n  {}\n  {}\n  {}".format(classification_train_file, metric_learn_file, classification_test_file)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file", type=str, help="path to input folder index file")
    ap.add_argument("output_folder", type=str, help="path to output index files")
    ap.add_argument("--metric_learn_count", type=int, help="count of training pids", default=10000)
    ap.add_argument("--classification_count", type=int, help="count of testing pids", default=5000)
    args = ap.parse_args()

    dump_training_test(args.index_file, args.output_folder, args.metric_learn_count, args.classification_count)
