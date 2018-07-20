"""
batch run detection given a list of video file and start,end frames
Quan Yuan
2018-04-26
"""
import argparse
import os
import multiprocessing
import subprocess
import time

GROUP_SIZE = 16
OVERWRITE_OUTPUT=False


def run_detection(inputs):  # video_path, detection_file, start_frame, end_frame, num_gpus, model_type):
    folder_path = inputs[0]
    ready_path = inputs[1]
    log_path = inputs[2]
    # num_processes = inputs[3]
    # current = multiprocessing.current_process()
    # id = current._identity[0]
    cmd_str = ''
    if os.path.isdir(ready_path) and (not OVERWRITE_OUTPUT):
        print 'output {0} already exist. skipped.'.format(ready_path)
        return
    elif os.path.isdir(ready_path):
        print 'output {0} already exist. will be overwritten.'.format(ready_path)

        cmd_str = "python transform_folder.py {0} {1} --num_test 0 --num_folds 1 ".format(folder_path,
                                                                                             ready_path,
                                                                                             )
    _, folder_name = os.path.split(os.path.normpath(ready_path))
    std_out_file = os.path.join(log_path, folder_name+'.stdout')
    std_error_file = os.path.join(log_path, folder_name+'.stderr')
    with open(std_out_file, "wb") as out, open(std_error_file, "wb") as err:
        p = subprocess.Popen(cmd_str, stdout=out, stderr=err, shell=True)
        print 'launched {0}'.format(cmd_str)
        p.wait()
        print 'finished {0}'.format(cmd_str)


def run(args):
    a = os.listdir(args.input_folder)
    set_folders = [folder for folder in a if os.path.isdir(os.path.join(args.input_folder, folder)) ]
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)
    input_set_folders = [os.path.join(args.input_folder, set_folder) for set_folder in set_folders]
    output_set_folders = [os.path.join(args.output_folder, set_folder) for set_folder in set_folders]
    num_commands = len(set_folders)
    start_time = time.time()
    # pool to queue trackers at gpus
    p = multiprocessing.Pool(processes=GROUP_SIZE)
    p.map(run_detection, zip(input_set_folders, output_set_folders, [args.log_folder]*num_commands))
    output_index_file = os.path.join(args.output_folder, 'transformed_folders.txt')
    with open(output_index_file, 'w') as fp:
        for ready_folder in output_set_folders:
            fp.write('{0}\n'.format(ready_folder))
    print "output ready folder list to {0}".format(output_index_file)
    finish_time = time.time()
    elapsed = finish_time - start_time
    print 'all finished in {0} for folders in {1}'.format(elapsed, args.input_folder)

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder", type=str,help="path to input folder sets")
    ap.add_argument("output_folder", type=str,help="path to output folder sets")
    ap.add_argument("--log_folder", type=str, help="path to output folder sets", default='/tmp/tranform_logs/')
    ap.add_argument("--email_address", type=str, default='qyuan@aibee.com', help='email address')
    args = ap.parse_args()
    run(args)
