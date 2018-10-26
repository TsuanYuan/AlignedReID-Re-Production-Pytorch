"""
crop head crops from body crops
Quan Yuan
2018-10-26
"""
import argparse
import os
import glob
import json
import cv2
import numpy

def enforce_boundary(corner_box, im_w, im_h):
    corner_box[0] = min(max(int(round(corner_box[0])), 0), im_w - 1)
    corner_box[2] = min(max(int(round(corner_box[2])), 0), im_w - 1)
    corner_box[1] = min(max(int(round(corner_box[1])), 0), im_h - 1)
    corner_box[3] = min(max(int(round(corner_box[3])), 0), im_h - 1)
    return corner_box.astype(int)


def extend_box(corner_box, head_box_extension):
    box_w = corner_box[2] - corner_box[0]
    box_h = corner_box[3] - corner_box[1]
    radius = max([box_w, box_h]) / 2 * head_box_extension
    box_center = numpy.array([(corner_box[0] + corner_box[2]) / 2, (corner_box[1] + corner_box[3]) / 2])
    extended_corner_box = numpy.zeros(4)
    extended_corner_box[0] = box_center[0] - radius
    extended_corner_box[2] = box_center[0] + radius
    extended_corner_box[1] = box_center[1] - radius
    extended_corner_box[3] = box_center[1] + radius
    return extended_corner_box


def get_best_head_box(head_json_file, head_score_threshold, box_extension):
    if os.path.isfile(head_json_file):
        with open(head_json_file, 'r') as fp:
            head_info = json.load(fp)
        head_boxes = head_info['head_boxes']
        head_scores = head_info['scores']
        n = len(head_boxes)
        if n > 0:
            valid_heads = [head_boxes[k] for k in range(n) if head_scores[k] > head_score_threshold]
            if len(valid_heads) > 0:
                # debug only
                if len(valid_heads) > 1:
                    head_debug = 0
                best_head_corner_box = sorted(valid_heads, key=lambda x: x[1])[
                    0]  # find the smallest Y value for the highest head
                best_crop_box = extend_box(best_head_corner_box, box_extension)
                return best_crop_box
        else:
            return None
    else:
        return None

def crop_heads(input_folder, output_folder, head_score_threshold, box_extension):
    pid_folders = os.listdir(input_folder)
    for sub_folder in pid_folders:
        pid_folder = os.path.join(input_folder, sub_folder)
        output_pid_folder = os.path.join(output_folder, sub_folder)
        jhd_list = glob.glob(os.path.join(pid_folder, '*.jhd'))
        for jhd_file in jhd_list:
            head_corner_box = get_best_head_box(jhd_file, head_score_threshold, box_extension)
            if head_corner_box is not None:
                jpg_file = os.path.splitext(jhd_file)[0]+'.jpg'
                image = cv2.imread(jpg_file)
                head_corner_box = enforce_boundary(head_corner_box, image.shape[1], image.shape[0])
                head_crop = image[head_corner_box[1]:head_corner_box[3], head_corner_box[0]:head_corner_box[2], :]
                if not os.path.isdir(output_pid_folder):
                    os.makedirs(output_pid_folder)
                output_image_file = os.path.join(output_pid_folder, os.path.basename(jpg_file))
                cv2.imwrite(output_image_file, head_crop)
        print "finished cropping heads from {}".format(pid_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('folder', type=str,
                        help='folder of pid sub folders with full body crops')

    parser.add_argument('output_folder', type=str,
                        help='the output folder of head crops')

    parser.add_argument('--head_score_th', type=float, default=0.65,
                        help='threshold of head detection')
    parser.add_argument('--box_extension', type=float, default=1.2,
                        help='extension ratio to enlarge head box')

    args = parser.parse_args()
    crop_heads(args.folder, args.output_folder, args.head_score_th, args.box_extension)
    print "output head crops to {}".format(args.output_folder)
    