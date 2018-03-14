"""
plot images with rows of one query vs ranked galleries with distance
Quan Yuan
03-14-2018
"""

import pickle
import cv2

def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128)):
    color = [0, 0, 0]  # zero padding
    aspect_ratio = desired_size[0] / float(desired_size[1])
    current_ar = im.shape[0] / float(im.shape[1])
    if current_ar > aspect_ratio:  # current height is too high, pad width
        delta_w = int(round(im.shape[0] / aspect_ratio - im.shape[1]))
        left, right = delta_w / 2, delta_w - (delta_w / 2)
        new_im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    else:  # current width is too wide, pad height
        delta_h = int(round(im.shape[1] * aspect_ratio - im.shape[0]))
        top, bottom = delta_h / 2, delta_h - (delta_h / 2)
        new_im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                    value=color)

    return new_im

def plot_ims(im_folder, rows_file, im_size=(256,128)):
    with open(rows_file, 'rb') as f:
        im_rows = pickle.load(f)
        f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('im_folder', type=str, help="images folder", required=True)
    parser.add_argument('im_rows_file', type=str, help="file with image name rows", required=True)

    args = parser.parse_args()

    plot_ims(args.im_folder, args.im_rows_file)

