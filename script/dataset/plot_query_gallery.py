"""
plot images with rows of one query vs ranked galleries with distance
Quan Yuan
03-14-2018
"""

import pickle
import cv2
import os
import numpy

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

def plot_ims(im_folder, rows_file, output_file, im_size=(256,128), top_k = 50):
    with open(rows_file, 'rb') as f:
        im_rows = pickle.load(f)
        id_rows_tf = pickle.load(f)
        id_rows = pickle.load(f)
        f.close()
    n_row = len(im_rows)
    n_per_row = min(top_k, len(im_rows[0]))
    canvas = numpy.zeros((im_size[0]*n_row, im_size[1]*n_per_row, 3), dtype=numpy.uint8)

    for k in range(n_row):
        im_row = im_rows[k]
        id_row_tf = id_rows_tf[k]
        id_row = id_rows[k]
        for j in range(n_per_row):
            im_path = os.path.join(im_folder, im_row[j])
            flag = id_row_tf[j]
            im = cv2.imread(im_path)
            im_pad = crop_pad_fixed_aspect_ratio(im, desired_size=im_size)
            im_pad = cv2.resize(im_pad, im_size[::-1])
            if flag:
                box_color =(0, 255, 0)
            else:
                box_color =(0, 0, 255)
            cv2.rectangle(im_pad, (0,0), (im_size[1]-4, im_size[0]-4), box_color, 4)
            id = id_row[j]
            cv2.putText(im_pad, str(id),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0))
            canvas[k*im_size[0]:(k+1)*im_size[0], j*im_size[1]:(j+1)*im_size[1],:] = im_pad
    cv2.imwrite(output_file, canvas)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('im_folder', type=str, help="images folder")
    parser.add_argument('im_rows_file', type=str, help="file with image name rows")
    parser.add_argument('output_file', type=str, help="output image file")
    args = parser.parse_args()

    plot_ims(args.im_folder, args.im_rows_file, args.output_file)

