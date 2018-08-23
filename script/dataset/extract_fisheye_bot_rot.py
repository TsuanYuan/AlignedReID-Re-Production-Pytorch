"""
extract bot rotated examples in fisheye
Quan Yuan
08-17-2018
"""

import argparse
import os, glob
import json
import cv2

def decode_rot_image_name(im_file):
    parts = str(im_file).split('_')
    track_id = int(parts[-3])
    frame_id = int(parts[-2])
    video_name = parts[0]+'_'+parts[1]
    return track_id, frame_id, video_name


def load_rot_ims(im_folder, save_folder):
    json_files = glob.glob(os.path.join(im_folder, '*.json'))
    for json_file in json_files:
        with open(json_file, 'r') as fp:
            data = json.load(fp)
        crop_box = data['rot_bbox']
        rot_im_base =  os.path.basename(data['rotation_img'])
        rot_image_file = os.path.join(im_folder, rot_im_base)
        image = cv2.imread(rot_image_file)
        crop = image[crop_box[1]:crop_box[1]+crop_box[3],crop_box[0]:crop_box[0]+crop_box[2],:]
        track_id, _ , video_name= decode_rot_image_name(rot_im_base)
        id_folder = os.path.join(save_folder, video_name, str(track_id))
        if not os.path.isdir(id_folder):
            os.makedirs(id_folder)
        dest_im_file = os.path.join(id_folder, rot_im_base[:-8]+'.jpg')
        cv2.imwrite(dest_im_file, crop)


def extract_rot_crops(im_folder, save_folder):
    load_rot_ims(im_folder, save_folder)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="extract fisheye bot tracklets from rot images")
    parser.add_argument('im_folder', type=str, help="folder of all rotated images")
    parser.add_argument('save_dir', type=str, help="saved target folder")
    parser.add_argument('--skip_ids', nargs='+',type=int,  default=[], help='ids to skip')
    args = parser.parse_args()
    extract_rot_crops(args.im_folder, args.save_dir)

