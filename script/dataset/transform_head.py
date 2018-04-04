"""
crop head crop with head and keypoint detection results
Quan Yuan
"""
import os
import glob
import json
import numpy
import scipy.misc
import transform_folder
import cv2

HEAD_DETECTION_TH = 0.90


def enforce_box(box, image_w, image_h):
    if box[2] >= image_w:
        box[2] = image_w-1
    if box[3] >= image_h:
        box[3] = image_h-1
    if box[0] < 0:
        box[0] = 0
    if box[1] < 0:
        box[1] = 0
    if box[0]+box[2] >= image_w:
        box[0] = image_w - 1 - box[2]
    if box[1] + box[3] >= image_h:
        box[1] = image_h - 1 - box[3]
    return box


def crop_from_box(image, box):
    box = numpy.round(box).astype(int)
    box = enforce_box(box, image.shape[1], image.shape[0])
    if len(image.shape) == 2:
        crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    elif len(image.shape) == 3:
        crop = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2],:]
    else:
        raise Exception("image dimensions other than 2 or 3!")
    return crop


def crop_extended_box(image, box=None, extension=2.0):
    # crop box at and extension ratio
    # param: image a numpy array of [w,h]
    # param: box [left, top, width, height]
    if box is None:
        box = numpy.array([0,0,image.shape[1],image.shape[0]])
    box_center = box[0:2]+box[2:4]/2
    new_wh = box[2:4]*extension
    new_box = numpy.array(list(box_center-new_wh/2)+list(new_wh))
    new_box = enforce_box(new_box, image.shape[1], image.shape[0])
    return crop_from_box(image, new_box), new_box


def extend_square(box):
    box = numpy.array(box)
    center = box[0:2] + box[2:4]/2
    radius = numpy.max(box[2:4])/2
    new_box = numpy.array([center[0]-radius, center[1]-radius, radius*2, radius*2])
    return new_box


def crop_save_head_crops(image_path, annotations, save_path):
    head_detections = annotations['head']
    key_point_detections = annotations['keypoints']
    max_box, max_keypoints = None, None
    max_th = HEAD_DETECTION_TH
    image = cv2.imread(image_path)
    for key_point_detection in key_point_detections:
        sum_weights = numpy.sum(numpy.array(key_point_detection[3]))
        if numpy.mean(sum_weights) > max_th:
            max_keypoints = key_point_detection
            for point in zip(key_point_detection[0], key_point_detection[1]):
                cv2.circle(image, (int(point[0]), int(point[1])),5,(0,255,0), -1)
            cv2.imshow('kps', image)
            cv2.waitKey()
    for head_detection in head_detections:
        if head_detection[4] >= max_th:
            max_th = head_detection[4]
            max_box = head_detection

    if max_box is None:
        return False
    else:
        image = scipy.misc.imread(image_path)
        crop_box = numpy.array(max_box)
        crop_box[2:4] = crop_box[2:4] - crop_box[0:2]
        crop_box = extend_square(crop_box)
        head_crop, new_box = crop_extended_box(image, box=crop_box, extension=2.0)
        scipy.misc.imsave(save_path, head_crop)
        return True


def crop_head_images(image_folder, annotation_folder, save_dir, id_prefix):
    person_folder_list = os.listdir(annotation_folder)
    dest_image_list = []
    for person_folder in person_folder_list:
        if not person_folder.isdigit():
            print '{0} is not a digit folder for person id'.format(person_folder)
        save_person_folder = os.path.join(save_dir, str(int(person_folder) + id_prefix))
        if not os.path.isdir(save_person_folder):
            os.makedirs(save_person_folder)
        json_list = glob.glob(os.path.join(annotation_folder, person_folder, '*.json'))
        for json_file in json_list:
            json_file_basename = os.path.basename(json_file)
            image_file_basename = json_file_basename[0:-5]
            image_file_path = os.path.join(image_folder, person_folder, image_file_basename)
            save_file_path = os.path.join(save_person_folder, image_file_basename)
            if not os.path.isfile(image_file_path):
                print '{0} is not a valid image path, skipped'.format(image_file_path)
            with open(json_file, 'r') as fp:
                anno = json.load(fp)
                success = crop_save_head_crops(image_file_path, anno, save_file_path)
                if success:
                    dest_image_list.append(save_file_path)

    return dest_image_list

def load_head_keypoint_mask_jsons(head_json, keypoint_json, mask_json):
    with open(head_json, 'r') as fp:
        head_data = json.load(fp)
    with open(keypoint_json, 'r') as fp:
        keypoint_data = json.load(fp)
    with open(mask_json, 'r') as fp:
        mask_data = json.load(fp)
    return head_data, keypoint_data, mask_data

def cross_check_head_keypoint_mask(head_data, keypoint_data, mask_data):
    head_box = None
    keypoint_head = None
    max_score = HEAD_DETECTION_TH
    for box in head_data['boxes']:
        if box[4] > max_score:
            head_box = box
            max_score = box[4]
    max_score = HEAD_DETECTION_TH
    for i,box in enumerate(keypoint_data['boxes']):
        if box[4] > max_score:
            max_score = box[4]
            keypoint_head = (keypoint_data['keypoints'][i][0][0], keypoint_data['keypoints'][i][1][0])
    # if keypoint head inside head box
    verified_box = None
    if head_box is None or keypoint_head is None:
        return verified_box
    if keypoint_head[0] >= head_box[0] and  keypoint_head[0] < head_box[2] and keypoint_head[1] >= head_box[1] and keypoint_head[1] < head_box[3]:
        verified_box = head_box
    return verified_box

def merge_anntoations_and_crop(image_folder, save_folder):
    mask_folder = image_folder+'_mask'
    keypoint_folder = image_folder+'_keypoint'
    head_folder = image_folder+'_head_box'
    mask_person_folders = os.listdir(mask_folder)
    dest_images = []
    dest_image_dir = os.path.join(save_folder, 'images')
    if not os.path.isdir(dest_image_dir):
        os.makedirs(dest_image_dir)
    for person_folder in mask_person_folders:
        mask_jsons = glob.glob(os.path.join(mask_folder, person_folder, '*.json'))
        count = 0
        cameraIDs={}
        for mask_json in mask_jsons:
            file_only = os.path.basename(mask_json)
            file_base = file_only[0:-10]
            keypoint_json = os.path.join(keypoint_folder, person_folder, file_base+'_keypoint.json')
            if not os.path.isfile(keypoint_json):
                continue
            head_json = os.path.join(head_folder, person_folder, file_base+'_head.json')
            if not os.path.isfile(head_json):
                continue
            data = load_head_keypoint_mask_jsons(head_json, keypoint_json, mask_json)
            verified_head_box = cross_check_head_keypoint_mask(data[0], data[1], data[2])
            if verified_head_box is not None:
                count+=1
                verified_head_box = numpy.array(verified_head_box)
                verified_head_box[2:4] = verified_head_box[2:4] - verified_head_box[0:2]
                head_box_square = extend_square(verified_head_box)
                image_file = os.path.join(image_folder, person_folder, file_base+'.jpg')
                image = scipy.misc.imread(image_file)
                head_crop, new_box = crop_extended_box(image, head_box_square, extension=2.0)
                dest_path = transform_folder.transfer_one_image(image_file, dest_image_dir, int(person_folder), count, cameraIDs)

                scipy.misc.imsave(dest_path, head_crop)
                dest_images.append(os.path.basename(dest_path))
    return dest_images

def transform(image_folder, save_dir, num_test, num_folds, id_prefix):
    dest_image_list = merge_anntoations_and_crop(image_folder, save_dir)
    #dest_image_list = crop_head_images(image_folder, annotations, save_dir, id_prefix)
    transform_folder.split_train_test(dest_image_list, num_test, num_folds, save_dir)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
  parser.add_argument('crop_folder', type=str, help="folder with image crops")
  #parser.add_argument('annotation_folder', type=str, help="folder with annotations keypoints, head")
  parser.add_argument('save_dir', type=str, help="output folder")
  parser.add_argument('--id_prefix', type=int, help="prefix to add on an ID", required=False,
                      default=0)
  parser.add_argument('--max_count_per_id', type=int, help="max count to sample in one id", required=False,
                      default=1000)
  parser.add_argument('--num_test',  type=int, help="num ids in test set", required=False,
                      default=100)
  parser.add_argument('--num_folds', type=int, help="num folds in cross validation tests", required=False,
                      default=5)

  args = parser.parse_args()
  transform(args.crop_folder, args.save_dir,
                   args.num_test, args.num_folds, args.id_prefix)
