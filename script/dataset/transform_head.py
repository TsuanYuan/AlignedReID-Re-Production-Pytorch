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
    if box[2] > image_w:
        box[2] = image_w
    if box[3] > image_h:
        box[3] = image_h
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


def transform(image_folder, annotation_folder, save_dir, num_test, num_folds, id_prefix, max_count_per_id):
    dest_image_list = crop_head_images(image_folder, annotation_folder, save_dir, id_prefix)
    transform_folder.split_train_test(dest_image_list, num_test, num_folds, save_dir)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
  parser.add_argument('crop_folder', type=str, help="folder with image crops")
  parser.add_argument('annotation_folder', type=str, help="folder with annotations keypoints, head")
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
  transform(args.crop_folder, args.annotation_folder, args.save_dir,
                   args.num_test, args.num_folds, args.id_prefix, args.max_count_per_id)
