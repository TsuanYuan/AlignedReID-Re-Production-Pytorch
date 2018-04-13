"""create a set of masks from detection outputs
Quan Yuan
2018-04-12
"""
import json
import pycocotools.mask as mask_util
import numpy
import cPickle
import scipy.misc

def extract_masks(masks_json, output_file, full_width=1920):
    with open(masks_json, 'r') as f:
        detections = json.loads(f.read())
    count = 0
    mask_list = []
    for i, detection in enumerate(detections):
        if detection['human_detector_score'] < 0.75:
            continue
        full_mask = mask_util.decode(detection['human_segment_mscoco'])*255
        scale_ratio = full_width/float(full_mask.shape[1])
        full_mask = scipy.misc.imresize(full_mask.astype(float), scale_ratio)
        box = numpy.round(numpy.array([detection['human_box_x'], detection['human_box_y'],
                                       detection['human_box_width'], detection['human_box_height']])).astype(int)
        mask = numpy.array(full_mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]>128, dtype=numpy.uint8)
        count+=1
        mask_list.append(mask)
    with open(output_file, 'wb') as f:
        cPickle.dump(mask_list, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description="extract masks from detection outputs")
  parser.add_argument('mask_json', type=str, help="detection masks json")
  parser.add_argument('output_file', type=str, help="save_file of a list of masks")
  args = parser.parse_args()
  extract_masks(args.mask_json, args.output_file)