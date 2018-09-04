"""
load results of different formats
Quan Yuan
2018-09-03
"""

import json
import glob, os
from common import tracking_results_pb2
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def find_video_path(video_paths, video_name):
    # assume "ch004_20180956120012_xxx" format
    vn,_ = os.path.splitext(video_name)
    vps = vn.split('_')
    for video_path in video_paths:
        _, file_only = os.path.split(video_path)
        file_base, _ = os.path.splitext(file_only)
        file_parts = file_base.split('_')
        if file_parts[0] == vps[0] and file_parts[1] == vps[1]:
            return video_path
    print 'cannot find video name {0} in video_paths like {1}'.format(video_name, video_paths[0])
    return None


def load_global_data(tracker_records, video_folder):
    video_data = {}
    video_paths = glob.glob(os.path.join(video_folder, '*.mp4'))
    item = {}
    for video_name in tracker_records:
        if len(video_folder) > 0:
            video_path = find_video_path(video_paths, video_name)
        else:
            video_path = video_name
        if video_path is None:
            continue
        if video_path not in video_data:
            video_data[video_path] = {}
        for frame_index in tracker_records[video_name]:
            frame_index = int(frame_index)
            records = tracker_records[video_name][str(frame_index)]
            for record in records:
                if frame_index not in video_data[video_path]:
                    video_data[video_path][frame_index] = []
                item['track_id'] = record[-1]
                item['human_box'] = record[0]
                if int(item['track_id']) <= 0:
                    continue
                video_data[video_path][frame_index].append(item.copy())
    return video_data


def bb_overlapped(boxAin, boxBin):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = list(boxAin)[0:4]
    boxB = list(boxBin)[0:4]

    boxA[2] += boxA[0]
    boxA[3] += boxA[1]
    boxB[2] += boxB[0]
    boxB[3] += boxB[1]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # only check overlapped on A
    overlapped = interArea / float(boxAArea)

    # return the overlapped, not intersection over union value
    return overlapped


def remove_partial_boxes(data, w_h_ratio_th=0.75):
    for frame_index in data:
        boxes = data[frame_index]
        non_partial_boxes = []
        for i, box in enumerate(boxes):
            partial_flag = False
            if type(box) == int:
                boxA = boxes[box]['human_box']
            else:
                boxA = box['human_box']
            box_w_h = float(boxA[2])/boxA[3]
            if box_w_h > w_h_ratio_th:
                partial_flag = True
            if not partial_flag:
                non_partial_boxes.append(box)
        data[frame_index] = list(non_partial_boxes)

    return data


def remove_overlapped_boxes(data, overlap_th=0.33):
    for frame_index in data:
        boxes = data[frame_index]
        non_overlap_boxes = []
        for i, box in enumerate(boxes):
            overlap_flag = False
            for j, other_box in enumerate(boxes):
                if i==j:
                    continue
                boxA = box['human_box']
                boxB = other_box['human_box']
                overlap_on_A = bb_overlapped(boxA, boxB)
                if overlap_on_A > overlap_th:
                    overlap_flag = True
                    break
            if not overlap_flag:
                non_overlap_boxes.append(box)
        data[frame_index] = list(non_overlap_boxes)

    return data


def load_dontcare_regions(camera_info_path):

    regions = json.loads(open(camera_info_path, 'r').read())
    assert 'dont_care' in regions
    dc_regions = regions['dont_care']
    return dc_regions


def check_if_box_ignored(box, dc_regions):
    if len(dc_regions) == 0:
        polygons = []
    else:
        if type(dc_regions[0]) is not list:
            dc_regions = [dc_regions]
        polygons = [Polygon([(dcr[i], dcr[i + 1]) for i in range(0, len(dcr), 2)]) for dcr in dc_regions]
    point = Point(box[0] + box[2] / 2.0, box[1] + box[3])
    invalid = (True in [polygon.contains(point) for polygon in polygons])
    return not invalid


def load_pb(result_file, video_folder, id_type):
    per_video_tracker_record, _ = load_tracking_pb_with_pid(result_file, id_type=id_type)
    name_parts = os.path.basename(result_file).split('.')
    video_name = name_parts[0] + '.' + name_parts[1]
    if len(video_folder) > 0:
        video_path = os.path.join(video_folder, video_name)
    else:
        video_path = video_name
    d = {}
    d[video_path] = per_video_tracker_record.copy()
    return d


def load_tracking_pb_with_pid(tracking_pb, remove_overlap=False, keep_partial=True, id_type='pid', use_dict=False):
    tracking_results = tracking_results_pb2.TrackingResults()
    with open(tracking_pb, 'rb') as f:
        tracking_results.ParseFromString(f.read())
    results_per_frame = {}
    first_last_frame = {}

    for detection in tracking_results.tracked_detections:
        frame_index = int(detection.frame_index)

        if id_type == 'pid':
            if hasattr(detection.labeling_result, 'pid'):
                person_id = int(detection.labeling_result.pid)
            else:
                continue
        elif id_type == 'long_tracklet_id':
            if hasattr(detection.labeling_result, 'long_tracklet_id'):
                person_id = int(detection.labeling_result.long_tracklet_id)
            else:
                continue
        elif id_type == 'tracklet_index':
            if hasattr(detection, 'tracklet_index'):
                person_id = int(detection.tracklet_index)
            else:
                continue
        else:
            raise Exception('unknown id_type {0}'.format(id_type))
        if person_id <= 0:
            continue
        if person_id not in first_last_frame:
            first_last_frame[person_id] = [frame_index, frame_index]
        first_last_frame[person_id][1] = frame_index  # update last frame
        body_box = {'human_box':[detection.human_box_x, detection.human_box_y, detection.human_box_width, detection.human_box_height],'track_id':person_id}
        if use_dict:
            if frame_index not in results_per_frame:
                results_per_frame[frame_index] = {}
            results_per_frame[frame_index][person_id]=body_box
        else:
            if frame_index not in results_per_frame:
                results_per_frame[frame_index] = []
            results_per_frame[frame_index].append(body_box)

    if remove_overlap:
        data0 = remove_overlapped_boxes(results_per_frame)
    else:
        data0 = results_per_frame

    if keep_partial:
        data = data0
    else:
        data = remove_partial_boxes(data0)

    id_count = len(first_last_frame.keys())
    print 'loaded {0} {1} from {2}'.format(str(id_count), str(id_type), str(tracking_pb))
    return data, first_last_frame


def load_json_results(json_results_file, video_folder):
    fp = open(json_results_file, 'r')
    tracker_records = json.load(fp)
    global_data = load_global_data(tracker_records, video_folder)
    return global_data
