import sys
sys.path.append('..')
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import os
import argparse

import numpy
import glob
import random

MAX_COUNT_PER_ID = -1
MAX_ROW_PER_IMG = 8

# Returned descriptor has the following format:
# {'folder_id': [{'file': 'file_name', 'descriptor': 'descriptor as a vector'}, ...], ...}
def get_descriptors(top_folder, ext, max_count_per_id=MAX_COUNT_PER_ID):
    id_folders = os.listdir(top_folder)
    data,item = {},{}
    for id_folder in id_folders:
    # print id_folder
        if not id_folder.isdigit():
            continue
        p = os.path.join(top_folder, id_folder)
        print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        for i, crop_file in enumerate(crop_files):
            if max_count_per_id>0 and i > max_count_per_id:
                break
            descriptor_file = crop_file[:-4]+ext
            if os.path.isfile(descriptor_file):
                descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            else:
                continue
            descriptor = numpy.squeeze(descriptor)
            item['descriptor'] = descriptor
            item['file'] = crop_file
            if id_folder not in data:
                data[id_folder] = []
            data[id_folder].append(item.copy())
    return data

def distance(a,b):
    # cosine
    d0 = (1-numpy.dot(a,b))
    # euclidean
    # d1 = numpy.linalg.norm(a-b)
    # if abs(d0*2-d1)>0.0001:
    #     raise Exception('cosine and euclidean distance not equal')
    return d0

# Returned similary matrix has the following format:
def get_similarity_matrix(descriptors, num_query_imgs, num_result_imgs):
    results = []

    for query_folder_id in descriptors.keys():
        if not is_folder_pure(query_folder_id):
            print 'Skipping folder ', query_folder_id
            continue
        query_folder_descriptors = descriptors[query_folder_id]
        # Randomly pick num_query_imgs images.
        # assert(len(query_folder_descriptors) >= num_query_imgs)
        if len(query_folder_descriptors) >= num_query_imgs:
            query_descriptors = random.sample(query_folder_descriptors, num_query_imgs)
        else:
            query_descriptors = query_folder_descriptors

        for query_item in query_descriptors:
            distances = []
            for gallery_folder_id in descriptors.keys():
                if not is_folder_pure(gallery_folder_id):
                    print 'Skipping folder ', gallery_folder_id
                    continue
                gallery_folder_descriptors = descriptors[gallery_folder_id]
                # Randomly pick num_result_imgs images.
                # assert(len(gallery_folder_descriptors) >= num_result_imgs)
                if len(gallery_folder_descriptors) >= num_result_imgs:
                    gallery_descriptors = random.sample(gallery_folder_descriptors, num_result_imgs)
                else:
                    gallery_descriptors = gallery_folder_descriptors
                for gallery_item in gallery_descriptors:
                    dist = distance(query_item['descriptor'], gallery_item['descriptor'])
                    distances.append([dist, gallery_item['file']])

            sorted_distance = sorted(distances, key=lambda distance: distance[0])
            # assert(len(sorted_distance) >= num_result_imgs)
            # results.append({'query': query_item['file'], 'result': sorted_distance[:num_result_imgs]})
            results.append({'query': query_item['file'], 'result': sorted_distance[:16]})
    # print results
    return results


def get_filename_for_display(file_path):
    path, filename = os.path.split(file_path)
    head, dirname = os.path.split(path)
    return dirname + '/' + filename

def is_folder_pure(folder_id):
    head, dirname = os.path.split(folder_id)
    return ('_' not in dirname)

def is_from_same_folder(file_path_a, file_path_b):
    path_a, filename_a = os.path.split(file_path_a)
    head_a, dirname_a = os.path.split(path_a)
    path_b, filename_b = os.path.split(file_path_b)
    head_b, dirname_b = os.path.split(path_b)
    return (dirname_a == dirname_b)

def plot_imgs(similary_matrix, output_file_prefix, img_size=(256,512), rect_color=(0,255,0)):
    num_rows = len(similary_matrix)
    num_imgs_per_row = len(similary_matrix[0]['result']) + 1 # +1 for query img
    img_width, img_height = img_size
    canvas = numpy.zeros((img_height * MAX_ROW_PER_IMG, img_width * num_imgs_per_row, 3), dtype=numpy.uint8)

    idx_row = 0
    output_suffix = 0
    for row in similary_matrix:
        idx_col = 0
        query_img = cv2.resize(cv2.imread(row['query']), img_size)
        cv2.putText(query_img, str(get_filename_for_display(row['query'])), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        canvas[idx_row * img_height : (idx_row + 1) * img_height,
            idx_col * img_width : (idx_col + 1) * img_width, :] = query_img
        idx_col += 1

        for result in row['result']:
            reuslt_img = cv2.resize(cv2.imread(result[1]), img_size)
            cv2.putText(reuslt_img, str(get_filename_for_display(result[1])), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(reuslt_img, str('%.4f'%result[0]), (10 , img_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            if is_from_same_folder(row['query'], result[1]):
                cv2.rectangle(reuslt_img, (1,1), (img_width - 1, img_height - 1), rect_color, 8)
            canvas[idx_row * img_height : (idx_row + 1) * img_height,
                idx_col * img_width : (idx_col + 1) * img_width, :] = reuslt_img
            idx_col += 1

        idx_row += 1
        if idx_row == MAX_ROW_PER_IMG:
            output_file = output_file_prefix + '_' + str(output_suffix) + '.jpg'
            cv2.imwrite(output_file, canvas)
            idx_row = 0
            output_suffix += 1
    # last image might have duplicate results from the second last one.
    if idx_row > 0:
        output_file = output_file_prefix + '_' + str(output_suffix) + '.jpg'
        cv2.imwrite(output_file, canvas)


def process(input_folder, output_path, ext, max_count_per_id, rect_color):
    descriptors = get_descriptors(input_folder, ext, max_count_per_id)
    similary_matrix = get_similarity_matrix(descriptors, 2, 2)
    plot_imgs(similary_matrix, output_path, rect_color=rect_color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_folder', type=str,
                        help='path to input crops')

    parser.add_argument('ext', type=str,
                        help='the ext of descriptor file')

    parser.add_argument('output_path', type=str,
                        help='the path to the output file')

    parser.add_argument('--max_per_id', type=int, default=100000,
                    help='number of crops per id')

    parser.add_argument('--rect_color', type=str, default='g',
                        help='colors for rectangle')


    color_dict = {'r': (0, 0, 255), 'g': (0,255,0), 'b': (255,0,0), 'w':(255,255,255), 'p':(255,0,255)}
    args = parser.parse_args()
    print 'max count per folder is {0}'.format(str(MAX_COUNT_PER_ID))
    # Remove this if you'd like to have different results for every run.
    color = color_dict[args.rect_color]
    process(args.input_folder, args.output_path, args.ext, args.max_per_id, color)
