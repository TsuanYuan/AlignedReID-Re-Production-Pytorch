"""
benchmark test on crops
Quan Yuan
2018-05-12
"""
import os, glob
import numpy
import cv2


def get_descriptors(top_folder,model, max_count_per_id=-1, force_compute=False, ext='.dsc'):
    id_folders = os.listdir(top_folder)
    data,item = {},{}
    for id_folder in id_folders:
        if not id_folder.isdigit():
            continue
        p = os.path.join(top_folder, id_folder)
        print 'descriptor computing in {0}'.format(p)
        crop_files = glob.glob(os.path.join(p, '*.jpg'))
        for i, crop_file in enumerate(crop_files):
            if max_count_per_id>0 and i > max_count_per_id:
                break
            descriptor_file = crop_file[:-4]+ext
            if os.path.isfile(descriptor_file) and (not force_compute):
                descriptor = numpy.fromfile(descriptor_file, dtype=numpy.float32)
            else:
                im_bgr = cv2.imread(crop_file)
                im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                box = numpy.array([0,0,im.shape[1], im.shape[0]])
                descriptor = model(im,[box])
                descriptor.tofile(descriptor_file)
            descriptor = numpy.squeeze(descriptor)
            item['descriptor'] = descriptor
            item['file'] = crop_file
            if id_folder not in data:
                data[id_folder] = []
            data[id_folder].append(item.copy())
    return data


def process(model_path, annotated_folder, long_folder, output_folder, device, force_compute_desc):
    model = aligned_reid_model.create_alignedReID_model_ml(model_path, sys_device_ids=((device,),), image_shape=(256, 128, 3), local_conv_out_channels=128, num_classes=514, num_models=2)
