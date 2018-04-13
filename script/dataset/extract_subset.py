"""
transform folder id dataset for alignedReID training
"""
import os
import shutil
import transfer_folder
import glob
import json
import scipy.misc
import numpy
import pycocotools.mask as mask_util
import cPickle

def extract_mask(mask_json_file, score_th=0.5):
    with open(mask_json_file, 'r') as f:
        mask_data = json.load(f)
    max_score = score_th
    best_mask = None
    for box, mask in zip(mask_data['boxes'],mask_data['segments']):
        if box[4] > max_score:
            max_score = box[4]
            best_mask = mask_util.decode(mask)
    return best_mask

def align_masks(mask, occlusion_mask, down_shift=(0.35,0.75), left_right_shift=(-0.5, 0.5)):
    occlusion_mask_sc = scipy.misc.imresize(occlusion_mask, mask.shape, interp='nearest')
    down_range = numpy.round(mask.shape[0]*numpy.array(down_shift)).astype(int)
    down = numpy.random.randint(down_range[0], down_range[1])
    left_right_range = numpy.round(mask.shape[1]*numpy.array(left_right_shift)).astype(int)
    left_right = numpy.random.randint(left_right_range[0], left_right_range[1])
    if left_right>=0:
        occlusion_mask_sc = numpy.pad(occlusion_mask_sc,((down, 0), (left_right,0)),mode='constant')[0:-down, 0:-left_right]
    else:
        occlusion_mask_sc = numpy.pad(occlusion_mask_sc, ((down, 0), (0, -left_right)), mode='constant')[0:-down, -left_right:0]
    mask_both = mask*(1-occlusion_mask_sc)
    return mask_both

def apply_image_mask(sub_set_folder, dest_folder, occlusion_mask_list, im_mean=(0.486, 0.459, 0.408)):
    set_folder, id_folder = os.path.split(os.path.normpath(sub_set_folder))
    mask_folder = os.path.join(set_folder+'_mask', id_folder)
    image_files = glob.glob(os.path.join(sub_set_folder, '*.jpg'))
    im_mean_255 = numpy.round(numpy.array(im_mean)*255).astype(numpy.uint8)
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        base_name, ext = os.path.splitext(image_name)
        mask_name = base_name+'_mask.json'
        mask_file = os.path.join(mask_folder, mask_name)
        try:
            mask = extract_mask(mask_file)
        except:
            mask = None
            print 'error extracting mask from {0}'.format(mask_file)
        if mask is not None:
            image = scipy.misc.imread(image_file)
            mask_3 = numpy.dstack((mask, mask, mask))
            image_with_mask = image*mask_3
            channel_r = numpy.squeeze(image_with_mask[:,:,0])
            channel_r[channel_r==0] = im_mean_255[0]
            channel_g = numpy.squeeze(image_with_mask[:, :, 1])
            channel_g[channel_g == 0] = im_mean_255[1]
            channel_b = numpy.squeeze(image_with_mask[:, :, 2])
            channel_b[channel_b == 0] = im_mean_255[2]
            image_with_mask = numpy.dstack((channel_r, channel_g, channel_b))
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)
            dest_path = os.path.join(dest_folder, base_name+'.png')
            scipy.misc.imsave(dest_path, image_with_mask)
            occlusion_mask_list.append(mask)

def transfer_train_test(raw_folder_list, save_dir, test_subsets_file, prefix_base=10000, mask_on=False):
    test_folder = os.path.join(save_dir,'test')
    train_folder = os.path.join(save_dir, 'train')
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    if not os.path.isdir(train_folder):
        os.makedirs(train_folder)

    with open(raw_folder_list,'r') as f:
        raw_folders = [os.path.normpath(x.strip('\n')) for x in f]
    occlusion_mask_list = []
    with open(test_subsets_file,'r') as f:
        test_subsets = [os.path.normpath(x.strip('\n')) for x in f]
    for i, raw_folder in enumerate(raw_folders):
        prefix = i*prefix_base # prefix to add to subset to seperate different supersets
        raw_folder_name = os.path.basename(os.path.normpath(raw_folder))
        sub_sets = os.listdir(raw_folder)
        for sub_set in sub_sets:
            sub_set_name = os.path.basename(os.path.normpath(sub_set))
            if sub_set_name.isdigit() is False:  # ignore junk/distractor folder
                continue
            sub_set_folder = transfer_folder.check_body_head_sub_folders(os.path.join(raw_folder,sub_set), ishead=False)
            if sub_set_folder is None:
                continue
            tail2_path = os.path.join(raw_folder_name, sub_set_name)
            dest_sub_set_name = str(int(sub_set_name)+prefix)
            if tail2_path in test_subsets:
                dest_folder = os.path.join(test_folder, dest_sub_set_name)
            else:
                dest_folder = os.path.join(train_folder, dest_sub_set_name)
            if os.path.isdir(dest_folder):
                print "warning: destination {0} exist for source {1}, skipped".format(dest_folder, sub_set_folder)
            else:
                if mask_on:
                    apply_image_mask(sub_set_folder, dest_folder,occlusion_mask_list)
                else:
                    shutil.copytree(sub_set_folder, dest_folder)

    print "data from {0} are transfered to training and test sets in {1}".format(str(raw_folder_list), save_dir)
    mask_file = os.path.join(save_dir, 'mask_list.pkl')
    with open(mask_file, 'wb') as f:
        cPickle.dump(occlusion_mask_list, f, cPickle.HIGHEST_PROTOCOL)
    print "segmentation masks are saved at {0}".format(mask_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform folder Dataset. Each folder is of one ID")
    parser.add_argument('folder_list_file', type=str, help="list of raw data folders")
    parser.add_argument('save_dir', type=str, help="saved target folder")
    parser.add_argument('test_subsets_file', type=str, help="list of person id folders for test only")
    parser.add_argument('--mask_on', action='store_true',default=False, help="whether to apply masks on image")

    args = parser.parse_args()

    transfer_train_test(args.folder_list_file, args.save_dir, args.test_subsets_file, mask_on=args.mask_on)