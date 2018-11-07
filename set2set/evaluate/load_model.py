"""
load appearance model file
Quan Yuan
2018-09-05
"""

import os
import numpy
import torch
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
from enum import Enum
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from Model import MGNModel, SwitchClassHeadModel, PoseReIDModel, PCBModel, PlainModel, PoseReWeightModel, \
    MGNWithHead, MGNWithParts, MGNSelfAtten, PlainModelWithFeatureExposed, MGNWithPoseLayer, MGNWithHeadBox
from Model import load_ckpt


class Model_Types(Enum):
    Plain = 0
    MGN = 1
    PCB_6 = 2
    HEAD_POSE = 3
    LIMB_POSE = 4
    HEAD_ONLY = 5
    PLAIN_PARTS = 6
    HEAD_POSE_REWEIGHT = 7
    HEAD_EXTRA = 8
    LIMB_EXTRA = 9
    LIMB_ONLY = 10
    PCB_3 =11
    MGN_SELF_ATTEN = 12
    HEAD_PLAIN = 13
    ALPHA_MGN = 14
    HEAD_BOX_ATTEN = 15

class AppearanceModelForward(object):
    def __init__(self, model_path, aux_model_path = None, device_ids=(0,), desired_size=(256, 128), batch_max=128, skip_fc=True):
        self.im_mean, self.im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]
        torch.cuda.set_device(min(device_ids))
        model_file = os.path.split(model_path)[1]
        self.aux_model_ws = None
        self.require_keypoints = False
        self.require_head_box = False
        self.box_extension = 1.0
        if model_file.find('alpha_mgn') >= 0:
            from AlphaPoseLoader import AlphaPoseLoader
            model = MGNWithPoseLayer()
            self.model_type = Model_Types.ALPHA_MGN
            pose_model = AlphaPoseLoader(device_ids[0])
            self.aux_model_ws = DataParallel(pose_model, device_ids=device_ids)
            self.aux_model_ws.eval()
        elif model_file.find('head_plain') >= 0:
            model = PlainModel(base_model='resnet34')
            self.model_type = Model_Types.HEAD_PLAIN
        elif model_file.find('mgn_self_atten') >= 0:
            #model = MGNSelfAtten().cuda(device=device_ids[0])
            model = MGNSelfAtten(sum_weights=False).cuda(device=device_ids[0])
            self.model_type = Model_Types.MGN_SELF_ATTEN
            print "use mgn self atten model with concat"
        elif model_file.find('mgn') >= 0:
            model = MGNModel().cuda(device=device_ids[0])
            self.model_type = Model_Types.MGN
            print "use mgn model"
        elif  model_file.find('plain_parts') >= 0:
            model = SwitchClassHeadModel(parts_model=True).cuda(device=device_ids[0])
            self.model_type = Model_Types.PLAIN_PARTS
        elif model_file.find('plain') >= 0:
            model = PlainModel().cuda(device=device_ids[0])
            self.model_type = Model_Types.Plain
        elif model_file.find('pcb') >= 0:
            model = PCBModel(backbone='resnet101').cuda(device=device_ids[0])
            desired_size = (384, 128)
            self.model_type = Model_Types.PCB_6
        elif model_file.find('head_only') >= 0:
            pose_ids = (2,)
            model = PoseReIDModel(pose_ids=pose_ids, no_global=True).cuda(device=device_ids[0])
            self.model_type = Model_Types.HEAD_ONLY
            self.require_keypoints = True
            print "head only model!"
        elif model_file.find('limbs_only') >= 0:
            pose_ids = (2,9,10,15,16)
            model = PoseReIDModel(pose_ids=pose_ids, no_global=True).cuda(device=device_ids[0])
            self.model_type = Model_Types.LIMB_ONLY
            self.require_keypoints = True
            print "limbs only model!"
        elif model_file.find('head_pose_reweight') >= 0:
            pose_ids = (0, 2, 4)
            model = PoseReWeightModel(pose_ids=pose_ids).cuda(device=device_ids[0])
            self.model_type = Model_Types.HEAD_POSE_REWEIGHT
            self.require_keypoints = True
            print "head pose reweight attention model!"
        elif model_file.find('head_pose_parts') >= 0:
            pose_ids = (0, 2, 4)
            model = PoseReIDModel(pose_ids=pose_ids).cuda(device=device_ids[0])
            self.require_keypoints = True
            self.model_type = Model_Types.HEAD_POSE
        elif model_file.find('head_extra') >= 0:
            pose_id = 0
            if model_file.find('head_extra_attention') >= 0:
                model = MGNWithHead(pose_id=pose_id, attention_weight=True).cuda(device=device_ids[0])
                print "head extra model with attention weight!"
            else:
                model = MGNWithHead(pose_id=pose_id).cuda(device=device_ids[0])
                print "head extra model without attention weight!"
            self.require_keypoints = True
            self.model_type = Model_Types.HEAD_EXTRA
        elif model_file.find('limbs_extra') >= 0:
            pose_ids = (2, 9, 10, 15, 16)
            if model_file.find('limbs_extra_attention') >= 0:
                model = MGNWithParts(pose_ids=pose_ids, attention_weight=True).cuda(device=device_ids[0])
                print "limbs extra model with attention weight!"
            else:
                model = MGNWithParts(pose_ids=pose_ids).cuda(device=device_ids[0])
                print "limbs extra model without attention weight!"
            self.require_keypoints = True
            self.model_type = Model_Types.HEAD_EXTRA
        elif model_file.find('limb_pose_parts') >= 0:
            pose_ids = (2,9,10,15,16)
            model = PoseReIDModel(pose_ids=pose_ids).cuda(device=device_ids[0])
            self.require_keypoints = True
            self.model_type = Model_Types.LIMB_POSE
        elif model_file.find('head_box_attention') >= 0:
            self.require_head_box = True
            self.box_extension = 1.2
            self.model_type = Model_Types.HEAD_BOX_ATTEN
            model = MGNWithHeadBox().cuda(device=device_ids[0])
        else:
            raise Exception("unknown model type!")

        self.desired_size = desired_size
        self.device_ids = device_ids
        self.batch_max = batch_max
        self.model_ws = DataParallel(model, device_ids=device_ids)
        # load the model
        load_ckpt([model], model_path, skip_fc=skip_fc)
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model_ws.eval()

    def get_num_gpus(self):
        return len(self.device_ids)

    def __del__(self):
        torch.cuda.empty_cache()

    def compute_batch_within_limit(self, image_batch, keypoints):
        global_feats = self.extract_feature(image_batch, keypoints)
        return global_feats

    def compute_features_on_batch(self, images, keypoints=None):
        normalized_patches = self.normalize_images(images)
        n = normalized_patches.shape[0]
        global_feats = None
        for i in range(0, n, self.batch_max):
            images = normalized_patches[i:i + self.batch_max, :, :, :]
            features =  self.extract_feature(images, keypoints)
            if global_feats is None:
                global_feats = features
            else:
                global_feats = numpy.concatenate((global_feats, features), axis=0)
        return global_feats

    def normalize_images(self, images):
        normalized_patches = []
        for image in images:
            image_normalized = self.crop_pad_fixed_aspect_ratio(image, desired_size=self.desired_size)
            image_normalized = cv2.resize(image_normalized, (self.desired_size[1], self.desired_size[0]))
            patch = image_normalized / 255.0
            patch = patch - numpy.array(self.im_mean)
            patch = patch / numpy.array(self.im_std).astype(float)
            patch = patch.transpose((2, 0, 1))
            normalized_patches.append(patch)
        return numpy.asarray(normalized_patches)

    def extract_feature(self, ims, extra_inputs=None):
        if self.aux_model_ws is not None:
            aux_feature = self.aux_model_ws(ims)
            global_feat = self.model_ws(ims, aux_feature)[0].data.cpu().numpy()
        elif self.require_keypoints and extra_inputs is not None:
            keypoints = Variable(torch.from_numpy(numpy.array(extra_inputs)).float())
            global_feat = self.model_ws(ims, keypoints).data.cpu().numpy()
        elif self.require_head_box and extra_inputs is not None:
            head_boxes = numpy.asarray(extra_inputs)
            corner_boxes = head_boxes[0:2] + head_boxes[2:4]
            normalized_boxes = numpy.zeros(corner_boxes.shape)
            for k in range(corner_boxes.shape[0]):
                h, w = ims[k].shape[0], ims[k].shape[1]
                normalized_boxes[k, :] = self.enforce_boundary(self.extend_box(corner_boxes[i, :]), w, h)
            normalized_boxes = Variable(torch.from_numpy(numpy.array(normalized_boxes)).float())
            global_feat = self.model_ws(ims, normalized_boxes).data.cpu().numpy()
        else:
            global_feat = self.model_ws(ims)[0].data.cpu().numpy()
        l2_norm = numpy.sqrt((global_feat * global_feat + 1e-10).sum(axis=1))
        global_feat = global_feat / (l2_norm[:, numpy.newaxis])
        return global_feat

    def crop_im(self, im, bbox):
        """
        :param im: full rgb image
        :param bbox as an array: [x,y,w,h]
        :return:
        """
        corner_box = bbox.copy()
        corner_box[2:4] += bbox[0:2]
        extended_corner_box = self.extend_box(corner_box)
        extended_corner_box = self.enforce_boundary(extended_corner_box, im.shape[1], im.shape[0])
        im_patch = im[extended_corner_box[1]:extended_corner_box[3], extended_corner_box[0]:extended_corner_box[2], :]
        return im_patch

    def enforce_boundary(self, corner_box, im_w, im_h):
        corner_box[0] = min(max(int(round(corner_box[0])), 0), im_w - 1)
        corner_box[2] = min(max(int(round(corner_box[2])), 0), im_w - 1)
        corner_box[1] = min(max(int(round(corner_box[1])), 0), im_h - 1)
        corner_box[3] = min(max(int(round(corner_box[3])), 0), im_h - 1)
        return corner_box.astype(int)

    def extend_box(self, corner_box):
        box_w = corner_box[2] - corner_box[0]
        box_h = corner_box[3] - corner_box[1]
        w_ext = box_w/2*self.box_extension
        h_ext = box_h/2*self.box_extension
        box_center = numpy.array([(corner_box[0]+corner_box[2])/2, (corner_box[1]+corner_box[3])/2])
        extended_corner_box = numpy.zeros(4)
        extended_corner_box[0] = box_center[0] - w_ext
        extended_corner_box[2] = box_center[0] + w_ext
        extended_corner_box[1] = box_center[1] - h_ext
        extended_corner_box[3] = box_center[1] + h_ext
        return extended_corner_box

    def crop_pad_fixed_aspect_ratio(self, im, desired_size, head_top=False):
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
            if head_top:
                top, bottom = 0, delta_h
            else:
                top, bottom = delta_h / 2, delta_h - (delta_h / 2)
            new_im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                        value=color)
        return new_im

    def get_model_type(self):
        return self.model_type

    def get_aux_model(self):
        return self.aux_model_ws

if __name__ == '__main__':
    import argparse, cv2

    parser = argparse.ArgumentParser(description="Test loading model on gpus")

    parser.add_argument('image_file', type=str, help="the image file")
    parser.add_argument('model_file', type=str, help="the model file")

    args = parser.parse_args()
    im = cv2.imread(args.image_file)
    im = cv2.resize(im, (128, 256))
    ims = [im] * 64

    import threading

    datas = []
    for gpu_id in [4, 5, 6]:
        data = {'device_id': gpu_id, 'model_file': args.model_file, 'ims': ims}
        datas.append(data)


    class myThread(threading.Thread):
        def __init__(self, threadID, name, data):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.name = name
            self.data = data
            self.model = AppearanceModelForward(data['model_file'], data['device_id'])

        def run(self):
            features = self.model.compute_features_on_batch(self.data['ims'])


    all_threads = []
    id_count = 0
    for i, data in enumerate(datas):
        all_threads.append(myThread(i, 'thread_{}'.format(str(i)), data))
    for t in all_threads:
        t.start()

    for t in all_threads:
        t.join()
