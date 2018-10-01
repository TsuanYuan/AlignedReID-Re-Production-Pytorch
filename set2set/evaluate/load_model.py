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
from Model import MGNModel, SwitchClassHeadModel, PoseReIDModel

class Model_Types(Enum):
    Base = 0
    MGN = 1
    PCB = 2
    HEAD_POSE = 3
    LIMB_POSE = 4


class AppearanceModelForward(object):
    def __init__(self, model_path, single_device=0):
        self.im_mean, self.im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]
        torch.cuda.set_device(single_device)
        model_file = os.path.split(model_path)[1]

        if model_file.find('mgn') >= 0:
            model = MGNModel()
            self.model_type = Model_Types.MGN
        elif model_file.find('head_pose_parts') >= 0:
            pose_ids = (0, 2, 4)
            model = PoseReIDModel(pose_ids=pose_ids)
            self.model_type = Model_Types.HEAD_POSE
        else:
            raise Exception("unknown model type!")

        self.model_ws = DataParallel(model, device_ids=(single_device,))
        # load the model
        load_ckpt([model], model_path, skip_fc=True)
        # Set eval mode.
        # Force all BN layers to use global mean and variance, also disable
        # dropout.
        self.model_ws.eval()

    def __del__(self):
        torch.cuda.empty_cache()

    def compute_features_on_batch(self, image_batch, keypoints=None):
        patches = []
        for image in image_batch:
            patch = image / 255.0
            patch = patch - numpy.array(self.im_mean)
            patch = patch / numpy.array(self.im_std).astype(float)
            patch = patch.transpose((2, 0, 1))
            patches.append(patch)
        patches = numpy.asarray(patches)
        global_feats = self.extract_feature(patches, keypoints=keypoints)
        return global_feats

    def extract_feature(self, ims, keypoints=None):
        ims = Variable(torch.from_numpy(ims).float())
        if keypoints is None:
            global_feat = self.model_ws(ims)[0].data.cpu().numpy()
        else:
            keypoints = Variable(torch.from_numpy(numpy.array(keypoints)).float())
            global_feat = self.model_ws(ims, keypoints).data.cpu().numpy()
        l2_norm = numpy.sqrt((global_feat * global_feat + 1e-10).sum(axis=1))
        global_feat = global_feat / (l2_norm[:, numpy.newaxis])
        return global_feat

    def get_model_type(self):
        return self.model_type


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True, skip_fc=False, skip_merge=False):
    """Load state_dict's of modules/optimizers from file.
    Args:
      modules_optims: A list, which members are either torch.nn.optimizer
        or torch.nn.Module.
      ckpt_file: The file path.
      load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
        to cpu type.
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    if skip_fc:
        print('skip fc layers when loading the model!')
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        if m is not None:
            if skip_fc:
                for k in sd.keys():
                    if k.find('fc') >= 0:
                        sd.pop(k, None)
            if skip_merge:
                for k in sd.keys():
                    if k.find('merge_layer') >= 0:
                        sd.pop(k, None)
            if hasattr(m, 'param_groups'):
                m.load_state_dict(sd)
            else:
                m.load_state_dict(sd, strict=False)
    if verbose:
        print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
            ckpt_file, ckpt['ep'], ckpt['scores']))
    return ckpt['ep'], ckpt['scores']


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