from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import sys
from aligned_reid.model.Model  import Model

import reid_transforms
import cv2 
from PIL import Image
import os

def prepare_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    trans = reid_transforms.ReidTransform()
    img = trans(img)
    return img

def init_reid_model(model_path, NUM_CLS):
    torch.cuda.set_device(0)
    # set this to True to enable faster inference
    cudnn.benchmark = True
    model = Model(num_classes = NUM_CLS)
    # If set > 0, will resume training from a given checkpoint.
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()
    torch.no_grad()
    return model

def get_softmax_prob(model, img):
    m = nn.Softmax()
    ## img is a person patch cropped from a opencv read image
    data = prepare_img(img)
    data = data.cuda()
    data = data.unsqueeze_(0)
    output = model(data)
    # get the index of the max log-probability
    output = m(output)
    prob = output.cpu()
    label = prob.max(1, keepdim=True)[1].cpu()
    return prob, label


model_path = './checkpoint-9.pth.tar'
NUM_CLS = 581
model = init_reid_model(model_path, NUM_CLS)
test_img = 'debug/20107-ch00001_20180811131113_1151_18848.jpg'
img = cv2.imread(test_img)
prob, label = get_softmax_prob(model, img)
print(prob)
print(label)
