"""
utils for debug in pytorch training
Quan Yuan
2018-06-15
"""
from torch.autograd import Variable
import numpy
import os
import scipy.misc

def dump_images_in_batch(images_5d, output_folder):
    images_5d_np = images_5d.cpu().numpy()# *numpy.array([0.229, 0.224, 0.225])+numpy.array([0.486, 0.459, 0.408]))*255+128
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    s = images_5d_np.shape
    for i in range(s[0]):
        for j in range(s[1]):
            image_chw = numpy.squeeze(images_5d_np[i,j,:,:,:])
            image = image_chw.transpose((1, 2, 0)) #.astype(numpy.uint8)
            image = ((image*numpy.array([0.229, 0.224, 0.225])+numpy.array([0.486, 0.459, 0.408]))*255).astype(numpy.uint8)
            image_path = os.path.join(output_folder, str(i)+'_'+str(j)+'.jpg')
            scipy.misc.imsave(image_path, image)
    print 'all saved to {0}'.format(output_folder)


def dump_images_4d_in_batch(images_4d, output_folder):
    images_4d_np = images_4d.cpu().numpy()# *numpy.array([0.229, 0.224, 0.225])+numpy.array([0.486, 0.459, 0.408]))*255+128
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    s = images_4d_np.shape
    for i in range(s[0]):
            image_chw = numpy.squeeze(images_4d_np[i,:,:,:])
            image = image_chw.transpose((1, 2, 0)) #.astype(numpy.uint8)
            image = ((image*numpy.array([0.229, 0.224, 0.225])+numpy.array([0.486, 0.459, 0.408]))*255).astype(numpy.uint8)
            image_path = os.path.join(output_folder, str(i)+'.jpg')
            scipy.misc.imsave(image_path, image)
    print 'all saved to {0}'.format(output_folder)
