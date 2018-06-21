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
    images_5d_np = images_5d.cpu().numpy()*255+128
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    s = images_5d_np.shape
    for i in range(s[0]):
        for j in range(s[1]):
            image_chw = numpy.squeeze(images_5d_np[i,j,:,:,:])
            image = image_chw.transpose((1, 2, 0)).astype(numpy.uint8)
            image_path = os.path.join(output_folder, str(i)+'_'+str(j)+'.jpg')
            scipy.misc.imsave(image_path, image)
    print 'all saved to {0}'.format(output_folder)


