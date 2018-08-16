import numpy as np
import cv2
import scipy.ndimage
import scipy.misc
import cPickle
import uuid
import os


class PreProcessIm(object):
  def __init__(
      self,
      crop_prob=0,
      crop_ratio=1.0,
      resize_h_w=None,
      scale=True,
      im_mean=None,
      im_std=None,
      mirror_type=None,
      batch_dims='NCHW',
      prng=np.random,
      masks_path=None
      ):
    """
    Args:
      crop_prob: the probability of each image to go through cropping
      crop_ratio: a float. If == 1.0, no cropping.
      resize_h_w: (height, width) after resizing. If `None`, no resizing.
      scale: whether to scale the pixel value by 1/255
      im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
        numpy array with shape [3]
      im_std: (Optionally) divided by image std; `None` or a tuple or list or
        numpy array with shape [3]. Dividing is applied only when subtracting
        mean is applied.
      mirror_type: How image should be mirrored; one of
        [None, 'random', 'always']
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
        'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
        uses 'NHWC'.
      prng: can be set to a numpy.random.RandomState object, in order to have
        random seed independent from the global one
    """
    self.crop_prob = crop_prob
    self.crop_ratio = crop_ratio
    self.resize_h_w = resize_h_w
    self.scale = scale
    self.im_mean = im_mean
    self.im_std = im_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims
    if masks_path is not None and len(masks_path)>0 and os.path.isfile(masks_path):
      with open(masks_path, 'rb') as mf:
        self.occlusion_masks = cPickle.load(mf)
    else:
      self.occlusion_masks = []
    self.prng = prng

  def __call__(self, im):
    return self.pre_process_im(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  @staticmethod
  def rand_crop_im(im, new_size, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    if (new_size[0] == im.shape[1]) and (new_size[1] == im.shape[0]):
      return im
    h_start = prng.randint(0, im.shape[0] - new_size[1])
    w_start = prng.randint(0, im.shape[1] - new_size[0])
    im = np.copy(
      im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])
    return im

  @staticmethod
  def rand_mask_im(im, mask_size_max, prng=np.random):
    """Crop `im` to `new_size`: [new_w, new_h]."""
    mask_size_w = prng.randint(mask_size_max/2, mask_size_max)
    mask_size_h = prng.randint(mask_size_max/2, mask_size_max)
    h_start = prng.randint(0, im.shape[0] - mask_size_h)
    w_start = prng.randint(0, im.shape[1] - mask_size_w)
    random_rgb = [prng.randint(0,256), prng.randint(0,256), prng.randint(0,256)]
    im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 0] = random_rgb[0]
    im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 1] = random_rgb[1]
    im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 2] = random_rgb[2]

    return im

  @staticmethod
  def rand_flip_lr_im(im, prng=np.random):
    """flip left-right randomly `im` to `new_size`: [new_w, new_h]."""
    if prng.rand(1) [0] > 0.5:
      im_new = np.fliplr(im)
    else:
      im_new = im
    return im_new

  @staticmethod
  def crop_pad_fixed_aspect_ratio(im, desired_size=(256, 128), head_top=False):
    color = [0, 0, 0] # zero padding
    aspect_ratio = desired_size[0]/float(desired_size[1])
    current_ar = im.shape[0]/float(im.shape[1])
    if current_ar > aspect_ratio: # current height is too high, pad width
      delta_w = int(round(im.shape[0]/aspect_ratio - im.shape[1]))
      left, right = delta_w / 2, delta_w - (delta_w / 2)
      new_im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT,
                                  value=color)
    else: # current width is too wide, pad height
      delta_h = int(round(im.shape[1]*aspect_ratio - im.shape[0]))
      if head_top:
        top, bottom = 0, delta_h
      else:
        top, bottom = delta_h/2, delta_h - (delta_h / 2)
      new_im = cv2.copyMakeBorder(im, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                  value=color)
    #debug
    import scipy.misc
    scipy.misc.imsave('/tmp/new_im.jpg', new_im)
    return new_im

  @staticmethod
  def apply_occlusion_masks(im, occlusion_mask, down_shift=(0.25, 0.75), left_right_shift=(-0.5, 0.5),im_mean=(0.486, 0.459, 0.408)):
    occlusion_mask_sc = scipy.ndimage.zoom(occlusion_mask, (float(im.shape[0]) / occlusion_mask.shape[0], float(im.shape[1]) / occlusion_mask.shape[1]), order=0)
    occlusion_mask_sc = occlusion_mask_sc.astype(np.uint8)
    down_range = np.round(im.shape[0] * np.array(down_shift)).astype(int)
    down = np.random.randint(down_range[0], down_range[1])
    left_right_range = np.round(im.shape[1] * np.array(left_right_shift)).astype(int)
    left_right = np.random.randint(left_right_range[0], left_right_range[1])
    im_mean_255 = np.round(np.array(im_mean) * 255).astype(np.uint8)
    if left_right >= 0:
      if left_right>0:
        occlusion_mask_sc = np.pad(occlusion_mask_sc, ((down, 0), (left_right, 0)), mode='constant')[0:-down, 0:-left_right]
      else:
        occlusion_mask_sc = np.pad(occlusion_mask_sc, ((down, 0), (left_right, 0)), mode='constant')[0:-down, :]
    else:
      occlusion_mask_sc = np.pad(occlusion_mask_sc, ((down, 0), (0, -left_right)), mode='constant')[0:-down, -left_right:]

    for i in range(3):
      im[:, :, i] = im[:, :, i] * (1 - occlusion_mask_sc) + occlusion_mask_sc * im_mean_255[i]
    return im

  def pre_process_im(self, im, desired_size=(256, 128), stretch=False):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""

    if stretch:
      im = cv2.resize(im, desired_size[::-1])
    else:
      im = self.crop_pad_fixed_aspect_ratio(im, desired_size, head_top=True)

    # Randomly crop a sub-image.
    if ((self.crop_ratio < 1)
        and (self.crop_prob > 0)
        and (self.prng.uniform() < self.crop_prob)):
      h_ratio = self.prng.uniform(self.crop_ratio, 1)
      w_ratio = h_ratio
      crop_h = int(im.shape[0] * h_ratio)
      crop_w = int(im.shape[1] * w_ratio)
      im = self.rand_crop_im(im, (crop_w, crop_h), prng=self.prng)

    # apply k blocks for occlusion
    #for k in range(12):
    #  im = self.rand_mask_im(im, im.shape[1]/4)
    # debug
    # import scipy.misc
    # scipy.misc.imsave('/tmp/new_im.jpg', im)


    im = self.rand_flip_lr_im(im, prng=self.prng)
    if len(self.occlusion_masks) > 0:
      occlusion_mask = self.occlusion_masks[np.random.randint(len(self.occlusion_masks))]
      im = self.apply_occlusion_masks(im, occlusion_mask)
    # for debug
    #fname = str(uuid.uuid4())+'.jpg'
    #scipy.misc.imsave(os.path.join('/tmp/occlusions/', fname), im)
    #print 'saved a masked patch at /tmp/masked_crop.jpg'
    # Resize.
    if (self.resize_h_w is not None) \
        and (self.resize_h_w != (im.shape[0], im.shape[1])):
      im = cv2.resize(im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)

    # scaled by 1/255.
    if self.scale:
      im = im / 255.

    # Subtract mean and scaled by std
    # im -= np.array(self.im_mean) # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.im_mean is not None:
      im = im - np.array(self.im_mean)
    if self.im_mean is not None and self.im_std is not None:
      im = im / np.array(self.im_std).astype(float)

    # May mirror image.
    mirrored = False
    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True

    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored