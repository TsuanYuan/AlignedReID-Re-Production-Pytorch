"""
transformations of the image crops
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
Quan Yuan
2018-05-15
"""

from skimage import transform
import numpy, math
import random
import torch
from PIL import Image
from torchvision.transforms import functional as F
import scipy.ndimage


class Rescale_Image(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=True)
        return img


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        images = sample
        images_scaled = []
        for image in images:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w), mode='constant', preserve_range=True, anti_aliasing=False)
            images_scaled.append(img)

        return images_scaled #{'images': images_scaled, 'person_id': person_id}


class RandomHorizontalFlip(object):
    def __init__(self, prng=numpy.random):
        self.prng = prng

    def __call__(self, sample):
        #images, person_id = sample['images'], sample['person_id']
        images = sample
        images_flipped = []
        for image in images:
            if self.prng.rand(1)[0] > 0.5:
                im_new = numpy.fliplr(image)
                # import scipy.misc, os
                # scipy.misc.imsave('/tmp/im.jpg', image)
                # scipy.misc.imsave('/tmp/im_flip.jpg', im_new)
            else:
                im_new = image
            images_flipped.append(im_new)
        return images_flipped #{'images': images_cropped, 'person_id': person_id}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        #images, person_id = sample['images'], sample['person_id']
        images = sample
        images_cropped = []
        for image in images:
            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            top = numpy.random.randint(0, h - new_h)
            left = numpy.random.randint(0, w - new_w)

            image = image[top: top + new_h,
                          left: left + new_w]
            images_cropped.append(image)
        return images_cropped #{'images': images_cropped, 'person_id': person_id}


class PixelNormalizeImage(object):
    """normalize pixel value to [-1, 1] by substract 128 then divid by 255.
"""

    def __init__(self):
        pass

    def __call__(self, image):
        image_out = image/255.0
        return image_out


class PixelNormalize(object):
    """normalize pixel value to [-1, 1] by substract 128 then divid by 255.
"""

    def __init__(self):
        self.im_mean, self.im_std = [0.486, 0.459, 0.408], [0.229, 0.224, 0.225]

    def __call__(self, sample):
        # images, person_id = sample['images'], sample['person_id']

        images = sample
        images_n = []
        for image in images:
            image = image/255.0
            image = image - numpy.array(self.im_mean)
            image = image / numpy.array(self.im_std).astype(float)
            images_n.append(image)
        return images_n  # {'images': images_cropped, 'person_id': person_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        #images, person_id = sample['images'], sample['person_id']
        images = sample
        images_transposed = []
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for image in images:
            image = image.transpose((2, 0, 1))
            images_transposed.append(image)

        return torch.from_numpy(numpy.array(images_transposed)).float()
                #{'images': images_transposed,
               # 'person_id': torch.from_numpy(person_id)}

class RandomBlockMask(object):
    """
    create random blocks to mask out regions on a crop
    """
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks # size assumed as a tuple (w,h)

    @staticmethod
    def rand_mask_im(im, mask_size_max, prng=numpy.random):
        """Crop `im` to `new_size`: [new_w, new_h]."""
        mask_size_w = prng.randint(mask_size_max / 2, mask_size_max)
        mask_size_h = prng.randint(mask_size_max / 2, mask_size_max)
        h_start = prng.randint(0, im.shape[0] - mask_size_h)
        w_start = prng.randint(0, im.shape[1] - mask_size_w)
        random_rgb = [prng.randint(0, 256), prng.randint(0, 256), prng.randint(0, 256)]
        im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 0] = random_rgb[0]
        im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 1] = random_rgb[1]
        im[h_start: h_start + mask_size_h, w_start: w_start + mask_size_w, 2] = random_rgb[2]

        return im

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): list of Images to be flipped.

        Returns:
            PIL Images: Randomly cropped and resize image list.
        """
        blocked_images = []
        for img in imgs:
            blocked_image = img
            for k in range(self.num_blocks):
                blocked_image = RandomBlockMask.rand_mask_im(blocked_image, blocked_image.shape[1]/4)
            blocked_images.append(blocked_image)
        return blocked_images


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = size # size assumed as a tuple (w,h)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = img.shape[1]
        h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return i, j, h, w

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Image): list of Images to be flipped.

        Returns:
            PIL Images: Randomly cropped and resize image list.
        """
        cropped_images = []
        for img in imgs:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            cropped_images.append(F.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        return cropped_images

