import torchvision
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


#
#  Extended Transforms for multiple inputs
#

class StandardTransform(object):
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        return ( (t(input) if t is not None else input) for (t, input) in zip( self.transforms, inputs ) )

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        for t in self.transforms:
            body += self._format_transform_repr(self.transform, "Transform: ")
        return '\n'.join(body)

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class CenterCrop(object):
    """Crops the given PIL Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return ( F.center_crop(img, self.size) for img in imgs )

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomScale(object):
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be scaled.
    
        Returns:
            PIL Image: Rescaled images.
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )

        if isinstance(self.interpolation, (list, tuple) ):
            return ( F.resize(img, target_size, interp) for (img, interp) in zip( imgs, self.interpolation ) )
        else:
            return ( F.resize(img, target_size, self.interpolation) for img in imgs )
    
    def __repr__(self):
        if isinstance( self.interpolation, (tuple, list) ):
            interpolate_str = ""
            interpolate_str+= (_pil_interpolation_to_str[interp] + ", ") for interp in self.interpolation
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(scale_range=({0}, {1}), interpolation={2})'.format(self.scale_range[0], self.scale_range[1], interpolate_str)

class Scale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be scaled.

        Returns:
            PIL Image: Rescaled images.
        """
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)

        if isinstance(self.interpolation, (list, tuple) ):
            return ( F.resize(img, target_size, interp) for (img, interp) in zip( imgs, self.interpolation ) )
        else:
            return ( F.resize(img, target_size, self.interpolation) for img in imgs )

    def __repr__(self):
        if isinstance( self.interpolation, (tuple, list) ):
            interpolate_str = ""
            interpolate_str+= (_pil_interpolation_to_str[interp] + ", ") for interp in self.interpolation
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, *imgs):
        """
            imgs (PIL Image): Images to be rotated.
        Returns:
            PIL Image: Rotated images.
        """
        angle = self.get_params(self.degrees)
        return ( F.rotate(img, angle, self.resample, self.expand, self.center) for img in imgs )

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be flipped.
        Returns:
            PIL Image: Randomly flipped images.
        """
        if random.random() < self.p:
            return ( F.hflip(img) for img in imgs )
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be flipped.
        Returns:
            PIL Image: Randomly flipped images.
        """

        if random.random() < self.p:
            return ( F.vflip(img) for img in imgs )
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be padded.
        Returns:
            PIL Image: Padded images.
        """
        return ( F.pad(img, self.padding, self.fill, self.padding_mode) for img in imgs )

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, do_transform):
        self.do_transform = do_transform
        self.target_type = target_type
    
    def __call__(self, *imgs):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            imgs (PIL Image or numpy.ndarray): Images to be converted to tensor.
        Returns:
            Tensor: Converted images
        """
        return ( ( F.to_tensor( img ) if dt==True else torch.from_numpy( np.array( lbl ) )  )  for (img, dt) in zip(imgs, self.do_transform))

    def __repr__(self):
        return self.__class__.__name__ + '(do_transform={})'.format( self.do_transform )

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, do_transform):
        self.mean = mean
        self.std = std
        self.do_transform = do_transform

    def __call__(self, *tensors):
        """
        Args:
            tensors (Tensor): Tensor images of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor images.
        """
        return ( ( F.normalize(tensor, self.mean, self.std) if dt==True else tensor) for (tensor, dt) in zip(tensors, self.do_transform))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, do_transform={2})'.format(self.mean, self.std, self.do_transform)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be cropped.
        Returns:
            PIL Image: Cropped images.
        """
        if self.padding > 0:
            imgs = ( F.pad(img, self.padding) for img in imgs )

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            imgs = ( F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2)) for img in imgs)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            imgs = ( F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2)) for img in imgs )

        i, j, h, w = self.get_params(imgs[0], self.size)

        return ( F.crop(img, i, j, h, w) for img in imgs )

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, *imgs):
        """
        Args:
            imgs (PIL Image): Images to be scaled.
        Returns:
            PIL Image: Rescaled images.
        """

        if isinstance(self.interpolation, (list, tuple) ):
            return ( F.resize(img, self.size, interp) for (img, interp) in zip( imgs, self.interpolation ) )
        else:
            return ( F.resize(img, self.size, self.interpolation) for img in imgs )

    def __repr__(self):
        if isinstance( self.interpolation, (tuple, list) ):
            interpolate_str = ""
            interpolate_str+= (_pil_interpolation_to_str[interp] + ", ") for interp in self.interpolation
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]

        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 
    
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, do_transform, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.do_transform = do_transform

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, *imgs):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return ( (transform(img) if dt==True else img) for (img, dt) in zip(imgs, self.do_transform) )

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'do_transform={0}'.format(self.do_transform)
        format_string += ', brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd, do_transform):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd
        self.do_transform=do_transform

    def __call__(self, *imgs):
        return ( (self.lambd(img) if dt==True else img) for (img,dt) in zip(imgs, self.do_transform)
        
    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.do_transform)
