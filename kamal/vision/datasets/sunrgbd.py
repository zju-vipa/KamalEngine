import os
from glob import glob
from PIL import Image
from .utils import colormap
from torchvision.datasets import VisionDataset

class SunRGBD(VisionDataset):
    """SUNRGBD dataset loader where the dataset is arranged as in https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    
    **Parameters:**
        - **root_dir** (string): Root directory path.
        - **mode** (string): The type of dataset: 'train' for training set, 'val'. for validation set, and 'test' for test set.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. Default: None.
        - **label_transform** (callable, optional): A function/transform that takes in the target and transform it. Default: None.
        - **loader** (callable, optional): A function to load an image given its path. By default ``default_loader`` is used.
    """

    # Default encoding for pixel value, class name, and class color
    cmap = colormap()
    def __init__(self,
                 root,
                 split='train',
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super( SunRGBD, self ).__init__( root, transform=transform, target_transform=target_transform, transforms=transforms )
        self.root = root
        self.split = split

        self.images = glob(os.path.join(self.root, 'SUNRGBD-%s_images'%self.split, '*.jpg'))
        self.labels = glob(os.path.join(self.root, '%s13labels'%self.split, '*.png'))

        self.images.sort()
        self.labels.sort()

    def __getitem__(self, idx):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """

        img, label = Image.open(self.images[idx]), Image.open(self.labels[idx])

        if self.transform is not None:
            img, label = self.transform(img, label)
        label = label-1  # void 0=>255
        return img, label

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_fn(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask.astype('uint8')+1]