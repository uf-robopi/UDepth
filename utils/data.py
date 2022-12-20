"""
Loading and transforming functions for USOD10K dataset
Implementaion of converting RGB to RMI input space
"""
import os 
import pandas as pd
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
    
        
def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class TestDataset(Dataset):
    """Data loader for USOD10k dataset <image, depth map> for testing"""
    def __init__(self, root_dir, mask_dir, data_type, transform=None, use_RMI=False):
        
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.data_type = data_type
        self.transform = transform
        self.use_RMI = use_RMI

    def __len__(self):
        if self.data_type == "image":
            return 1

        elif self.data_type == "folder":
            return len(os.listdir(self.root_dir))

    def RGB_to_RMI(self, image):
        #Convert the RGB color space into RMI input space
        r, g, b = image.split()
        r = np.array(r)
        gb_max = np.maximum.reduce([np.array(g), np.array(b)])
        gray_c = np.array(ImageOps.grayscale(image))
        combined = np.stack((r, gb_max, gray_c), axis=-1)
        return Image.fromarray(combined)
    

    def __getitem__(self, idx):

        if self.data_type == "image":
            img_fn = self.root_dir
            print(img_fn)
            mask_fn = self.mask_dir

        elif self.data_type == "folder":
            img_fn = os.path.join(self.root_dir,os.listdir(self.root_dir)[idx])
            mask_fn = os.path.join(self.mask_dir,os.listdir(self.root_dir)[idx])
        
        image = (Image.open(img_fn))
        mask = (Image.open(mask_fn)).resize((320, 240))
        mask = np.array(mask)

        if self.use_RMI:
            image = self.RGB_to_RMI(image)

        sample1={'image': image, 'mask':mask, 'file_name':img_fn}

        img_fn = img_fn.split("/")[-1]

        if self.transform:  sample1 ={'image': self.transform(image), 'mask':np.array(image.resize((320, 240))), 'file_name':img_fn}
        return sample1


class ToTestTensor(object):
    """Pipelines the test data into tensors"""
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, image):
        #image= sample['image']
        
        image = image.resize((640, 480))
        image = self.to_tensor(image)

        #return {'image': image}
        return image

    def to_tensor(self, pic):
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
                raise TypeError(  'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
                             
        if isinstance(pic, np.ndarray):
            if pic.ndim==2:
                pic=pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)


def RGB_to_RMI(image):
        """Convert the RGB color space into RMI input space"""
        r, g, b = image.split()
        # Compute R channel
        r = np.array(r)
        # Compute M channel
        gb_max = np.maximum.reduce([np.array(g), np.array(b)])
        # Compute I channel
        gray_c = np.array(ImageOps.grayscale(image))
        # Combine three channels
        combined = np.stack((r, gb_max, gray_c), axis=-1)
        return Image.fromarray(combined)


def totensor(image):
    """Pipelines the inference data into tensors"""
    image = np.array(image.resize((640, 480)))
    if not (_is_numpy_image(image) or _is_pil_image(image)):
        raise TypeError(  'pic should be PIL Image or ndarray. Got {}'.format(type(image)))
    
    if isinstance(image, np.ndarray):
        if image.ndim==2:
            image=image[..., np.newaxis]   
        
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        
    return image_tensor.float().div(255)