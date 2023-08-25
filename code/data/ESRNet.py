import os
import torch

import numpy as np

from utils import image

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).to(torch.float16).div_(255.0)



class ESRNet(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        self.lr_images = image.get_image_paths(os.path.join(dataroot))

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, index):
        lr = self.lr_images[index]
              
        # load image
        lr_img = image.imread_uint(lr, n_channels=3)

        # To tensor
        lr_img = uint2tensor3(lr_img)

        return {"lr":lr_img}
    
    
def esrNet(config):
    return ESRNet(config.dataroot)