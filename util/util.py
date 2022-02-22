from __future__ import print_function

import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def load_image(path):
    if(path[-3:] == 'dng'):
        import rawpy
        with rawpy.imread(path) as raw:
            img = raw.postprocess()
    elif(path[-3:]=='bmp' or path[-3:]=='jpg' or path[-3:]=='png'):
        import cv2
        return cv2.imread(path)[:,:,::-1]
    else:
        img = (255*plt.imread(path)[:,:,:3]).astype('uint8')

    return img

def save_image(image_numpy, image_path, ):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + cent) * factor
    return image_numpy.astype(imtype)

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

def imscatter(x, y, image, color=None, ax=None, zoom=1.):
    """ Auxiliary function to plot an image in the location [x, y]
        image should be an np.array in the form H*W*3 for RGB
    """
    if ax is None:
        ax = plt.gca()
    # try:
    #     image=image.numpy().transpose((1,2,0))*0.5+0.5
    #     #image = plt.imread(image)
    #     size = min(image.shape[0], image.shape[1])
    #     image = transform.resize(image[:size, :size], (256, 256))
    # except TypeError:
    #     # Likely already an array...
    #     pass
    #print(x)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        edgecolor = dict(boxstyle='round,pad=0.05',
                         edgecolor=color, lw=1) \
            if color is not None else None
        ab = AnnotationBbox(im, (x0, y0),
                            xycoords='data',
                            frameon=True,
                            bboxprops=edgecolor,
                            )
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists