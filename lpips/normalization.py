from email.errors import InvalidMultipartContentTransferEncodingDefect
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def get_gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype='float64')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

    return torch.Tensor(x / np.sum(x))


def LocalContrastNorm(image,radius=9):
    """
    image: torch.Tensor , .shape => (n,channels,height,width) 
    
    radius: Gaussian filter size (int), odd
    """
    if radius%2 == 0:
        radius += 1


    n,c,h,w = image.shape[0],image.shape[1],image.shape[2],image.shape[3]

    gaussian_filter = get_gaussian_filter((1,c,radius,radius)).to(image.get_device())
    filtered_out = F.conv2d(image,gaussian_filter,padding=radius-1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    centered_image = image - filtered_out[:,:,mid:-mid,mid:-mid]
    
    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,padding=radius-1)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    per_img_mean = s_deviation.mean(dim=[1,2,3],keepdim=True)
    
    ## Divisive Normalization
    divisor = torch.maximum(per_img_mean,s_deviation)
    divisor = torch.maximum(divisor, torch.Tensor([1e-4]).to(image.get_device()))
    new_image = centered_image / divisor
    return new_image

def MeanNorm(image):
    """
    image: torch.Tensor , .shape => (n,channels,height,width) 
    
    """
    per_img_mean = image.mean(dim=[1,2,3],keepdim=True)
    #print(torch.min(image), torch.max(image))
    new_image = image - per_img_mean
    #print(torch.min(new_image), torch.max(new_image), "----")
    return new_image

def std(image):
    per_img_mean = image.mean(dim=[1,2,3],keepdim=True)
    centered_image = image - per_img_mean    
    ## Variance Calc
    radius = 9
    mid = int((radius - 1)/2)
    gaussian_filter = torch.Tensor(get_gaussian_filter((1,image.shape[1],radius,radius)))
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,padding=radius-1)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    return s_deviation

def UnitNorm(image):
    """
    image: torch.Tensor , .shape => (n,channels,height,width) 
    
    """
    per_img_mean = image.mean(dim=[1,2,3],keepdim=True)
    centered_image = image - per_img_mean
    #print(torch.min(image), torch.max(image))
    
    ## Variance Calc
    radius = 9
    mid = int((radius - 1)/2)
    gaussian_filter = get_gaussian_filter((1,image.shape[1],radius,radius)).to(image.get_device())
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,padding=radius-1)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    #print(s_deviation.mean())

    divisor = s_deviation #torch.maximum(s_deviation, torch.Tensor([1e-4]))
    new_image = centered_image / divisor
    #print(std(new_image).mean())
    #print(torch.min(new_image), torch.max(new_image), "----")
    return new_image

def NormalizeImage(image, method):
    if method == "lcn":
        return LocalContrastNorm(image,9)
    elif method == "mean":
        return MeanNorm(image)
    elif method == "unit":
        return UnitNorm(image)
    elif method == "none":
        return image
    else:
        print("normalization method not supported")
