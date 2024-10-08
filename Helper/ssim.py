import os
import math
import random
import numpy as np
import torch
import cv2


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim_floder(path1,path2,mode='ori'):
    file_list=os.listdir(path1)
    ss=0

    for name in file_list:
        #names = os.path.split(name)[-1]
        # print(file_list)
        im1=cv2.imread(path2 +'gt_' + name)

        # im1 = cv2.resize(im1, (256,256), interpolation = cv2.INTER_LINEAR)
        # print(im1.shape)\
        
        if mode == 'input':
            image_dir = os.path.splitext(name)[0]
            image_root = os.path.splitext(image_dir)[0] +'.png'
            im2=cv2.imread(path2+'/'+'out_' + image_root)
        else:
            im2=cv2.imread(path2+'/'+name)
        
        # m=int(im1.shape[0]/16)
        # n=int(im1.shape[1]/16)   
        # im1=im1[:m*16,:n*16,:]
        # im2=im2[:,:,:]
        s=calculate_ssim(im1,im2)
        ss+=s
    return(ss/len(file_list))