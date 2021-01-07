

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import datetime
import time
import numpy as np
import torch
import pickle
from numpy import unravel_index


def cmp_npy(npy_path1, npy_path2):
    npy1 = np.load(npy_path1)
    npy2 = np.load(npy_path2)
    print('npy1', npy1.shape, npy1.sum(), npy1.mean(), 'npy2', npy2.shape, npy2.sum(), npy2.mean())
    print('np.mean(np.abs))', np.mean(np.abs(npy1 - npy2)))
    diff = (npy1 - npy2) / abs(npy1)
    print('diff shape', diff.shape)
    diff_mean = np.mean(diff)
    print('diff_mean %', diff_mean, 'diff_max', np.max(diff))
    print('argmax', np.argmax(diff))
    print('argmax', unravel_index(diff.argmax(), diff.shape))

    diff_flat = diff.reshape(-1)
    print(diff_flat.shape)
    cnt = 0
    for e in diff_flat:
        if e >= 1e-6:
            cnt += 1
    print('>= 1e-6 cnt', cnt)


def cmp_npy_img(npy_path1, npy_path2):
    print('npy_path1', npy_path1)
    print('npy_path2', npy_path2)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    npy1 = np.load(npy_path1)
    npy2 = np.load(npy_path2)
    img1 = np.transpose(npy1[0, :, :, :], (1,2,0))
    img2 = np.transpose(npy2[0, :, :, :], (1,2,0))
    img1 = ((img1 * std) + mean) * 255
    uimg1 = img1.astype(np.uint8)
    import cv2
    cv2.imwrite('convert/img1.jpg', uimg1[:,:,::-1])

    img2 = ((img2 * std) + mean) * 255
    uimg2 = img2.astype(np.uint8)
    cv2.imwrite('convert/img2.jpg', uimg2[:,:,::-1])
    

if __name__ == "__main__":
    npy_path1 = '/Users/liuhui29/Downloads/npy_2021/2021_debug_x_128.npy'
    npy_path2 = 'convert/pd_npy/feat_128.npy'

    #cmp_npy('demo/extract_feat_in_img.npy', 'demo/pd_input_image.npy')
    #cmp_npy(npy_path1, npy_path2)
    
    #cmp_npy('demo/2021_debug_x_128.npy', '/Users/liuhui29/Downloads/npy_2021/2021_debug_x_128.npy')
    #cmp_npy('demo/2021_debug_offset_128.npy', '/Users/liuhui29/Downloads/npy_2021/2021_debug_offset_128.npy')
    
    #cmp_npy_img('demo/extract_feat_in_img.npy', 'demo/pd_input_image.npy')
    
    #cmp_npy('/Users/liuhui29/Downloads/npy_2021/2021_debug_offset_128.npy', 'demo/2021_debug_offset_128.npy')
    #cmp_npy('/Users/liuhui29/Downloads/npy_2021/refine_anchor_128.npy', 'demo/2021_debug_refine_anchors_128.npy')
    #cmp_npy('/Users/liuhui29/Downloads/npy_2021/2021_debug_anchors_128.npy', 'demo/2021_debug_anchor_128.npy')
    
    #cmp_npy('/Users/liuhui29/Downloads/npy_odm/odm_reg_feat_0.npy', 'demo/2021_debug_offset_dcn_128.npy')

    #cmp_npy('/Users/liuhui29/Downloads/npy/get_bbox/bbox_pred_list_0.npy', 'convert/mlvl_bboxes.npy')
    
    cmp_npy('/Users/liuhui29/Downloads/npy/get_bbox/mlvl_bboxes.npy', 'convert/mlvl_bboxes.npy')