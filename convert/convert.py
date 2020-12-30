

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


def load_s2anet_torch(model_path):
    print('load model_path=', model_path)
    torch_loaded_model = torch.load(model_path, map_location=torch.device('cpu'))
    
    model_meta = torch_loaded_model['meta']
    model_state_dict = torch_loaded_model['state_dict']
    # print(model_meta)
    for k in model_state_dict.keys():
        if k.find('or_') > 0:
            print(k, model_state_dict[k].shape, model_state_dict[k].cpu().numpy().mean())
    print('rbox_head.or_conv.indices\n', model_state_dict['rbox_head.or_conv.indices'].cpu().numpy()[:,:,:,1])
    return model_state_dict


def merge_dict():
    torch_lsit = [e.strip() for e in open('s2anet_list1.txt')]
    
    pd_lsit_map = [e.strip() for e in open('weight_name_map.txt')]
    pd_lsit = [e.split(' ')[-1] for e in pd_lsit_map]
    
    fout = open('s2anet_to_paddle_map.txt', 'w')
    for i in range(265):
        t = torch_lsit[i].ljust(64, ' ')
        fout.write('{}{}\n'.format(t, pd_lsit[i]))


def convert_param(model_path):
    paddle_model_dict = {}
    
    torch_model_static = load_s2anet_torch(model_path)
    
    s2anet_2_paddle_map = {}
    paddle_2_s2anet_map = {}
    for line in open('s2anet_to_paddle_map_fpn.txt'):
        elems = line.strip().split(' ')
        torch_key = elems[0]
        pd_key = elems[-1]
        s2anet_2_paddle_map[torch_key] = pd_key
        # paddle_2_s2anet_map[pd_key] = torch_key
        
    for k in s2anet_2_paddle_map.keys():
        print('pd_k {} torch_k: {}'.format(k, s2anet_2_paddle_map[k]))
    
    for k in torch_model_static.keys():
        param = torch_model_static[k]
        print('now k is ', k)
        if k not in s2anet_2_paddle_map:
            continue
        pd_k = s2anet_2_paddle_map[k]
        print('pd_key {} torch_key {} param.shape: {}'.format(pd_k, k, param.shape))
        paddle_model_dict[pd_k] = param.cpu().numpy()

    pickle.dump(paddle_model_dict, open('paddle_s2anet.pdparams', 'wb'), protocol=2)
   
   
def verify_pd(pdparam_path):
    import paddle
    param_state_dict = paddle.load(pdparam_path)
    print(param_state_dict.keys())

    print(type(param_state_dict))
    print('\n\n')
    for k in param_state_dict.keys():
        print(k, param_state_dict[k].shape, type(param_state_dict[k]))



if __name__ == "__main__":
    model_path = '/Users/liuhui29/Downloads/s2anet_r50_fpn_1x_epoch_12_20200815.pth'
    model_path = '/Users/liuhui29/Downloads/epoch_12.pth'
    #load_s2anet_torch(model_path)
    #merge_dict()
    convert_param(model_path)
    pdparam_path = 'paddle_s2anet.pdparams'
    # pdparam_path = '/Users/liuhui29/Downloads/faster_rcnn_r50_fpn_1x_coco.pdparams'
    verify_pd(pdparam_path)
    sys.exit(0)
    
    #load_s2anet_torch(model_path)

    test_path = '/Users/liuhui29/Downloads/faster_rcnn_r50_fpn_1x_coco.pdparams'
    import paddle
    model_dict = paddle.load(test_path)
    print(type(model_dict))
    print('\n\n')
    for k in model_dict.keys():
        print(k)