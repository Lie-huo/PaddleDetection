# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch
import numpy as np

__all__ = ['S2ANet']


# anchor to rotated_anchor
def rect2rbox(bboxes0):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    bboxes = []
    for bbox in bboxes0:
        np_bbox = bbox
        np_bbox = np_bbox.reshape(-1, 4)
        bboxes.append(np_bbox)
    
    print('create before', bboxes[0][0:10, :])
    bboxes = np.concatenate(bboxes, axis=0)
    print('bboxes debug', bboxes.shape)
    print('create before', bboxes[0:10, :])
    
    num_boxes = bboxes.shape[0]
    
    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = np.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = np.abs(bboxes[:, 3] - bboxes[:, 1])
    # angles = bboxes.new_zeros(num_boxes)
    angles = np.zeros(num_boxes, dtype=bboxes.dtype)
    
    inds = edges1 < edges2
    
    rboxes = np.stack((x_ctr, y_ctr, edges1, edges2, angles), axis=1)
    rboxes[inds, 2] = edges2[inds]
    rboxes[inds, 3] = edges1[inds]
    rboxes[inds, 4] = np.pi / 2.0
    
    print('create after', rboxes.shape, rboxes[0:10, :])
    # print('rect2rbox return', rboxes)
    return rboxes


@register
class S2ANet(BaseArch):
    """
    S2ANet
    Args:
        backbone (object): backbone instance
        neck (object): feature pyramid network instance
        s2anet_head (object):
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'neck', 's2anet_head']

    def __init__(self, backbone, neck, s2anet_head):
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = s2anet_head

    def model_arch(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        spatial_scale = 0.0625
        
        
        print('len body_feats', len(body_feats))
        for k in body_feats:
            print(k.shape, k.sum())
            
            
        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)
        
        print('debug after FPN')
        for k in body_feats:
            print(k.shape, k.numpy().sum())

        # s2anet Head
        self.s2anet_head_outs = self.s2anet_head(body_feats)


    def get_loss(self, ):
        loss = self.s2anet_head.get_loss(self.inputs, self.s2anet_head_outs)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self, return_numpy=True):
        pred_out = self.s2anet_head.get_prediction(self.inputs)

        output = {
            'bbox': np.array([pred_out[0].numpy()]),
            'bbox_num': np.array([pred_out[1].numpy()]),
            'im_id': self.inputs['im_id'].numpy()
        }
        #print('output array', output['bbox'], output['bbox_num'], output['im_id'])
        #print('get_pred out', output)
        return output
