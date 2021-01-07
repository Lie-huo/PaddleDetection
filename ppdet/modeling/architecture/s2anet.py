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
    __inject__ = ['backbone', 'neck', 's2anet_head', 's2anet_bbox_post_process']

    def __init__(self, backbone, neck, s2anet_head, s2anet_bbox_post_process):
        super(S2ANet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = s2anet_head
        self.s2anet_bbox_post_process = s2anet_bbox_post_process

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
        scale_factor = self.inputs['scale_factor'][0]
        det_ploys, det_scores = self.s2anet_head.get_prediction(self.s2anet_bbox_post_process.nms_pre,
                                                                scale_factor)
        
        pred_out, nms_rois_num, index = self.s2anet_bbox_post_process.nms(det_ploys, det_scores)

        output = {
            'bbox': np.array([pred_out.numpy()]),
            'bbox_num': np.array([nms_rois_num.numpy()]),
            'im_id': self.inputs['im_id'].numpy()
        }
        #print('output array', output['bbox'], output['bbox_num'], output['im_id'])
        #print('get_pred out', output)
        return output
