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

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops
import numpy as np


def delta2rbox(Rrois,
               deltas,
               means=[0, 0, 0, 0, 0],
               stds=[1, 1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    """
    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param max_shape:
    :param wh_ratio_clip:
    :return:
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]
    
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    gx = dx * Rroi_w * paddle.cos(Rroi_angle) \
         - dy * Rroi_h * paddle.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * paddle.sin(Rroi_angle) \
         + dy * Rroi_h * paddle.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()
    
    ga = np.pi * dangle + Rroi_angle
    ga = (ga + np.PI / 4) % np.PI - np.PI / 4
    
    bboxes = paddle.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes


def bbox_decode(bbox_preds,
                anchors,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1]):
    """decode bbox from deltas
    Args:
        bbox_preds: [N,5,H,W]
        anchors: [H*W,5]
    return:
        bboxes: [N,H,W,5]
    """
    num_imgs, _, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
        bboxes = delta2rbox(
            anchors,
            bbox_delta,
            means,
            stds,
            wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return paddle.stack(bboxes_list, dim=0)


def anchor2offset(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    import torch
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        
        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        
        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset
    
    num_imgs, H, W = anchors.shape[:3]
    featmap_size = (H, W)
    offset_list = []
    for i in range(num_imgs):
        anchor = anchors[i].reshape(-1, 5)  # (NA,5)
        offset = _calc_offset(anchor, kernel_size, featmap_size, stride)
        offset_list.append(offset)  # [2*ks**2,H,W]
    offset_tensor = torch.stack(offset_list, dim=0)
    return offset_tensor


class S2ANetHead_FAM_CLS(nn.Layer):
    def __init__(self,
                 feat_in=2048,
                 feat_size=256,
                 num_anchors=9):
        super(S2ANetHead_FAM_CLS, self).__init__()
        self.feat_in = feat_in
        self.feat_size = feat_size
        self.num_anchors = num_anchors
    
    def model_arch(self, ):
        # Backbone
        self.gbd = self.inputs
        # print('inputs', self.inputs.shape)
        bb_out = self.backbone(self.gbd)
        print('bb_out', bb_out.shape)
        self.gbd.update(bb_out)
        
        # RPN
        rpn_head_out = self.rpn_head(self.gbd)
        self.gbd.update(rpn_head_out)
        
        # Anchor
        anchor_out = self.anchor(self.gbd)
        self.gbd.update(anchor_out)
        
        # Proposal BBox
        self.gbd['stage'] = 0
        proposal_out = self.proposal(self.gbd)
        self.gbd.update({'proposal_0': proposal_out})
        
        # BBox Head
        bboxhead_out = self.bbox_head(self.gbd)
        self.gbd.update({'bbox_head_0': bboxhead_out})
        
        if self.gbd['mode'] == 'infer':
            bbox_out = self.proposal.post_process(self.gbd)
            self.gbd.update(bbox_out)


@register
class S2ANetHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']
    
    def __init__(self,
                 stacked_convs=4,
                 feat_in=256,
                 feat_out=256,
                 fam_feat_channels=256,
                 num_classes=16,
                 loss='S2ANetLoss'):
        super(S2ANetHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.fam_feat_channels = fam_feat_channels
        self.anchor_list = None
        self.num_classes = num_classes
        self.num_classes = 16  # for debug
        self.loss = loss
        
        self.fam_cls_convs = Sequential()
        self.fam_reg_convs = Sequential()
        
        fan_conv = feat_out * 3 * 3
        
        for i in range(self.stacked_convs):
            chan_in = self.feat_in if i == 0 else self.fam_feat_channels
            # chan_in = self.feat_channels
            self.fam_cls_convs.add_sublayer(
                'fam_cls_conv_{}'.format(i),
                Conv2D(
                    num_channels=chan_in,
                    num_filters=self.fam_feat_channels,
                    filter_size=3,
                    act='relu',
                    padding=1,
                    param_attr=ParamAttr(
                        # name=conv_name+'_w',
                        initializer=MSRA(
                            uniform=False, fan_in=fan_conv)),
                    bias_attr=ParamAttr(
                        # name=conv_name+'_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
            
            self.fam_reg_convs.add_sublayer(
                'fam_reg_conv',
                Conv2D(
                    num_channels=chan_in,
                    num_filters=self.fam_feat_channels,
                    filter_size=3,
                    act='relu',
                    padding=1,
                    param_attr=ParamAttr(
                        # name=conv_name+'_w',
                        initializer=MSRA(
                            uniform=False, fan_in=fan_conv)),
                    bias_attr=ParamAttr(
                        # name=conv_name+'_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.))))
        
        self.fam_reg = Conv2D(self.fam_feat_channels, 5, 1)
        self.fam_cls = Conv2D(self.fam_feat_channels, self.num_classes, 1)
    
    def align_conv(self):
        pass
    
    def forward(self, feats):
        print('feats', len(feats))
        
        fam_reg_branch_list = []
        fam_cls_branch_list = []
        for i, feat in enumerate(feats):
            print('i==', i)
            fam_cls_feat = self.fam_cls_convs(feat)
            print('fam_cls_feat', fam_cls_feat.shape)
            fam_cls = self.fam_cls(fam_cls_feat)
            fam_cls = fam_cls.transpose([0, 2, 3, 1])
            fam_cls = fam_cls.reshape([fam_cls.shape[0], -1, self.num_classes])
            fam_cls_branch_list.append(fam_cls)
            
            fam_reg_feat = self.fam_reg_convs(feat)
            print('fam_reg_feat', fam_reg_feat.shape)
            fam_reg = self.fam_reg(fam_reg_feat)
            fam_reg = fam_reg.transpose([0, 2, 3, 1])
            fam_reg = fam_reg.reshape([fam_reg.shape[0], -1, 5])
            fam_reg_branch_list.append(fam_reg)
            
            print('fam_cls', fam_cls.shape)
            print('fam_reg', fam_reg.shape)
            input('debug')
        
        print('feats out')
        fam_reg_branch = paddle.concat(fam_reg_branch_list, axis=1)
        fam_cls_branch = paddle.concat(fam_cls_branch_list, axis=1)
        
        print('fam_reg_branch:', fam_reg_branch.shape)
        print('fam_cls_branch:', fam_cls_branch.shape)
        
        return None
    
    def get_loss(self, inputs, head_outputs):
        return self.loss(inputs, head_outputs)


if __name__ == '__main__':
    anchor_generator = S2ANetAnchor()
    anchor_list = anchor_generator()
    print(anchor_list)
    print(anchor_list.shape)