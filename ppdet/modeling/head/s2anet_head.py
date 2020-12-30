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
from paddle.nn import Conv2D
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddle.nn import Layer, Sequential
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops
import numpy as np




class AnchorGenerator(object):
    """
    Examples:
        >>> anchor = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = anchor.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = paddle.to_tensor(scales)
        self.ratios = paddle.to_tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.shape[0]

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = paddle.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:] * self.scales[:]).reshape([-1])
            hs = (h * h_ratios[:] * self.scales[:]).reshape([-1])
        else:
            ws = (w * self.scales[:] * w_ratios[:]).reshape([-1])
            hs = (h * self.scales[:] * h_ratios[:]).reshape([-1])

        # yapf: disable
        base_anchors = paddle.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            axis=-1)
        #print(base_anchors)
        base_anchors = paddle.round(base_anchors)
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = paddle.meshgrid(x, y)
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='gpu'):
        # featmap_size*stride project it to original area
        paddle.set_device(device)
        base_anchors = self.base_anchors

        feat_h, feat_w = featmap_size
        shift_x = paddle.fluid.layers.range(0, feat_w, 1, 'int32') * stride
        shift_y = paddle.fluid.layers.range(0, feat_h, 1, 'int32') * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = paddle.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        #shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[ :, :] + shifts[:, :]
        all_anchors = all_anchors.reshape([-1, 4])
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='gpu'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = paddle.zeros([feat_w], dtype='uint8')
        valid_y = paddle.zeros([feat_h], dtype='uint8')
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            [valid.size(0), self.num_base_anchors]).reshape([-1])
        return valid


def delta2rbox(Rrois,
               deltas,
               means=[0, 0, 0, 0, 0],
               stds=[1, 1, 1, 1, 1],
               wh_ratio_clip=16.0 / 1000.0):
    """
    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param wh_ratio_clip:
    :return:
    """
    
    #means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    #stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    
    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)
    denorm_deltas = deltas * stds + means
    
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]
    
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
    dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)
    
    print('Rrois', Rrois.shape)
    Rroi_x = Rrois[:, 0]
    Rroi_y = Rrois[:, 1]
    Rroi_w = Rrois[:, 2]
    Rroi_h = Rrois[:, 3]
    Rroi_angle = Rrois[:, 4]
    
    gx = dx * Rroi_w * paddle.cos(Rroi_angle) \
         - dy * Rroi_h * paddle.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * paddle.sin(Rroi_angle) \
         + dy * Rroi_h * paddle.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()
    
    ga = np.pi * dangle + Rroi_angle
    ga = (ga + np.PI / 4) % np.PI - np.PI / 4
    
    bboxes = paddle.stack([gx, gy, gw, gh, ga], axis=-1).view_as(deltas)
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
    print('bbox_preds', bbox_preds.shape)
    num_imgs, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred
        bboxes = delta2rbox(
            anchors,
            bbox_delta,
            means,
            stds,
            wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return paddle.stack(bboxes_list, axis=0)


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


def rect2rbox(bboxes):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = bboxes.shape[0]

    x_ctr = (bboxes[:, 2]+bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3]+bboxes[:, 1]) / 2.0
    edges1 = paddle.abs(bboxes[:, 2]-bboxes[:, 0])
    edges2 = paddle.abs(bboxes[:, 3]-bboxes[:, 1])
    angles = paddle.zeros([num_boxes], dtype=bboxes.dtype)

    inds = edges1 < edges2

    print('x_ctr', x_ctr.shape, y_ctr.shape, edges1.shape, edges2.shape)
    rboxes = paddle.stack((x_ctr, y_ctr, edges1, edges2, angles), axis=1)
    print('after stack rboxes', rboxes.shape)
    print('inds', inds.shape, inds)
    #rboxes[inds, 2] = edges2[inds]
    #rboxes[inds, 3] = edges1[inds]
    #rboxes[inds, 4] = np.pi / 2.0

    return rboxes



class AlignConv(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = paddle.vision.ops.DeformConv2D(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2,
                                      groups=groups,
                                      weight_attr=None,
                                      bias_attr=None,)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    def forward(self, x, anchors, stride):
        offset = anchor2offset(anchors, self.kernel_size, stride)
        x = self.relu(self.deform_conv(x, offset))
        return x


@register
class S2ANetHead(Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']
    
    def __init__(self,
                 stacked_convs=2,
                 feat_in=256,
                 feat_out=256,
                 num_classes=15,
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 align_conv_type='AlignConv',
                 with_ORConv=False,
                 loss=None):
        super(S2ANetHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.anchor_list = None
        self.num_classes = num_classes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(anchor_strides)
        self.target_means = target_means
        self.target_stds = target_stds
        assert align_conv_type in ['AlignConv',]
        self.align_conv_type = align_conv_type
        self.with_ORConv = with_ORConv
        self.loss = loss
        
        # anchor
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
        self.base_anchors = dict()
        
        self.fam_cls_convs = Sequential()
        self.fam_reg_convs = Sequential()
        
        fan_conv = feat_out * 3 * 3
        
        for i in range(self.stacked_convs):
            chan_in = self.feat_in if i == 0 else self.feat_out

            self.fam_cls_convs.add_sublayer(
                'fam_cls_conv_{}'.format(i),
                Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan_conv)),
                    bias_attr=ParamAttr(
                        learning_rate=1.,
                        regularizer=L2Decay(0.))))

            self.fam_cls_convs.add_sublayer('fam_cls_conv_{}_act'.format(i), ReLU())
            
            self.fam_reg_convs.add_sublayer(
                'fam_reg_conv_{}'.format(i),
                Conv2D(
                    in_channels=chan_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan_conv)),
                    bias_attr=ParamAttr(
                        learning_rate=1., # TODO: check lr
                        regularizer=L2Decay(0.))))
            self.fam_reg_convs.add_sublayer('fam_reg_conv_{}_act'.format(i), ReLU())
        
        self.fam_reg = Conv2D(self.feat_out, 5, 1)
        self.fam_cls = Conv2D(self.feat_out, self.num_classes, 1)
        
        
        if self.with_ORConv:
            self.or_conv = ORConv2d(self.feat_out, int(self.feat_out / 8), kernel_size = 3, padding = 1,
                                    arf_config = (1, 8))
        else:
            self.or_conv = Conv2D(self.feat_out, self.feat_out, kernel_size=3, padding=1)
        # TODO: add
        #self.or_pool = RotationInvariantPooling(256, 8)


        # ODM
        self.odm_cls_convs = Sequential()
        self.odm_reg_convs = Sequential()
        
        for i in range(self.stacked_convs):
            #ch_in = int(self.feat_out / 8) if i == 0 and self.with_ORConv else self.feat_out
            ch_in = int(self.feat_out / 8) if i == 0 else self.feat_out
            
            self.odm_cls_convs.add_sublayer(
                'odm_cls_conv_{}'.format(i),
                Conv2D(
                    in_channels=ch_in,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan_conv)),
                    bias_attr=ParamAttr(
                        learning_rate=1.,
                        regularizer=L2Decay(0.))))

            self.odm_cls_convs.add_sublayer('odm_cls_conv_{}_act'.format(i), ReLU())

            self.odm_reg_convs.add_sublayer(
                'odm_reg_conv_{}'.format(i),
                Conv2D(
                    in_channels=self.feat_out,
                    out_channels=self.feat_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan_conv)),
                    bias_attr=ParamAttr(
                        learning_rate=1.,
                        regularizer=L2Decay(0.))))

            self.odm_reg_convs.add_sublayer('odm_reg_conv_{}_act'.format(i), ReLU())

        self.odm_cls = Conv2D(self.feat_out, self.num_classes, 3, padding=1)
        self.odm_reg = Conv2D(self.feat_out, 5, 3, padding=1)


    def align_conv(self):
        pass
    
    def forward(self, feats):
        print('feats', len(feats))
        
        fam_reg_branch_list = []
        fam_cls_branch_list = []
        for i, feat in enumerate(feats):
            print('i==', i, 'in feat', feat.shape, feat.sum())
            fam_cls_feat = self.fam_cls_convs(feat)
            print('fam_cls_feat', fam_cls_feat.shape, fam_cls_feat.sum())

            fam_cls = self.fam_cls(fam_cls_feat)
            fam_cls = fam_cls.transpose([0, 2, 3, 1])
            fam_cls = fam_cls.reshape([fam_cls.shape[0], -1, self.num_classes])
            fam_cls_branch_list.append(fam_cls)

            fam_reg_feat = self.fam_reg_convs(feat)
            print('fam_reg_feat', fam_reg_feat.shape, fam_reg_feat.sum())
            
            fam_reg = self.fam_reg(fam_reg_feat)
            fam_reg = fam_reg.transpose([0, 2, 3, 1])
            fam_reg = fam_reg.reshape([fam_reg.shape[0], -1, 5])
            fam_reg_branch_list.append(fam_reg)
            
            print('fam_cls', fam_cls.shape, fam_cls.sum())
            print('fam_reg', fam_reg.shape, fam_reg.sum())
            
            # prepare anchor
            featmap_size = fam_reg.shape[-2:]
            print(self.base_anchors)
            if i in self.base_anchors.keys():
                init_anchors = self.base_anchors[i]
            else:
                device = 'cpu'
                init_anchors = self.anchor_generators[i].grid_anchors(
                    featmap_size, self.anchor_strides[i], device=device)
                init_anchors = rect2rbox(init_anchors)
                self.base_anchors[i] = init_anchors

            # TODO: do not backward
            fam_reg1 = fam_reg
            fam_reg1.stop_gradient = True
            refine_anchor = bbox_decode(
                            fam_reg,
                            init_anchors,
                            self.target_means,
                            self.target_stds)

            self.align_conv_type = "skip"
            if self.align_conv_type == 'AlignConv':
                align_feat = self.align_conv(feat, refine_anchor.clone(), stride)
            elif self.align_conv_type == 'DCN':
                align_offset = self.align_conv_offset(x)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'GA_DCN':
                align_offset = self.align_conv_offset(fam_bbox_pred)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'Conv':
                align_feat = self.align_conv(feat)

            '''
            or_feat = self.or_conv(align_feat)
            if self.with_ORConv:
                odm_cls_feat = self.or_pool(or_feat)
            else:
                odm_cls_feat = or_feat
            '''
            np_odm_cls_feat = np.load('/Users/liuhui29/Downloads/npy_odm//odm_cls_feat_{}.npy'.format(i))
            odm_cls_feat = paddle.to_tensor(np_odm_cls_feat)

            np_odm_reg_feat = np.load('/Users/liuhui29/Downloads/npy_odm//odm_reg_feat_{}.npy'.format(i))
            odm_reg_feat = paddle.to_tensor(np_odm_reg_feat)

            print('skip alignconv')
            print('odm_cls_feat', odm_cls_feat.shape, odm_cls_feat.sum())
            print('odm_reg_feat', odm_reg_feat.shape, odm_reg_feat.sum())

            odm_reg_feat = self.odm_reg_convs(odm_reg_feat)
            odm_cls_feat = self.odm_cls_convs(odm_cls_feat)
            odm_cls_score = self.odm_cls(odm_cls_feat)
            odm_bbox_pred = self.odm_reg(odm_reg_feat)
            print('odm_cls_score', odm_cls_score.shape, odm_cls_score.sum())
            print('odm_bbox_pred', odm_bbox_pred.shape, odm_bbox_pred.sum())
                
            #input('debug')
        
        print('feats out')
        fam_reg_branch = paddle.concat(fam_reg_branch_list, axis=1)
        fam_cls_branch = paddle.concat(fam_cls_branch_list, axis=1)
        
        print('fam_reg_branch:', fam_reg_branch.shape, fam_reg_branch.sum())
        print('fam_cls_branch:', fam_cls_branch.shape, fam_cls_branch.sum())
        
        

        return None
    
    def get_loss(self, inputs, head_outputs):
        return {'loss': 0.1}
        return self.loss(inputs, head_outputs)


if __name__ == '__main__':
    anchor_generator = S2ANetAnchor()
    anchor_list = anchor_generator()
    print(anchor_list)
    print(anchor_list.shape)
