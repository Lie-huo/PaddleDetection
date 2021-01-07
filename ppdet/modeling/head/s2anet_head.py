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
from paddle.nn import Conv2D, BatchNorm2D
import paddle.nn.functional as F
from paddle.nn import ReLU
from paddle.nn import Layer, Sequential
from paddle.nn.initializer import Normal, XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ppdet.modeling import ops
import numpy as np
from shapely.geometry import Polygon


class RotationInvariantPooling(Layer):
    def __init__(self, nInputPlane, nOrientation=8):
        super(RotationInvariantPooling, self).__init__()
        self.nInputPlane = nInputPlane
        self.nOrientation = nOrientation
        
        hiddent_dim = int(nInputPlane / nOrientation)
        self.conv = nn.Sequential(
            Conv2D(hiddent_dim, nInputPlane, 1, 1),
            BatchNorm2D(nInputPlane),
        )
    
    def forward(self, x):
        # x: [N, c, 1, w]
        ## first, max_pooling along orientation.
        N, c, h, w = x.shape
        x = paddle.reshape(x, [N, -1, self.nOrientation, h, w])
        x, _ = paddle.max(x, axis=2, keepdim=False)  # [N, nInputPlane/nOrientation, 1, w]
        # MODIFIED
        # x = self.conv(x) # [N, nInputPlane, 1, w]
        return x


class AnchorGenerator_paddle(object):
    """
    AnchorGenerator by paddle
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


class AnchorGenerator(object):
    """
    AnchorGenerator by np
    Examples:
        >>> anchor = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = anchor.grid_anchors((2, 2))
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
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

        h_ratios = np.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:] * self.scales[:]).reshape([-1])
            hs = (h * h_ratios[:] * self.scales[:]).reshape([-1])
        else:
            ws = (w * self.scales[:] * w_ratios[:]).reshape([-1])
            hs = (h * self.scales[:] * h_ratios[:]).reshape([-1])

        # yapf: disable
        base_anchors = np.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            axis=-1)
        #print(base_anchors)
        base_anchors = np.round(base_anchors)
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        yy, xx = np.meshgrid(x, y)
        #yy = yy.reshape([-1])
        #xx = xx.reshape([-1])
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16):
        # featmap_size*stride project it to original area
        base_anchors = self.base_anchors

        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w, 1, 'int32') * stride
        shift_y = np.arange(0, feat_h, 1, 'int32') * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        #shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[ :, :] + shifts[:, :]
        #all_anchors = all_anchors.reshape([-1, 4])
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='gpu'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = np.zeros([feat_w], dtype='uint8')
        valid_y = np.zeros([feat_h], dtype='uint8')
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            [valid.size(0), self.num_base_anchors]).reshape([-1])
        return valid


def delta2rbox_np(Rrois,
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
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dw = denorm_deltas[:, 2]
    dh = denorm_deltas[:, 3]
    dangle = denorm_deltas[:, 4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = np.clip(dw, -max_ratio, max_ratio)
    dh = np.clip(dh, -max_ratio, max_ratio)

    Rroi_x = Rrois[:, 0]
    Rroi_y = Rrois[:, 1]
    Rroi_w = Rrois[:, 2]
    Rroi_h = Rrois[:, 3]
    Rroi_angle = Rrois[:, 4]

    gx = dx * Rroi_w * np.cos(Rroi_angle) - dy * Rroi_h * np.sin(Rroi_angle) + Rroi_x

    gy = dx * Rroi_w * np.sin(Rroi_angle) \
         + dy * Rroi_h * np.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * np.exp(dw)
    gh = Rroi_h * np.exp(dh)

    ga = np.pi * dangle + Rroi_angle
    ga = (ga + np.pi / 4) % np.pi - np.pi / 4

    bboxes = np.stack([gx, gy, gw, gh, ga], axis=-1)
    print('bboxes', bboxes.shape, bboxes.sum())
    return bboxes


def delta2rbox_pd(Rrois,
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
    means = paddle.to_tensor(means)
    stds = paddle.to_tensor(stds)
    deltas = paddle.reshape(deltas, [-1, deltas.shape[-1]])
    print('deltas', type(deltas), deltas.shape)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dw = denorm_deltas[:, 2]
    dh = denorm_deltas[:, 3]
    dangle = denorm_deltas[:, 4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
    dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)

    print('RroisRroisRroisRroisRrois', Rrois.shape)
    Rroi_x = Rrois[:, 0]
    Rroi_y = Rrois[:, 1]
    Rroi_w = Rrois[:, 2]
    Rroi_h = Rrois[:, 3]
    Rroi_angle = Rrois[:, 4]


    gx = dx * Rroi_w * paddle.cos(Rroi_angle) - dy * Rroi_h * paddle.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * paddle.sin(Rroi_angle) + dy * Rroi_h * paddle.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()
    ga = np.pi * dangle + Rroi_angle
    ga = (ga + np.pi / 4) % np.pi - np.pi / 4
    ga = paddle.to_tensor(ga)

    for d in [gx, gy, gw, gh, ga]:
        print(type(d), d.shape, d.sum())
    gw = paddle.to_tensor(gw, dtype='float32')
    gh = paddle.to_tensor(gh, dtype='float32')
    bboxes = paddle.stack([gx, gy, gw, gh, ga], axis=-1)
    print('bboxes', bboxes.shape)
    return bboxes


def bbox_decode(bbox_preds,
                anchors,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1]):
    """decode bbox from deltas
    Args:
        bbox_preds: [N,H,W,5]
        anchors: [H*W,5]
    return:
        bboxes: [N,H,W,5]
    """
    print('bbox_preds', bbox_preds.shape)
    num_imgs, H, W, _ = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred
        anchors = paddle.to_tensor(anchors)
        bboxes = delta2rbox_pd(
            anchors,
            bbox_delta,
            means,
            stds,
            wh_ratio_clip=1e-6)
        bboxes = paddle.reshape(bboxes, [H, W, 5])
        bboxes_list.append(bboxes)
    return paddle.stack(bboxes_list, axis=0)


def anchor2offset(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        dtype = anchors.dtype
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = paddle.arange(-pad, pad + 1, dtype=dtype)
        yy, xx = paddle.meshgrid(idx, idx)
        xx = paddle.reshape(xx, [-1])
        yy = paddle.reshape(yy, [-1])

        # get sampling locations of default conv
        xc = paddle.arange(0, feat_w, dtype=dtype)
        yc = paddle.arange(0, feat_h, dtype=dtype)
        yc, xc = paddle.meshgrid(yc, xc)
        xc = paddle.reshape(xc, [-1, 1])
        yc = paddle.reshape(yc, [-1, 1])
        x_conv = xc + xx
        y_conv = yc + yy

        # get sampling locations of anchors
        # x_ctr, y_ctr, w, h, a = np.unbind(anchors, dim=1)
        x_ctr = anchors[:, 0]
        y_ctr = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]
        a = anchors[:, 4]

        x_ctr = paddle.reshape(x_ctr, [x_ctr.shape[0], 1])
        y_ctr = paddle.reshape(y_ctr, [y_ctr.shape[0], 1])
        w = paddle.reshape(w, [w.shape[0], 1])
        h = paddle.reshape(h, [h.shape[0], 1])
        a = paddle.reshape(a, [a.shape[0], 1])

        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = paddle.cos(a), paddle.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw*xx, dh*yy
        xr = cos*x-sin*y
        yr = sin*x+cos*y
        x_anchor, y_anchor = xr+x_ctr, yr+y_ctr
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = paddle.stack([offset_y, offset_x], axis=-1)
        # NA,ks*ks*2
        print('now offset', offset.shape)
        #offset = offset.reshape(anchors.size(
        #    0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        offset = paddle.reshape(offset, [offset.shape[0], -1])
        offset = paddle.transpose(offset, [1,0])
        offset = paddle.reshape(offset, [-1, feat_h, feat_w])
        return offset

    num_imgs, H, W = anchors.shape[:3]
    featmap_size = (H, W)
    offset_list = []
    for i in range(num_imgs):
        print('anchors[i]', anchors[i].shape)
        anchor = paddle.reshape(anchors[i], [-1, 5])  # (NA,5)
        offset = _calc_offset(anchor, kernel_size, featmap_size, stride)
        offset_list.append(offset)  # [2*ks**2,H,W]
    offset_tensor = paddle.stack(offset_list, axis=0)
    return offset_tensor


def rect2rbox(bboxes):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    bboxes = bboxes.reshape(-1, 4)
    num_boxes = bboxes.shape[0]

    x_ctr = (bboxes[:, 2]+bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3]+bboxes[:, 1]) / 2.0
    edges1 = np.abs(bboxes[:, 2]-bboxes[:, 0])
    edges2 = np.abs(bboxes[:, 3]-bboxes[:, 1])
    angles = np.zeros([num_boxes], dtype=bboxes.dtype)

    inds = edges1 < edges2

    print('x_ctr', x_ctr.shape, y_ctr.shape, edges1.shape, edges2.shape)

    rboxes = np.stack((x_ctr, y_ctr, edges1, edges2, angles), axis=1)
    rboxes[inds, 2] = edges2[inds]
    rboxes[inds, 3] = edges1[inds]
    rboxes[inds, 4] = np.pi / 2.0
    return rboxes


class AlignConv(Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.align_conv = paddle.vision.ops.DeformConv2D(in_channels,
                                                           out_channels,
                                                           kernel_size=self.kernel_size,
                                                           padding=(self.kernel_size - 1) // 2,
                                                           groups=groups,
                                                           weight_attr=ParamAttr(
                                                               initializer=Normal(0, 0.01)),
                                                           bias_attr=None)

    def forward(self, x, refine_anchors, stride):
        print('2021_debug align conv forward x',x.shape, x.sum(), x.mean(), 'refine_anchors', refine_anchors.shape,
              refine_anchors.sum(), refine_anchors.mean(), stride)
        np.save('demo/2021_debug_refine_anchors_{}.npy'.format(x.shape[2]), refine_anchors.numpy())
        offset = anchor2offset(refine_anchors, self.kernel_size, stride)
        print('2021_debug offset', offset.shape, offset.mean())
        # 12.479357
        print('2021_debug', self.align_conv.weight.shape, self.align_conv.weight.sum())


        # debug
        np.save('demo/2021_debug_x_{}.npy'.format(x.shape[2]), x.numpy())
        np.save('demo/2021_debug_offset_{}.npy'.format(x.shape[2]), offset.numpy())

        #x_np = np.load('/Users/liuhui29/Downloads/npy_2021/2021_debug_x_{}.npy'.format(x.shape[2]))
        #x = paddle.to_tensor(x_np, dtype=x.dtype)
        #offset_np = np.load('/Users/liuhui29/Downloads/npy_2021/2021_debug_offset_{}.npy'.format(x.shape[2]))
        #offset = paddle.to_tensor(offset_np, dtype=refine_anchors.dtype)
        #x0 =self.align_conv(x, offset)
        #print('2021_debug offset dcn not relu ', x0.shape, x0.mean())
        x = F.relu(self.align_conv(x, offset))
        print('2021_debug offset dcn', x.shape, x.mean())
        np.save('demo/2021_debug_offset_dcn_{}.npy'.format(x.shape[2]), x.numpy())
        return x

def get_refine_anchors(featmap_sizes,
                       refine_anchors,
                       img_metas,
                       is_train=False):
    num_imgs = len(img_metas)
    num_levels = len(featmap_sizes)

    refine_anchors_list = []
    for img_id, img_meta in enumerate(img_metas):
        mlvl_refine_anchors = []
        for i in range(num_levels):
            refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
            mlvl_refine_anchors.append(refine_anchor)
        refine_anchors_list.append(mlvl_refine_anchors)

    valid_flag_list = []
    if is_train:
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
    return refine_anchors_list, valid_flag_list


# NMS

def cal_line_length(point1, point2):
    import math
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                 [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
                 [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
                     + cal_line_length(combinate[i][1], dst_coordinate[1]) \
                     + cal_line_length(combinate[i][2], dst_coordinate[2]) \
                     + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)

def rbox2poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly

def cpu_nms(boxes, scores, score_threshold, iou_threshold, max_num=None):
    """
    :param boxes:[N, 8] / 'N' means not sure
    :param scores:[N, 1]
    :param score_threshold: float
    :param iou_threshold:a scalar
    :param max_num:
    :return:keep_index
    """
    # boxes format : [x1, y1, x2, y2, x3, y3, x4, y4]
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert boxes.ndim == 2
    assert scores.ndim == 1
    assert boxes.shape[-1] == 8
    assert len(boxes) == len(scores)
    
    box_copy = boxes.copy()
    score_copy = scores.copy()
    
    ignore_mask = np.where(score_copy < score_threshold)[0]
    score_copy[ignore_mask] = 0.
    
    keep_index = []
    while np.sum(score_copy) > 0.:
        # mark reserved box
        max_score_index = np.argmax(score_copy)
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = cpu_iou(box1, box_copy)
        # mark unuseful box
        # keep_mask shape [N,] / 'N' means uncertain
        del_index = np.greater(ious, iou_threshold)
        score_copy[del_index] = 0.
    
    if max_num is not None and len(keep_index) > max_num:
        keep_index = keep_index[: max_num]
    
    return keep_index


def cpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert isinstance(bbox1, np.ndarray)
    assert isinstance(bbox2, np.ndarray)
    assert bbox1.ndim == 2
    assert bbox2.ndim == 2
    assert bbox1.shape[-1] == bbox2.shape[-1] == 8
    
    iou_out_lst = []
    poly1 = Polygon(bbox1[0].reshape(-1, 2))
    for i in range(bbox2.shape[0]):
        poly2 = Polygon(bbox2[i].reshape(-1, 2))
        inter = poly1.intersection(poly2)
        union = poly1.union(poly2)
        iou = inter.area / union.area
        iou_out_lst.append(iou)
    iou_out = np.array(iou_out_lst)
    return iou_out


def nms_rotated(dets_rboxes, scores, score_thresh=0.05, nms_thresh=0.3):
    """
    """
    print('dets_rboxes', dets_rboxes.shape)
    print('scores', scores.shape)
    print('score_thresh', score_thresh, 'nms_thresh', nms_thresh)
    assert isinstance(dets_rboxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    
    dets_ploy = []
    for e in dets_rboxes:
        dets_ploy_s = rbox2poly_single(e)
        dets_ploy.append(dets_ploy_s)
    dets_ploy = np.array(dets_ploy)
    
    scores_max_idx = np.argmax(scores, axis=1)
    print('scores_max_idx', scores_max_idx.shape, scores_max_idx)
    scores_max = np.max(scores, axis=1)
    print('scores_max', scores_max.shape, scores_max)
    nms_keep_index = cpu_nms(dets_ploy, scores_max, score_thresh, nms_thresh)
    det_bboxes = dets_ploy[nms_keep_index, :]
    det_labels = scores_max_idx[nms_keep_index]
    det_scores = scores_max[nms_keep_index]
    return det_bboxes, det_labels, det_scores


def get_bboxes_single(cls_score_list,
                      bbox_pred_list,
                      mlvl_anchors,
                      scale_factor,
                      cfg,
                      cls_out_channels=15,
                      rescale=True,
                      use_sigmoid_cls=True):
    """
    img_shape (1024, 593, 3) scale_factor 0.36755204594400576 rescale True
    """
    print('len(cls_score_list) ', len(cls_score_list), len(bbox_pred_list), len(mlvl_anchors))
    assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    
    mlvl_bboxes = []
    mlvl_scores = []
    idx = 0
    
    for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, mlvl_anchors):
        cls_score = cls_score[0, :, :, :]
        bbox_pred = bbox_pred[0, :, :, :]
        anchors = anchors[0, :, :, :]
        print('cls_score', cls_score.shape, 'bbox_pred', bbox_pred.shape, 'anchors', anchors.shape)
        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
        
        cls_score = paddle.transpose(cls_score, [1, 2, 0])
        cls_score = paddle.reshape(cls_score, [-1, cls_out_channels])
        print('anchors anchors anchors', anchors.shape, bbox_pred.shape, cls_score.shape)
        use_sigmoid_cls = True
        if use_sigmoid_cls:
            scores = F.sigmoid(cls_score)
        else:
            scores = F.softmax(cls_score, axis=-1)
        
        bbox_pred = paddle.transpose(bbox_pred, [1, 2, 0])
        bbox_pred = paddle.reshape(bbox_pred, [-1, 5])

        #anchors = paddle.transpose(anchors, [1, 2, 0])
        anchors = paddle.reshape(anchors, [-1, 5])
        
        ### anchors = rect2rbox(anchors)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            # Get maximum scores for foreground classes.
            if use_sigmoid_cls:
                max_scores = paddle.max(scores, axis=1)
            else:
                max_scores = paddle.max(scores[:, :], axis=1)
            topk_val, topk_inds = paddle.topk(max_scores, nms_pre)
            print(topk_inds)
            anchors = paddle.gather(anchors, topk_inds)
            bbox_pred = paddle.gather(bbox_pred, topk_inds)
            scores = paddle.gather(scores, topk_inds)
        target_means = (.0, .0, .0, .0, .0),
        target_stds = (1.0, 1.0, 1.0, 1.0, 1.0)
        print('xxx delta2rbox_pd', anchors.shape, bbox_pred.shape)
        bboxes = delta2rbox_pd(anchors, bbox_pred, target_means,
                               target_stds)
        
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        
        idx += 1
    
    mlvl_bboxes = paddle.concat(mlvl_bboxes, axis=0)
    
    if rescale:
        mlvl_bboxes[:, 0:4] /= paddle.to_tensor(scale_factor[0])
    mlvl_scores = paddle.concat(mlvl_scores)
    if use_sigmoid_cls:
        # Add a dummy background class to the front when using sigmoid
        padding = paddle.zeros([mlvl_scores.shape[0], 1], dtype=mlvl_scores.dtype)
        mlvl_scores = paddle.concat([padding, mlvl_scores], axis=1)


    print('error', mlvl_bboxes.shape, mlvl_scores.shape )
    print('cfg', cfg)
    if True:
        np_mlvl_scores = np.load('/Users/liuhui29/Downloads/npy0106/get_bbox/mlvl_scores.npy')
        print('aaaaaaaa   diff of np_mlvl_scores', np_mlvl_scores.shape, mlvl_scores.numpy().shape)
        print('diff of np_mlvl_scores', np_mlvl_scores.shape, mlvl_scores.numpy().shape,
              np.sum(np.abs(np_mlvl_scores - mlvl_scores.numpy())))
        
        print('np_mlvl_scores:', np_mlvl_scores[0:5, :])
        print('mlvl_scores:', mlvl_scores[0:5, :])
        
        np_mlvl_bboxes = np.load('/Users/liuhui29/Downloads/npy0106/get_bbox/mlvl_bboxes.npy')
        print('np_mlvl_bboxes', np_mlvl_bboxes[0:5, :])
        print('mlvl_bboxes pd', mlvl_bboxes.numpy()[0:5, :])
        print('diff of mlvl_bboxes', np_mlvl_bboxes.shape, mlvl_bboxes.numpy().shape,
              np.sum(np.abs(np_mlvl_bboxes - mlvl_bboxes.numpy())))

    nms_top_k = 2000
    keep_top_k = 200

    dets_ploy = []
    for e in mlvl_bboxes.numpy():
        dets_ploy_s = rbox2poly_single(e)
        dets_ploy.append(dets_ploy_s)
    dets_ploy = np.array(dets_ploy)
    dets_ploy = paddle.to_tensor(dets_ploy)
    dets_ploy = paddle.reshape(dets_ploy, [1, dets_ploy.shape[0], dets_ploy.shape[1]])
    mlvl_scores = paddle.transpose(mlvl_scores, [1, 0])
    mlvl_scores = paddle.reshape(mlvl_scores, [1, mlvl_scores.shape[0], mlvl_scores.shape[1]])

    score_threshold = 0.05
    output, nms_rois_num, index = ops.multiclass_nms(dets_ploy, mlvl_scores, score_threshold, nms_top_k, keep_top_k,
                                    nms_threshold=0.3)
    
    #det_bboxes, det_labels, det_scores = nms_rotated(mlvl_bboxes.numpy(), mlvl_scores.numpy(),0.1, 0.5)
    print('nms', output.shape, nms_rois_num)
    return output, nms_rois_num


@register
class S2ANetHead(Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']
    
    def __init__(self,
                 stacked_convs=2,
                 feat_in=256,
                 feat_out=256,
                 num_classes=16,
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 align_conv_type='AlignConv',
                 align_conv_size = 3,
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
        assert align_conv_type in ['AlignConv', 'Conv']
        self.align_conv_type = align_conv_type
        self.align_conv_size = align_conv_size
        self.with_ORConv = with_ORConv
        self.loss = loss

        self.use_sigmoid_cls = True
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes
        
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
        self.fam_cls = Conv2D(self.feat_out, self.cls_out_channels, 1)
        
        print('self.align_conv_type', self.align_conv_type)
        if self.align_conv_type == "AlignConv":
            self.align_conv = AlignConv(self.feat_out, self.feat_out, self.align_conv_size)
        elif self.align_conv_type == "Conv":
            self.align_conv = Conv2D(self.feat_out, self.feat_out, self.align_conv_size,
                                     padding=(self.align_conv_size - 1) // 2)

        
        if self.with_ORConv:
            self.or_conv = ORConv2d(self.feat_out, int(self.feat_out / 8), kernel_size = 3, padding = 1,
                                    arf_config = (1, 8))
        else:
            self.or_conv = Conv2D(self.feat_out, self.feat_out, kernel_size=3, padding=1,
                                  weight_attr=ParamAttr(
                                      initializer=XavierUniform(fan_out=fan_conv)),
                                  bias_attr=ParamAttr(
                                      learning_rate=1.,  # TODO: check lr
                                      regularizer=L2Decay(0.)))
        # TODO: add
        self.or_pool = RotationInvariantPooling(256, 8)
        
        # ODM
        self.odm_cls_convs = Sequential()
        self.odm_reg_convs = Sequential()
        
        for i in range(self.stacked_convs):
            ch_in = int(self.feat_out / 8) if i == 0 and self.with_ORConv else self.feat_out
            #ch_in = int(self.feat_out / 8) if i == 0 else self.feat_out
            
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

        self.odm_cls = Conv2D(self.feat_out, self.cls_out_channels, 3, padding=1)
        self.odm_reg = Conv2D(self.feat_out, 5, 3, padding=1)
        
        self.refine_anchor_list = []


    def forward(self, feats):
        print('feats', len(feats))
        for e in feats:
            sha = e.shape
            print('e', e.shape)
            np.save('convert/pd_npy/feat_{}.npy'.format(sha[2]), e.numpy())
            #input('2020-0104')
        
        fam_reg_branch_list = []
        fam_cls_branch_list = []

        odm_reg_branch_list = []
        odm_cls_branch_list = []

        for i, feat in enumerate(feats):
            print('i==', i, 'in feat', feat.shape, feat.sum())
            fam_cls_feat = self.fam_cls_convs(feat)
            print('fam_cls_feat', fam_cls_feat.shape, fam_cls_feat.sum())

            fam_cls = self.fam_cls(fam_cls_feat)
            fam_cls = fam_cls.transpose([0, 2, 3, 1])
            fam_cls_reshape = fam_cls.reshape([fam_cls.shape[0], -1, self.cls_out_channels])
            fam_cls_branch_list.append(fam_cls_reshape)

            fam_reg_feat = self.fam_reg_convs(feat)
            print('fam_reg_feat', fam_reg_feat.shape, fam_reg_feat.sum())
            
            fam_reg = self.fam_reg(fam_reg_feat)
            fam_reg = fam_reg.transpose([0, 2, 3, 1])
            fam_reg_reshape = fam_reg.reshape([fam_reg.shape[0], -1, 5])
            fam_reg_branch_list.append(fam_reg_reshape)
            
            print('fam_cls', fam_cls.shape, fam_cls.sum())
            print('fam_reg', fam_reg.shape, fam_reg.sum())
            
            # prepare anchor
            featmap_size = feat.shape[-2:]
            print('featmap_size ', featmap_size)
            if (i,featmap_size[0]) in self.base_anchors.keys():
                init_anchors = self.base_anchors[(i,featmap_size[0])]
                print('init_anchors before 11111 rect2rbox', init_anchors.shape, init_anchors.sum())
            else:
                init_anchors = self.anchor_generators[i].grid_anchors(
                    featmap_size, self.anchor_strides[i])
                print('init_anchors before rect2rbox featmap_size', featmap_size, init_anchors.shape, init_anchors.sum())
                np.save('demo/init_anchors_{}'.format(featmap_size[0]), init_anchors)
                init_anchors = rect2rbox(init_anchors)
                print('init_anchors after rect2rbox', init_anchors.shape, init_anchors.sum())
                np.save('demo/init_anchors_rect2rbox_{}'.format(featmap_size[0]), init_anchors)
                self.base_anchors[(i,featmap_size[0])] = init_anchors

            for e in self.base_anchors.keys():
                anchor_tmp = self.base_anchors[e]
                print(anchor_tmp.shape, anchor_tmp.sum())

            # TODO: do not backward
            fam_reg1 = fam_reg
            fam_reg1.stop_gradient = True
            print('before bbox_decode', fam_reg1.shape, fam_reg1.sum())
            print('2021 debug anchor init_anchors', init_anchors.shape, init_anchors.sum(), init_anchors.mean())
            #init_anchors = np.load('/Users/liuhui29/Downloads/npy_2021/2021_debug_anchors_{}.npy'.format(featmap_size[0]))
            #print('load 2021 init_anchors', init_anchors.shape, init_anchors.sum())
            #init_anchors = init_anchors.reshape(-1, 5)
            print('2021 debug anchor init_anchors', init_anchors.shape, init_anchors.sum(), init_anchors.mean())
            #np.save('demo/2021_debug_anchor_{}'.format(featmap_size[0]), init_anchors)
            #np.save('demo/fam_reg_{}'.format(featmap_size[0]), fam_reg.cpu().numpy())
            refine_anchor = bbox_decode(
                            fam_reg1,
                            init_anchors,
                            self.target_means,
                            self.target_stds)

            self.refine_anchor_list.append(refine_anchor)
            print('refine_anchor:', refine_anchor.shape, refine_anchor.sum(), refine_anchor.mean())
            print('self.align_conv_type', self.align_conv_type)
            #input('xxxx')

            if self.align_conv_type == 'AlignConv':
                print('refine_anchor', refine_anchor.shape, refine_anchor.sum(), 'self.anchor_strides[i]', self.anchor_strides[i])
                align_feat = self.align_conv(feat, refine_anchor.clone(), self.anchor_strides[i])
                print('align_feat', align_feat.shape, align_feat.sum(), align_feat.mean())
            elif self.align_conv_type == 'DCN':
                align_offset = self.align_conv_offset(x)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'GA_DCN':
                align_offset = self.align_conv_offset(fam_bbox_pred)
                align_feat = self.align_conv(feat, align_offset)
            elif self.align_conv_type == 'Conv':
                align_feat = self.align_conv(feat)

            or_feat = self.or_conv(align_feat)
            odm_reg_feat = or_feat
            odm_cls_feat = or_feat
            
            print('odm_cls_feat paddle', odm_cls_feat.shape, odm_cls_feat.sum(), odm_cls_feat.mean())
            print('odm_reg_feat paddle', odm_reg_feat.shape, odm_reg_feat.sum(), odm_reg_feat.mean())
            
            #np_odm_cls_feat = np.load('/Users/liuhui29/Downloads/npy_odm/odm_cls_feat_{}.npy'.format(i))
            #odm_cls_feat = paddle.to_tensor(np_odm_cls_feat)

            #np_odm_reg_feat = np.load('/Users/liuhui29/Downloads/npy_odm/odm_reg_feat_{}.npy'.format(i))
            #odm_reg_feat = paddle.to_tensor(np_odm_reg_feat)

            print('odm_cls_feat', odm_cls_feat.shape, odm_cls_feat.sum())
            print('odm_reg_feat', odm_reg_feat.shape, odm_reg_feat.sum())

            odm_reg_feat = self.odm_reg_convs(odm_reg_feat)
            odm_cls_feat = self.odm_cls_convs(odm_cls_feat)
            odm_cls_score = self.odm_cls(odm_cls_feat)
            odm_bbox_pred = self.odm_reg(odm_reg_feat)
            print('odm_cls_score', odm_cls_score.shape, odm_cls_score.sum())
            print('odm_bbox_pred', odm_bbox_pred.shape, odm_bbox_pred.sum())
                
            odm_reg_branch_list.append(odm_bbox_pred)
            odm_cls_branch_list.append(odm_cls_score)
        
        print('feats out')
        #fam_reg_branch = paddle.concat(fam_reg_branch_list, axis=1)
        #fam_cls_branch = paddle.concat(fam_cls_branch_list, axis=1)
        
        #print('fam_reg_branch:', fam_reg_branch.shape, fam_reg_branch.sum())
        #print('fam_cls_branch:', fam_cls_branch.shape, fam_cls_branch.sum())
        
        print('*'*64)
        for e in odm_reg_branch_list:
            print('odm_reg_branch_list  eee', e.shape, e.sum(), e.mean())
        for e in odm_cls_branch_list:
            print('odm_cls_branch_list  eee', e.shape, e.sum(), e.mean())
        #odm_reg_branch = paddle.concat(odm_reg_branch_list, axis=1)
        #odm_cls_branch = paddle.concat(odm_cls_branch_list, axis=1)


        self.fam_cls_branch_list = fam_cls_branch_list
        self.fam_reg_branch_list = fam_reg_branch_list
        self.odm_cls_branch_list = odm_cls_branch_list
        self.odm_reg_branch_list = odm_reg_branch_list
        print('*' * 64)
        for e in odm_cls_branch_list:
            print('odm_cls_branch_list', e.shape, e.sum(), e.mean())
        for e in odm_reg_branch_list:
            print('odm_reg_branch_list', e.shape, e.sum(), e.mean())
        print('*' * 64)
        return (fam_cls_branch_list, fam_reg_branch_list, odm_cls_branch_list, odm_reg_branch_list)

    def get_prediction(self, inputs):
    
        featmap_sizes = [featmap.shape[-2:] for featmap in self.odm_cls_branch_list]
        num_levels = len(self.odm_cls_branch_list)

        refine_anchors = self.refine_anchor_list
        im_shape = inputs['im_shape'].numpy()
        print('im_shape', im_shape)
        scale_factor = inputs['scale_factor'].numpy()
        print('scale_factor', scale_factor)
        cfg = {'nms_pre': 2000, 'min_bbox_size': 0, 'score_thr': 0.1, 'max_per_img': 2000}
        pred_out, pred_bbox_num = get_bboxes_single(self.odm_cls_branch_list, self.odm_reg_branch_list, refine_anchors,
                                      scale_factor[0], cfg, cls_out_channels=self.cls_out_channels, rescale=True)
        
        return pred_out, pred_bbox_num
    
    def get_loss(self, inputs, head_outputs):
        return {'loss': 0.1}
        return self.loss(inputs, head_outputs)


if __name__ == '__main__':
    anchor_generator = S2ANetAnchor()
    anchor_list = anchor_generator()
    print(anchor_list)
    print(anchor_list.shape)
