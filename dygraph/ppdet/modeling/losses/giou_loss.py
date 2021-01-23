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

import numpy as np

from paddle import fluid
from ppdet.core.workspace import register, serializable

__all__ = ['GiouLoss']


@register
@serializable
class GiouLoss(object):
    '''
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): diou loss weight, default as 10 in faster-rcnn
        is_cls_agnostic (bool): flag of class-agnostic
        num_classes (int): class num
        do_average (bool): whether to average the loss
        use_class_weight(bool): whether to use class weight
    '''
    __shared__ = ['num_classes']

    def __init__(self,
                 loss_weight=10.,
                 is_cls_agnostic=False,
                 num_classes=81,
                 do_average=True,
                 use_class_weight=True):
        super(GiouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.is_cls_agnostic = is_cls_agnostic
        self.num_classes = num_classes
        self.do_average = do_average
        self.class_weight = 2 if is_cls_agnostic else num_classes
        self.use_class_weight = use_class_weight

    # deltas: NxMx4
    def bbox_transform(self, deltas, weights):
        wx, wy, ww, wh = weights

        deltas = paddle.reshape(deltas, shape=(0, -1, 4))

        dx = paddle.slice(deltas, axes=[2], starts=[0], ends=[1]) * wx
        dy = paddle.slice(deltas, axes=[2], starts=[1], ends=[2]) * wy
        dw = paddle.slice(deltas, axes=[2], starts=[2], ends=[3]) * ww
        dh = paddle.slice(deltas, axes=[2], starts=[3], ends=[4]) * wh

        dw = paddle.clip(dw, -1.e10, np.log(1000. / 16))
        dh = paddle.clip(dh, -1.e10, np.log(1000. / 16))

        pred_ctr_x = dx
        pred_ctr_y = dy
        pred_w = paddle.exp(dw)
        pred_h = paddle.exp(dh)

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        x1 = paddle.reshape(x1, shape=(-1, ))
        y1 = paddle.reshape(y1, shape=(-1, ))
        x2 = paddle.reshape(x2, shape=(-1, ))
        y2 = paddle.reshape(y2, shape=(-1, ))

        return x1, y1, x2, y2

    def __call__(self,
                 x,
                 y,
                 inside_weight=None,
                 outside_weight=None,
                 bbox_reg_weight=[0.1, 0.1, 0.2, 0.2],
                 use_transform=True):
        eps = 1.e-10
        if use_transform:
            x1, y1, x2, y2 = self.bbox_transform(x, bbox_reg_weight)
            x1g, y1g, x2g, y2g = self.bbox_transform(y, bbox_reg_weight)
        else:
            x1, y1, x2, y2 = paddle.split(x, num_or_sections=4, axis=1)
            x1g, y1g, x2g, y2g = paddle.split(y, num_or_sections=4, axis=1)

        x2 = paddle.maximum(x1, x2)
        y2 = paddle.maximum(y1, y2)

        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)

        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * paddle.greater_than(
            xkis2, xkis1) * paddle.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + eps

        iouk = intsctk / unionk

        area_c = (xc2 - xc1) * (yc2 - yc1) + eps
        miouk = iouk - ((area_c - unionk) / area_c)

        iou_weights = 1
        if inside_weight is not None and outside_weight is not None:
            inside_weight = paddle.reshape(inside_weight, shape=(-1, 4))
            outside_weight = paddle.reshape(outside_weight, shape=(-1, 4))

            inside_weight = paddle.mean(inside_weight, axis=1)
            outside_weight = paddle.mean(outside_weight, axis=1)

            iou_weights = inside_weight * outside_weight
        elif outside_weight is not None:
            iou_weights = outside_weight

        if self.do_average:
            miouk = paddle.mean((1 - miouk) * iou_weights)
        else:
            iou_distance = paddle.multiply(1 - miouk, iou_weights, axis=0)
            miouk = paddle.sum(iou_distance)

        if self.use_class_weight:
            miouk = miouk * self.class_weight

        return miouk * self.loss_weight
