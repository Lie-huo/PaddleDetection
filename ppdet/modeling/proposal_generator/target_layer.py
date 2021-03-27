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

from ppdet.core.workspace import register, serializable

from .target import rpn_anchor_target, generate_proposal_target, generate_mask_target
from ppdet.modeling.utils import bbox_util
from ppdet.utils import bbox_utils
import numpy as np

g_idx=0

@register
@serializable
class RPNTargetAssign(object):
    def __init__(self,
                 batch_size_per_im=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 use_random=True):
        super(RPNTargetAssign, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.use_random = use_random

    def __call__(self, inputs, anchors):
        """
        inputs: ground-truth instances.
        anchor_box (Tensor): [num_anchors, 4], num_anchors are all anchors in all feature maps.
        """
        gt_boxes = inputs['gt_bbox']
        batch_size = gt_boxes.shape[0]
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors, gt_boxes, self.batch_size_per_im, self.positive_overlap,
            self.negative_overlap, self.fg_fraction, self.use_random,
            batch_size)
        norm = self.batch_size_per_im * batch_size

        return tgt_labels, tgt_bboxes, tgt_deltas, norm


@register
class BBoxAssigner(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh=.5,
                 use_random=True,
                 is_cls_agnostic=False,
                 cascade_iou=[0.5, 0.6, 0.7],
                 num_classes=80):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.use_random = use_random
        self.is_cls_agnostic = is_cls_agnostic
        self.cascade_iou = cascade_iou
        self.num_classes = num_classes

    def __call__(self,
                 rpn_rois,
                 rpn_rois_num,
                 inputs,
                 stage=0,
                 is_cascade=False):
        gt_classes = inputs['gt_class']
        gt_boxes = inputs['gt_bbox']
        # rois, tgt_labels, tgt_bboxes, tgt_gt_inds
        # new_rois_num
        outs = generate_proposal_target(
            rpn_rois, gt_classes, gt_boxes, self.batch_size_per_im,
            self.fg_fraction, self.fg_thresh, self.bg_thresh, self.num_classes,
            self.use_random, is_cascade, self.cascade_iou[stage])
        rois = outs[0]
        rois_num = outs[-1]
        # tgt_labels, tgt_bboxes, tgt_gt_inds
        targets = outs[1:4]
        return rois, rois_num, targets


@register
@serializable
class MaskAssigner(object):
    __shared__ = ['num_classes', 'mask_resolution']

    def __init__(self, num_classes=80, mask_resolution=14):
        super(MaskAssigner, self).__init__()
        self.num_classes = num_classes
        self.mask_resolution = mask_resolution

    def __call__(self, rois, tgt_labels, tgt_gt_inds, inputs):
        gt_segms = inputs['gt_poly']

        outs = generate_mask_target(gt_segms, rois, tgt_labels, tgt_gt_inds,
                                    self.num_classes, self.mask_resolution)

        # mask_rois, mask_rois_num, tgt_classes, tgt_masks, mask_index, tgt_weights
        return outs



class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        if not isinstance(v, Vector):
            return NotImplemented
        return self.x*v.y - self.y*v.x


class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a*p.x + self.b*p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a*other.b - self.b*other.a
        return Vector(
            (self.b*other.c - self.c*other.b)/w,
            (self.c*other.a - self.a*other.c)/w
        )


def rectangle_vertices(cx, cy, w, h, r):
    angle = np.pi*r/180
    dx = w/2
    dy = h/2
    dxcos = dx*np.cos(angle)
    dxsin = dx*np.sin(angle)
    dycos = dy*np.cos(angle)
    dysin = dy*np.sin(angle)
    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - -dysin,  dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - dysin,  dxsin + dycos),
        Vector(cx, cy) + Vector(-dxcos - dysin, -dxsin + dycos)
    )

def intersection_area(r1, r2):
    # r1 and r2 are in (center, width, height, rotation) representation
    # First convert these into a sequence of vertices

    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1],
            line_values, line_values[1:] + line_values[:1]):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p.x*q.y - p.y*q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))

@register
class S2ANetAnchorAssigner(object):
    def __init__(self, pos_iou_thr=0.5,
                 neg_iou_thr=0.4,
                 min_iou_thr=0.0,
                 ignore_iof_thr=-2):
        super(S2ANetAnchorAssigner, self).__init__()
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_iou_thr = min_iou_thr
        self.ignore_iof_thr = ignore_iof_thr

    def anchor_valid(self, anchors):
        """

        Args:
            anchor: M x 4

        Returns:

        """
        if anchors.ndim == 3:
            anchors = anchors.reshape(-1, anchor.shape[-1])
        assert anchors.ndim == 2
        anchor_num = anchors.shape[0]
        anchor_valid = np.ones((anchor_num), np.uint8)
        anchor_inds = np.arange(anchor_num)
        return anchor_inds

    def assign_anchor(self,
                      anchors,
                      gt_bboxes,
                      gt_lables,
                      pos_iou_thr,
                      neg_iou_thr,
                      min_iou_thr=0.0,
                      ignore_iof_thr=-2):
        """

        Args:
            anchors:
            gt_bboxes:[M, 5] rc,yc,w,h,angle
            gt_lables:

        Returns:

        """
        assert anchors.shape[1] == 4 or anchors.shape[1] == 5
        assert gt_bboxes.shape[1] == 4 or gt_bboxes.shape[1] == 5
        anchors_xc_yc = anchors
        gt_bboxes_xc_yc = gt_bboxes

        # calc rbox iou
        #anchors_xc_yc = anchors_xc_yc.astype(np.float32)
        #anchors_xc_yc = paddle.to_tensor(anchors_xc_yc)
        #gt_bboxes_xc_yc = paddle.to_tensor(gt_bboxes_xc_yc)

        # call custom_ops
        #iou = custom_ops.rbox_iou(anchors_xc_yc, gt_bboxes_xc_yc)
        #iou = iou.numpy()
        #global g_idx
        #iou = np.load('npy/overlaps_0322_{}.npy'.format(g_idx))
        #iou = iou.T
        #g_idx+=1

        def calc_iou(bboxes1, bboxes2):
            x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
            x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
            xA = np.maximum(x11, np.transpose(x21))
            yA = np.maximum(y11, np.transpose(y21))
            xB = np.minimum(x12, np.transpose(x22))
            yB = np.minimum(y12, np.transpose(y22))
            interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
            boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
            boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
            iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
            return iou

        anchors_xc_yc = anchors_xc_yc.astype(np.float32)
        gt_bboxes_xc_yc = gt_bboxes_xc_yc.astype(np.float32)
        iou = calc_iou(anchors_xc_yc[:, 0:4], gt_bboxes_xc_yc[:, 0:4])

        # every gt's anchor's index
        gt_bbox_anchor_inds = iou.argmax(axis=0)
        gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
        gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]

        # every anchor's gt bbox's index
        anchor_gt_bbox_inds = iou.argmax(axis=1)
        anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]

        # (1) set labels=-2 as default
        labels = np.ones((iou.shape[0],), dtype=np.int32) * ignore_iof_thr

        # (2) assign ignore
        labels[anchor_gt_bbox_iou < min_iou_thr] = ignore_iof_thr

        # (3) assign neg_ids -1
        assign_neg_ids1 = anchor_gt_bbox_iou >= min_iou_thr
        assign_neg_ids2 = anchor_gt_bbox_iou < neg_iou_thr
        assign_neg_ids = np.logical_and(assign_neg_ids1, assign_neg_ids2)
        labels[assign_neg_ids] = -1

        # anchor_gt_bbox_iou_inds
        # (4) assign max_iou as pos_ids >=0
        anchor_gt_bbox_iou_inds = anchor_gt_bbox_inds[gt_bbox_anchor_iou_inds]
        # gt_bbox_anchor_iou_inds = np.logical_and(gt_bbox_anchor_iou_inds, anchor_gt_bbox_iou >= min_iou_thr)
        labels[gt_bbox_anchor_iou_inds] = gt_lables[anchor_gt_bbox_iou_inds]

        # (5) assign >= pos_iou_thr as pos_ids
        iou_pos_iou_thr_ids = anchor_gt_bbox_iou >= pos_iou_thr
        iou_pos_iou_thr_ids_box_inds = anchor_gt_bbox_inds[iou_pos_iou_thr_ids]
        labels[iou_pos_iou_thr_ids] = gt_lables[iou_pos_iou_thr_ids_box_inds]

        return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels

    def __call__(self, anchors,
                       gt_bboxes,
                       gt_labels,
                       is_crowd,
                       im_scale):

        assert anchors.ndim == 2
        assert anchors.shape[1] == 5
        assert gt_bboxes.ndim == 2
        #print('gt_bboxes', gt_bboxes.shape, gt_bboxes)
        assert gt_bboxes.shape[1] == 5

        pos_iou_thr = self.pos_iou_thr
        neg_iou_thr = self.neg_iou_thr
        min_iou_thr = self.min_iou_thr
        ignore_iof_thr = self.ignore_iof_thr

        anchor_num = anchors.shape[0]

        # TODO: support not square image
        im_scale = im_scale[0][0]
        anchors_inds = self.anchor_valid(anchors)
        anchors = anchors[anchors_inds]
        gt_bboxes = gt_bboxes * im_scale
        is_crowd_slice = is_crowd
        not_crowd_inds = np.where(is_crowd_slice == 0)

        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = self.assign_anchor(anchors, gt_bboxes, gt_labels.reshape(-1),
                                                                        pos_iou_thr, neg_iou_thr, min_iou_thr,
                                                                        ignore_iof_thr)

        # Step2: sample anchor
        pos_inds = np.where(labels >= 0)[0]
        neg_inds = np.where(labels == -1)[0]

        # Step3: make output
        anchors_num = anchors.shape[0]
        bbox_targets = np.zeros_like(anchors)
        bbox_weights = np.zeros_like(anchors)
        pos_labels = np.ones(anchors_num, dtype=np.int32) * -1
        pos_labels_weights = np.zeros(anchors_num, dtype=np.float32)

        #print('anchors', anchors.shape)
        #print('pos_inds', pos_inds)
        pos_sampled_anchors = anchors[pos_inds]
        #print('ancho target pos_inds', pos_inds, len(pos_inds))
        pos_sampled_gt_boxes = gt_bboxes[anchor_gt_bbox_inds[pos_inds]]
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_util.rbox2delta(pos_sampled_anchors, pos_sampled_gt_boxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            pos_labels[pos_inds] = labels[pos_inds]
            pos_labels_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            pos_labels_weights[neg_inds] = 1.0
        return (pos_labels, pos_labels_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
