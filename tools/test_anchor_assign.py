from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
import numpy as np


def bbox2delta(bboxes1, bboxes2, weights):
    ex_w = bboxes1[:, 2] - bboxes1[:, 0] + 1
    ex_h = bboxes1[:, 3] - bboxes1[:, 1] + 1
    ex_ctr_x = bboxes1[:, 0] + 0.5 * ex_w
    ex_ctr_y = bboxes1[:, 1] + 0.5 * ex_h

    gt_w = bboxes2[:, 2] - bboxes2[:, 0] + 1
    gt_h = bboxes2[:, 3] - bboxes2[:, 1] + 1
    gt_ctr_x = bboxes2[:, 0] + 0.5 * gt_w
    gt_ctr_y = bboxes2[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    deltas = np.vstack([dx, dy, dw, dh]).transpose()
    return deltas

def generate_rpn_anchor_target(anchors,
                               gt_boxes,
                               is_crowd,
                               im_info,
                               rpn_straddle_thresh,
                               rpn_batch_size_per_im,
                               rpn_positive_overlap,
                               rpn_negative_overlap,
                               rpn_fg_fraction,
                               use_random=True,
                               anchor_reg_weights=[1., 1., 1., 1.]):
    anchor_num = anchors.shape[0]
    batch_size = gt_boxes.shape[0]
    
    loc_indexes = []
    cls_indexes = []
    tgt_labels = []
    tgt_deltas = []
    anchor_inside_weights = []
    
    for i in range(batch_size):
        
        # TODO: move anchor filter into anchor generator
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        
        if rpn_straddle_thresh >= 0:
            anchor_inds = np.where((anchors[:, 0] >= -rpn_straddle_thresh) & (
                    anchors[:, 1] >= -rpn_straddle_thresh) & (
                                           anchors[:, 2] < im_width + rpn_straddle_thresh) & (
                                           anchors[:, 3] < im_height + rpn_straddle_thresh))[0]
            anchor = anchors[anchor_inds, :]
        else:
            anchor_inds = np.arange(anchors.shape[0])
            anchor = anchors
        
        gt_bbox = gt_boxes[i] * im_scale
        is_crowd_slice = is_crowd[i]
        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_bbox = gt_bbox[not_crowd_inds]
        
        # Step1: match anchor and gt_bbox
        anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels = label_anchor(anchor,
                                                                       gt_bbox)
        # Step2: sample anchor
        fg_inds, bg_inds, fg_fake_inds, fake_num = sample_anchor(
            anchor_gt_bbox_iou, labels, rpn_positive_overlap,
            rpn_negative_overlap, rpn_batch_size_per_im, rpn_fg_fraction,
            use_random)
        
        # Step3: make output
        loc_inds = np.hstack([fg_fake_inds, fg_inds])
        cls_inds = np.hstack([fg_inds, bg_inds])
        
        sampled_labels = labels[cls_inds]
        
        sampled_anchors = anchor[loc_inds]
        sampled_gt_boxes = gt_bbox[anchor_gt_bbox_inds[loc_inds]]
        sampled_deltas = bbox2delta(sampled_anchors, sampled_gt_boxes,
                                    anchor_reg_weights)
        
        anchor_inside_weight = np.zeros((len(loc_inds), 4), dtype=np.float32)
        anchor_inside_weight[fake_num:, :] = 1
        
        loc_indexes.append(anchor_inds[loc_inds] + i * anchor_num)
        cls_indexes.append(anchor_inds[cls_inds] + i * anchor_num)
        tgt_labels.append(sampled_labels)
        tgt_deltas.append(sampled_deltas)
        anchor_inside_weights.append(anchor_inside_weight)
    
    loc_indexes = np.concatenate(loc_indexes)
    cls_indexes = np.concatenate(cls_indexes)
    tgt_labels = np.concatenate(tgt_labels).astype('float32')
    tgt_deltas = np.vstack(tgt_deltas).astype('float32')
    anchor_inside_weights = np.vstack(anchor_inside_weights)
    
    return loc_indexes, cls_indexes, tgt_labels, tgt_deltas, anchor_inside_weights


def bbox_overlaps(bboxes1, bboxes2):
    w1 = np.maximum(bboxes1[:, 2] - bboxes1[:, 0] + 1, 0)
    h1 = np.maximum(bboxes1[:, 3] - bboxes1[:, 1] + 1, 0)
    w2 = np.maximum(bboxes2[:, 2] - bboxes2[:, 0] + 1, 0)
    h2 = np.maximum(bboxes2[:, 3] - bboxes2[:, 1] + 1, 0)
    area1 = w1 * h1
    area2 = w2 * h2

    boxes1_x1, boxes1_y1, boxes1_x2, boxes1_y2 = np.split(bboxes1, 4, axis=1)
    boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2 = np.split(bboxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(boxes1_y2, np.transpose(boxes2_y2))
    all_pairs_max_ymin = np.maximum(boxes1_y1, np.transpose(boxes2_y1))
    inter_h = np.maximum(all_pairs_min_ymax - all_pairs_max_ymin + 1, 0.)
    all_pairs_min_xmax = np.minimum(boxes1_x2, np.transpose(boxes2_x2))
    all_pairs_max_xmin = np.maximum(boxes1_x1, np.transpose(boxes2_x1))
    inter_w = np.maximum(all_pairs_min_xmax - all_pairs_max_xmin + 1, 0.)

    inter_area = inter_w * inter_h

    union_area = np.expand_dims(area1, 1) + np.expand_dims(area2, 0)
    overlaps = inter_area / (union_area - inter_area)
    return overlaps


def label_anchor(anchors, gt_boxes):
    iou = bbox_overlaps(anchors, gt_boxes)
    # every gt's anchor's index
    gt_bbox_anchor_inds = iou.argmax(axis=0)
    gt_bbox_anchor_iou = iou[gt_bbox_anchor_inds, np.arange(iou.shape[1])]
    gt_bbox_anchor_iou_inds = np.where(iou == gt_bbox_anchor_iou)[0]
    
    # every anchor's gt bbox's index
    anchor_gt_bbox_inds = iou.argmax(axis=1)
    anchor_gt_bbox_iou = iou[np.arange(iou.shape[0]), anchor_gt_bbox_inds]
    
    labels = np.ones((iou.shape[0],), dtype=np.int32) * -1
    labels[gt_bbox_anchor_iou_inds] = 1
    
    return anchor_gt_bbox_inds, anchor_gt_bbox_iou, labels



def sample_anchor(anchor_gt_bbox_iou,
                  labels,
                  rpn_positive_overlap,
                  rpn_negative_overlap,
                  rpn_batch_size_per_im,
                  rpn_fg_fraction,
                  use_random=True):
    labels[anchor_gt_bbox_iou >= rpn_positive_overlap] = 1
    num_fg = int(rpn_fg_fraction * rpn_batch_size_per_im)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg and use_random:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        disable_inds = fg_inds[num_fg:]
    labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]
    
    num_bg = rpn_batch_size_per_im - np.sum(labels == 1)
    bg_inds = np.where(anchor_gt_bbox_iou < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg and use_random:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
    else:
        enable_inds = bg_inds[:num_bg]
    
    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0
    
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    
    return fg_inds, bg_inds, fg_fake_inds, fake_num


def filter_roi(rois, max_overlap):
    ws = rois[:, 2] - rois[:, 0] + 1
    hs = rois[:, 3] - rois[:, 1] + 1
    keep = np.where((ws > 0) & (hs > 0) & (max_overlap < 1))[0]
    if len(keep) > 0:
        return rois[keep, :]
    return np.zeros((1, 4)).astype('float32')


if __name__ == '__main__':
    st_npy_dir = '/Users/liuhui29/Desktop/work/code/work_push/ppdet_static/PaddleDetection/npy/'
    anchors = np.load(st_npy_dir + 'collect.tmp_0.npy') # 2000x4
    #anchors = np.load('input_anchors.npy')  # 2000x4
    gt_boxes = np.load(st_npy_dir + 'gt_bbox.npy') # 4x4
    is_crowd = np.load(st_npy_dir + 'is_crowd.npy') # 4xx1
    im_info = np.load(st_npy_dir + 'im_info.npy') # 1x3
    is_crowd = is_crowd.reshape(1,4)
    gt_boxes = np.expand_dims(gt_boxes, 0)
    
    rpn_straddle_thresh = 0.0
    rpn_batch_size_per_im = 256
    rpn_positive_overlap = 0.7
    rpn_negative_overlap = 0.3
    rpn_fg_fraction = 0.5
    use_random = True
    anchor_reg_weights = [1., 1., 1., 1.]
    ret = generate_rpn_anchor_target(anchors, gt_boxes, is_crowd, im_info, rpn_straddle_thresh, rpn_batch_size_per_im,
                               rpn_positive_overlap, rpn_fg_fraction, use_random, anchor_reg_weights)

    loc_indexes, cls_indexes, tgt_labels, tgt_deltas, anchor_inside_weights = ret
