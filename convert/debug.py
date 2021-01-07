

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


import paddle
import paddle.nn.functional as F


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
    #print('deltas', deltas.shape)
    #H, W, C = deltas.shape
    #print('deltas', deltas.shape)
    #deltas = paddle.reshape(deltas, [-1, C])
    #print('deltas', type(deltas), deltas.shape)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dw = denorm_deltas[:, 2]
    dh = denorm_deltas[:, 3]
    dangle = denorm_deltas[:, 4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    #print('max_ratio', max_ratio)
    dw = paddle.clip(dw, min=-max_ratio, max=max_ratio)
    dh = paddle.clip(dh, min=-max_ratio, max=max_ratio)

    Rroi_x = Rrois[:, 0]
    Rroi_y = Rrois[:, 1]
    Rroi_w = Rrois[:, 2]
    Rroi_h = Rrois[:, 3]
    Rroi_angle = Rrois[:, 4]

    print('now now now', dx.shape, Rroi_angle.shape, Rroi_x.shape)
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
    return bboxes


def multiclass_nms_rbox(multi_bboxes,
                        multi_scores,
                        score_thr,
                        nms_cfg,
                        max_num=-1,
                        score_factors=None):
    """
    NMS for multi-class bboxes.
    :param multi_bboxes:
    :param multi_scores:
    :param score_thr:
    :param nms_cfg:
    :param max_num:
    :param score_factors:
    :return:
    """
    print('multi_scores', multi_scores.shape)
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []

    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms_rotated')

    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr

        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 5:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 5: (i + 1) * 5]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = np.concatenate([_bboxes, _scores[:, None]], axis=1)
        print('cls_dets', cls_dets.shape)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)

        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
        
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]

    else:
        bboxes = multi_bboxes.new_zeros((0, 6))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


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


def rbox2poly_single_m(rrect):
    """
    rrect:[[x_ctr,y_ctr,w,h,angle]]
    to
    poly:[[x0,y0,x1,y1,x2,y2,x3,y3]]
    """
    x_ctr = rrect[:, 0]
    y_ctr = rrect[:, 1]
    width = rrect[:, 2]
    height = rrect[:, 3]
    angle = rrect[:, 4]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    print('R', R.shape, 'rect', rect.shape)
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly


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


import cv2
def poly_IoU(a, b):
    # step1:
    inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
    inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
    inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
    inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
    x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
    y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
    y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
    if x1 >= x2 or y1 >= y2 or (x2 - x1) < 2 or (y2 - y1) < 2:
        return 0.
    else:
        mask_w = np.int(np.ceil(x2 - x1))
        mask_h = np.int(np.ceil(y2 - y1))
        mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        a[:, 0] -= x1
        a[:, 1] -= y1
        b[:, 0] -= x1
        b[:, 1] -= y1
        mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
        mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = float(inter) / (float(union) + 1e-12)
        print('inter', inter, 'union', union)
        return iou


import shapely
from shapely.geometry import Polygon


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

def nms_rotated_0(dets_rboxes, scores, score_thresh=0.01, nms_thresh=0.3, sigma=0.1, post_nms_top_n=1):
    if len(dets_rboxes) == 0:
        return [], []
    
    # Bounding boxes
    dets_rboxes = np.array(dets_rboxes)
    print('dets_rboxes', dets_rboxes.shape)
    dets_ploy = []
    for e in dets_rboxes:
        dets_ploy_s = rbox2poly_single(e)
        dets_ploy.append(dets_ploy_s)
    dets_ploy = np.array(dets_ploy)
    print('dets_ploy sss', dets_ploy.shape)
    input('xxx')
    
    # Confidence scores of bounding boxes
    score = np.array(scores)
    
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    
    # Iterate bounding boxes
    while order.shape[0] > 0:
        # The index of largest confidence score
        index = order[-1]
        
        # Pick the bounding box with largest confidence score
        picked_boxes.append(dets_ploy[index])
        picked_score.append(score[index])
        
        #iou = poly_IoU(dets_ploy[index], dets_ploy[order[:-1]])
        poly1 = Polygon(dets_ploy[index].reshape(-1,2))
        iou_lst = []
        print('order', order.shape, order[:-1])
        for ii in range(order[:-1]):
            poly_tmp = Polygon(dets_ploy[ii].reshape(-1,2))
            inter = poly1.intersection(poly_tmp)
            union = poly1.union(poly_tmp)
            iou = inter.area / union.area
            print('iou', iou)
            input('xxx')
        
        left = np.where(iou < nms_thresh)
        order = order[left]
    
    return picked_boxes, picked_score


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
    #nms_keep_index = cpu_nms(dets_ploy, scores_max, score_thresh, nms_thresh)
    #np.save('convert/nms_keep_index.npy', nms_keep_index)
    nms_keep_index = np.load('convert/nms_keep_index.npy')
    det_bboxes = dets_ploy[nms_keep_index, :]
    det_labels = scores_max_idx[nms_keep_index]
    return det_bboxes, det_labels


def disp_res(det_bbox, det_cls):
    img = cv2.imread('demo/P2594.png')
    
    lst = [i for i in range(len(det_bbox))]
    for i in lst:
        pt_lst = [int(e) for e in det_bbox[i]]
        x1,y1,x2,y2,x3,y3,x4,y4 = pt_lst[0], pt_lst[1], pt_lst[2], pt_lst[3], pt_lst[4], pt_lst[5], pt_lst[6], pt_lst[7]
        cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 3)
        print('x1 y1 ',x1,y1,x2,y2)
        cv2.line(img, (x2,y2), (x3,y3), (0, 255, 0), 3)
        cv2.line(img, (x3,y3), (x4,y4), (0, 255, 0), 3)
        cv2.line(img, (x4,y4), (x1,y1), (0, 255, 0), 3)
        cls_name = 'cls_{}'.format(det_cls[i])
        cv2.putText(img, cls_name, (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
    cv2.imwrite('convert/det_last_res.jpg', img)


def get_bboxes_single(cls_score_list,
                    bbox_pred_list,
                    mlvl_anchors,
                    img_shape,
                    scale_factor,
                    cfg,
                    cls_out_channels=15,
                    rescale=True,
                    use_sigmoid_cls=True):
    """
        img_shape (1024, 593, 3) scale_factor 0.36755204594400576 rescale True
        """
    assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    
    mlvl_bboxes = []
    mlvl_scores = []
    idx = 0

    for cls_score, bbox_pred, anchors in zip(cls_score_list, bbox_pred_list, mlvl_anchors):
        print('cls_score', cls_score.shape, 'bbox_pred', bbox_pred.shape)
        assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
        
        cls_score = paddle.transpose(cls_score, [1, 2, 0])
        cls_score = paddle.reshape(cls_score, [-1, cls_out_channels])

        use_sigmoid_cls = True
        if use_sigmoid_cls:
            scores = F.sigmoid(cls_score)
        else:
            scores = F.softmax(cls_score, axis=-1)
        
        bbox_pred = paddle.transpose(bbox_pred, [1, 2, 0])
        bbox_pred = paddle.reshape(bbox_pred, [-1, 5])

        
        ### anchors = rect2rbox(anchors)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            # Get maximum scores for foreground classes.
            if use_sigmoid_cls:
                max_scores = paddle.max(scores, axis=1)
            else:
                max_scores = paddle.max(scores[:, :], axis=1)
            topk_val, topk_inds =paddle.topk(max_scores, nms_pre)
            print(topk_inds)
            anchors = paddle.gather(anchors, topk_inds)
            bbox_pred = paddle.gather(bbox_pred, topk_inds)
            scores = paddle.gather(scores, topk_inds)
        target_means=(.0, .0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)
        
        print('xxx delta2rbox_pd', anchors.shape, bbox_pred.shape)
        bboxes = delta2rbox_pd(anchors, bbox_pred, target_means,
                            target_stds)

        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
        
        idx += 1
        
    mlvl_bboxes = paddle.concat(mlvl_bboxes, axis=0)
    print('mlvl_bboxes', mlvl_bboxes.shape)

    rescale = True
    if rescale:
        mlvl_bboxes[:, 0:4] /= paddle.to_tensor(scale_factor)
    mlvl_scores = paddle.concat(mlvl_scores)
    if use_sigmoid_cls:
        # Add a dummy background class to the front when using sigmoid
        padding = paddle.zeros([mlvl_scores.shape[0], 1], dtype=mlvl_scores.dtype)
        mlvl_scores = paddle.concat([padding, mlvl_scores], axis=1)

    #mlvl_bboxes = paddle.reshape(mlvl_bboxes, [-1, mlvl_bboxes.shape[0], mlvl_bboxes.shape[1]])
    #mlvl_scores = paddle.reshape(mlvl_scores, [-1, mlvl_scores.shape[0], mlvl_scores.shape[1]])

    print('cfg', cfg)
    np.save('convert/mlvl_bboxes.npy', mlvl_bboxes.numpy())
    np.save('convert/mlvl_scores.npy', mlvl_scores.numpy())

    np_mlvl_scores = np.load('/Users/liuhui29/Downloads/npy0106/get_bbox/mlvl_scores.npy')
    print('aaaaaaaa   diff of np_mlvl_scores', np_mlvl_scores.shape, mlvl_scores.numpy().shape)
    print('diff of np_mlvl_scores', np_mlvl_scores.shape, mlvl_scores.numpy().shape,
          np.sum(np.abs(np_mlvl_scores - mlvl_scores.numpy())))
    
    print('np_mlvl_scores:', np_mlvl_scores[0:5, :])
    print('mlvl_scores:', mlvl_scores[0:5, :])
    
    np_mlvl_bboxes= np.load('/Users/liuhui29/Downloads/npy0106/get_bbox/mlvl_bboxes.npy')
    print('np_mlvl_bboxes', np_mlvl_bboxes[0:5, :])
    print('mlvl_bboxes pd', mlvl_bboxes.numpy()[0:5, :])
    print('diff of mlvl_bboxes', np_mlvl_bboxes.shape, mlvl_bboxes.numpy().shape,
          np.sum(np.abs(np_mlvl_bboxes - mlvl_bboxes.numpy())))

    print('start nms', mlvl_bboxes.shape, mlvl_scores.shape)
    score_threshold = 0.05
    nms_top_k = 2000
    keep_top_k = 2000
    nms_threshold=0.3
    det_bboxes, det_labels = nms_rotated(mlvl_bboxes.numpy(), mlvl_scores.numpy(), score_threshold, nms_threshold)

    return det_bboxes, det_labels
    

if __name__ == "__main__":
    npy_path = '/Users/liuhui29/Downloads/npy/get_bbox/cls_score_list_0.npy'
    npy_path_lst = []
    npy_path_lst.append('/Users/liuhui29/Downloads/npy/get_bbox/cls_score_list_0.npy')
    npy_path_lst.append('/Users/liuhui29/Downloads/npy/get_bbox/bbox_pred_list_0.npy')
    npy_path_lst.append('/Users/liuhui29/Downloads/npy/get_bbox/mlvl_anchors_0.npy')

    cls_score_list = []
    bbox_pred_list = []
    mlvl_anchors_list = []
    
    for i in range(5):
        p1 = '/Users/liuhui29/Downloads/npy0106/get_bbox/cls_score_list_{}.npy'.format(i)
        cls_score = np.load(p1)
        p2 = '/Users/liuhui29/Downloads/npy0106/get_bbox/bbox_pred_list_{}.npy'.format(i)
        bbox_pred = np.load(p2)
        p3 = '/Users/liuhui29/Downloads/npy0106/get_bbox/mlvl_anchors_{}.npy'.format(i)
        mlvl_anchors = np.load(p3)
        
        cls_score_list.append(paddle.to_tensor(cls_score))
        bbox_pred_list.append(paddle.to_tensor(bbox_pred))
        mlvl_anchors_list.append(paddle.to_tensor(mlvl_anchors))

    img_shape = (1024, 593, 3)
    scale_factor = 0.36755204594400576
    cfg = {'nms_pre': 2000, 'min_bbox_size': 0, 'score_thr': 0.05, 'nms': {'type': 'nms_rotated', 'iou_thr': 0.1}, 'max_per_img': 2000}
    res = get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors_list, img_shape, scale_factor,
                            cfg, rescale=False, use_sigmoid_cls=False)
    print(res)
    det_bbox, det_cls = res
    disp_res(det_bbox, det_cls)