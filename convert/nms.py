#!/usr/bin/env python
# _*_ coding: utf-8 _*_


import cv2
import numpy as np


"""
    Non-max Suppression Algorithm

    @param list  Object candidate bounding boxes
    @param list  Confidence score of bounding boxes
    @param float IoU threshold

    @return Rest boxes after nms operation
"""
def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        print('index', index)
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        
        print('x1,y1,x2,y2',(x1,y1,x2,y2))

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        print('ratio', ratio)
        input('ratio')

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score


def cpu_nms(boxes, scores, score_threshold, iou_threshold, max_num=None):
    """
    :param boxes:[N, 4] / 'N' means not sure
    :param scores:[N, 1]
    :param score_threshold: float
    :param iou_threshold:a scalar
    :param max_num:
    :return:keep_index
    """
    # boxes format : [xmin, ymin, xmax, ymax]
    assert isinstance(boxes, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert boxes.ndim == 2
    assert scores.ndim == 1
    assert boxes.shape[-1] == 4
    assert (boxes[2, 3] >= boxes[:, [0, 1]]).all(), 'boxes format must be [xmin, ymin, xmax, ymax]'
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
        print('before cpu_iou box1 box_copy', box1.shape, box_copy.shape)
        ious = cpu_iou(box1, box_copy)
        print('after cpu_iou ious', ious.shape)
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
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).all(), 'format of bbox must be [xmin, ymin, xmax, ymax]'
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).all(), 'format of bbox must be [xmin, ymin, xmax, ymax]'

    bbox1_area = np.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, axis=-1)
    bbox2_area = np.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, axis=-1)

    intersection_ymax = np.minimum(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = np.minimum(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = np.maximum(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = np.maximum(bbox1[:, 0], bbox2[:, 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin + 1)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out

# Image name
image_name = 'convert/nms.jpg'

# Bounding boxes
bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]
bounding_boxes = np.array(bounding_boxes)
confidence_score = np.array(confidence_score)
print('bounding_boxes', bounding_boxes.shape, 'confidence_score', confidence_score.shape)

# Read image
image = cv2.imread(image_name)

# Copy image as original
org = image.copy()

# Draw parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

# IoU threshold
threshold = 0.4

# Draw bounding boxes and confidence score
for (start_x, start_y, end_x, end_y), confidence in zip(bounding_boxes, confidence_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

# Run non-max suppression algorithm
#picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)
keep_index = cpu_nms(bounding_boxes, confidence_score, score_threshold=0.01, iou_threshold=threshold)
picked_boxes = bounding_boxes[keep_index]
picked_score = confidence_score[keep_index]

# Draw bounding boxes and confidence score after non-maximum supression
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

# Show image
#cv2.imshow('Original', org)
#cv2.imshow('NMS', image)
#cv2.waitKey(0)
cv2.imwrite('convert/nms_res.jpg', image)