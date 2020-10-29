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

import logging
import numpy as np
import math

__all__ = ["poly2rbox_single", "poly2rbox_single_v2", "rbox2poly_single"]

logger = logging.getLogger(__name__)


def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
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


def poly2rbox_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
    
    angle = 0
    width = 0
    height = 0
    
    if edge1 > edge2:
        width = edge1
        height = edge2
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))
    
    if angle > np.pi * 3 / 4:
        angle -= np.pi
    if angle < -np.pi / 4:
        angle += np.pi
    
    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rbox = np.array([x_ctr, y_ctr, width, height, angle])
    
    return rbox


def poly2rbox_single_v2(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))
    
    angle = 0
    width = 0
    height = 0
    
    if edge1 > edge2:
        width = edge1
        height = edge2
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))
    
    if angle > np.pi * 3 / 4:
        angle -= np.pi
    if angle < -np.pi / 4:
        angle += np.pi
    
    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    
    return float(x_ctr), float(y_ctr), float(width), float(height), float(angle)


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

def draw_rbox_1(image, pts):
    """
    draw_rbox
    image:
    pts: 8 points
    """

    pts = np.array(pts).astype(np.float32)
    pts = pts.reshape(4, 2)
    print(pts.shape)
    
    cen_pts = np.mean(pts, axis=0)
    tt = pts[0, :]
    rr = pts[1, :]
    bb = pts[2, :]
    ll = pts[3, :]
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 2, 1)
    
    return image


def draw_rbox_2(image, pts):
    """
    draw_rbox
    image:
    pts: 4 points [4, 2] array
    """
    pts = np.array(pts)
    pts = pts.reshape(4, 2)
    pts = np.around(pts, decimals=1).astype(np.int32)
    print(pts)
    cv2.line(image, (pts[0][0], pts[0][1]), (pts[1][0], pts[1][1]), (0, 0, 255), 2, 1)
    cv2.line(image, (pts[1][0], pts[1][1]), (pts[2][0], pts[2][1]), (255, 0, 255), 2, 1)
    cv2.line(image, (pts[2][0], pts[2][1]), (pts[3][0], pts[3][1]), (0, 255, 0), 2, 1)
    cv2.line(image, (pts[3][0], pts[3][1]), (pts[0][0], pts[0][1]), (255, 0, 0), 2, 1)
    
    return image
    

if __name__ == '__main__':
    import cv2
    img = cv2.imread('../../demo/DOTA_sample/P2060.png')
    print(img.shape)
    
    anno_lst = []
    for line in open('../../demo/DOTA_sample/P2060.txt'):
        elems = line.strip().split(' ')
        elems[0:8] = [float(e) for e in elems[0:8]]
        anno_lst.append(elems)
    
    for x in anno_lst:
        box1 = x[0:8]
        img = draw_rbox_2(img, box1)
    cv2.imwrite('test.png', img)
    pass