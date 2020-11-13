#coding=utf-8
import os,sys
import numpy as np
import cv2


def init_rectangle():
    """
    
    :return:
    """

    img = np.ones((512, 512, 3), np.uint8)
    img = img * 255
    img.astype(np.uint8)

    # draw axis
    cv2.line(img, (0, 256), (511, 256), (0, 0, 0), 1)
    # cv2.putText(img, "x", (511, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.line(img, (256, 0), (256, 511), (0, 0, 0), 1)
    # cv2.putText(img, "y", (511, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
    
    return img


def test_cv2():
    """
    :return:
    """
    
    # draw
    #img = cv2.rectangle(img, (100,100), (200, 300), (0,0,255), 2)
    
    #pts = np.array([[100, 100], [200, 120], [180, 300], [80, 180]])
    #rect = cv2.minAreaRect(pts)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

    for i in range(-90, 180, 10):
        img = init_rectangle()
        
        img = cv2.rectangle(img, (156, 206), (406, 306), (0, 0, 255), 2)
        
        angle = i
        rect = ((256, 256), (300, 100), angle)
        print('i', i, 'rect', rect)
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # draw
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    
        # putText
        cv2.putText(img, "angle: {}".format(angle), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    
        # show
        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_cv2()