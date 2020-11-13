from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch
import numpy as np

__all__ = ['S2ANet']


# anchor to rotated_anchor
def rect2rbox(bboxes0):
    """
    :param bboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    bboxes = []
    for bbox in bboxes0:
        np_bbox = bbox
        np_bbox = np_bbox.reshape(-1, 4)
        bboxes.append(np_bbox)

    print('create before', bboxes[0][0:10, :])
    bboxes = np.concatenate(bboxes, axis=0)
    print('bboxes debug', bboxes.shape)
    print('create before', bboxes[0:10, :])
    
    num_boxes = bboxes.shape[0]
    
    x_ctr = (bboxes[:, 2]+bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3]+bboxes[:, 1]) / 2.0
    edges1 = np.abs(bboxes[:, 2]-bboxes[:, 0])
    edges2 = np.abs(bboxes[:, 3]-bboxes[:, 1])
    # angles = bboxes.new_zeros(num_boxes)
    angles = np.zeros(num_boxes, dtype=bboxes.dtype)

    inds = edges1 < edges2

    rboxes = np.stack((x_ctr, y_ctr, edges1, edges2, angles), axis=1)
    rboxes[inds, 2] = edges2[inds]
    rboxes[inds, 3] = edges1[inds]
    rboxes[inds, 4] = np.pi / 2.0
    
    print('create after', rboxes.shape, rboxes[0:10, :])
    #print('rect2rbox return', rboxes)
    return rboxes



@register
class S2ANet(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'backbone',
        'neck',
        's2anet_head'
    ]

    def __init__(self,
                 anchor,
                 backbone,
                 neck,
                 s2anet_head,
                 *args,
                 **kwargs):
        super(S2ANet, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.backbone = backbone
        self.neck = neck
        self.s2anet_head = s2anet_head

    def model_arch(self, ):
        # Backbone
        body_feats = self.backbone(self.inputs)

        # neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)

        print('len body_feats', len(body_feats))
        for k in body_feats:
            print(k.shape)
        

        # Retain Head
        print('anchor', self.anchor)
        print('body_feats', len(body_feats))
        self.anchor_list_center = []
        self.anchor_list_xywh = []
        self.np_anchor_list_x1y1x2y2 = []
        for feat in body_feats:
            anchor_center, anchor_xywh = self.anchor(feat)
            self.anchor_list_center.append(anchor_center)
            self.anchor_list_xywh.append(anchor_xywh)
            anchor_x1y1x2y2 = anchor_xywh.numpy()
            anchor_x1y1x2y2[..., 2] = anchor_x1y1x2y2[..., 2] + anchor_x1y1x2y2[..., 0]
            anchor_x1y1x2y2[..., 3] = anchor_x1y1x2y2[..., 3] + anchor_x1y1x2y2[..., 1]
            self.np_anchor_list_x1y1x2y2.append(anchor_x1y1x2y2)
            print('anchor_out:', anchor_center.shape, anchor_xywh.shape)

        print('create model anchor_out np_anchor_list_x1y1x2y2', len(self.np_anchor_list_x1y1x2y2))

        self.anchor_list_xywhr = rect2rbox(self.np_anchor_list_x1y1x2y2)
        print('anchor convert finish!')
        print('create model anchor_out anchor_list_xywhr', len(self.anchor_list_xywhr))
        input('xxx')
        self.s2anet_head_outs = self.s2anet_head(body_feats)

    def get_loss(self, ):
        loss = self.s2anet_head.get_loss(self.inputs, self.yolo_head_outs)
        return loss

    def get_pred(self, ):
        outs = {
            'bbox': self.gbd['predicted_bbox'].numpy(),
            'bbox_nums': self.gbd['predicted_bbox_nums'].numpy(),
            'mask': self.gbd['predicted_mask'].numpy(),
            'im_id': self.gbd['im_id'].numpy(),
        }
        return outs
