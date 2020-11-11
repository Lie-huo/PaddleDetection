from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['S2ANet']


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
        input('xxx')

        # Retain Head
        self.anchor_list = self.anchor()
        print('create model anchor_list', self.anchor_list)
        self.s2anet_head_outs = self.s2anet_head(self.anchor_list)(body_feats)

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
