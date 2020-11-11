import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer


@register
class S2ANetHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 loss='S2ANetLoss'):
        super(S2ANetHead, self).__init__()
        self.anchor_list = None
        self.num_classes = num_classes
        self.loss = loss

    def forward(self, feats):
        print('feats', len(feats))
        assert len(feats) == len(self.mask_anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):
            yolo_output = self.yolo_outputs[i](feat)
            yolo_outputs.append(yolo_output)
        return yolo_outputs

    def get_loss(self, inputs, head_outputs):
        return self.loss(inputs, head_outputs)


if __name__ == '__main__':
    anchor_generator = S2ANetAnchor()
    anchor_list = anchor_generator()
    print(anchor_list)
    print(anchor_list.shape)