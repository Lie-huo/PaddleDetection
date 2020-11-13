import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbone.darknet import ConvBNLayer

def anchor2offset(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    import torch
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw[:, None]*xx, dh[:, None]*yy
        xr = cos[:, None]*x-sin[:, None]*y
        yr = sin[:, None]*x+cos[:, None]*y
        x_anchor, y_anchor = xr+x_ctr[:, None], yr+y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    num_imgs, H, W = anchors.shape[:3]
    featmap_size = (H, W)
    offset_list = []
    for i in range(num_imgs):
        anchor = anchors[i].reshape(-1, 5)  # (NA,5)
        offset = _calc_offset(anchor, kernel_size, featmap_size, stride)
        offset_list.append(offset)  # [2*ks**2,H,W]
    offset_tensor = torch.stack(offset_list, dim=0)
    return offset_tensor


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

        # self._init_layers()
        
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
    
        self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1)
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
    
        if self.align_conv_type == 'AlignConv':
            self.align_conv = AlignConv(
                self.feat_channels, self.feat_channels, kernel_size=self.align_conv_size)
        elif self.align_conv_type == 'DCN':
            self.align_conv_offset = nn.Conv2d(self.feat_channels, 18, 1)
            self.align_conv = DeformConv(self.feat_channels, self.feat_channels,
                                         self.align_conv_size, padding=(self.align_conv_size - 1) // 2)
        elif self.align_conv_type == 'GA_DCN':
            self.align_conv_offset = nn.Conv2d(5, 18, 1)
            self.align_conv = DeformConv(self.feat_channels, self.feat_channels,
                                         self.align_conv_size, padding=(self.align_conv_size - 1) // 2)
        elif self.align_conv_type == 'Conv':
            self.align_conv = nn.Conv2d(self.feat_channels, self.feat_channels,
                                        self.align_conv_size, padding=(self.align_conv_size - 1) // 2)
    
        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)
    
        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
    
        self.odm_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.odm_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

    def forward(self, feats):
        print('feats', len(feats))
        assert len(feats) == len(self.mask_anchors)
        
        fam_cls = []
        fam_reg = []
        
        odm_cls = []
        odm_reg = []
        
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