#coding=utf-8
import numpy as np
import paddle


def delta2rbox_np(Rrois,
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

    # means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    # stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)

    # means = paddle.to_tensor(means)
    # stds = paddle.to_tensor(stds)
    #deltas = deltas.numpy()
    denorm_deltas = deltas * stds + means
    print('denorm_deltas', denorm_deltas.shape)

    '''
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]
    '''

    dx = denorm_deltas[:, 0]
    dy = denorm_deltas[:, 1]
    dw = denorm_deltas[:, 2]
    dh = denorm_deltas[:, 3]
    dangle = denorm_deltas[:, 4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = np.clip(dw, -max_ratio, max_ratio)
    dh = np.clip(dh, -max_ratio, max_ratio)

    print('Rrois', Rrois.shape)

    Rroi_x = Rrois[:, 0]
    Rroi_y = Rrois[:, 1]
    Rroi_w = Rrois[:, 2]
    Rroi_h = Rrois[:, 3]
    Rroi_angle = Rrois[:, 4]

    print('Rrois start calc',Rroi_x.shape, Rroi_angle.shape)
    print('dx', dx.shape, Rroi_w.shape, Rroi_angle.shape)
    print('dy', dy.shape, Rroi_h.shape, Rroi_angle.shape, Rroi_x.shape)
    gx = dx * Rroi_w * np.cos(Rroi_angle) - dy * Rroi_h * np.sin(Rroi_angle) + Rroi_x


    gy = dx * Rroi_w * np.sin(Rroi_angle) \
         + dy * Rroi_h * np.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * np.exp(dw)
    gh = Rroi_h * np.exp(dh)

    ga = np.pi * dangle + Rroi_angle
    ga = (ga + np.pi / 4) % np.pi - np.pi / 4

    ttt = [gx, gy, gw, gh ,ga]
    for i,e in enumerate(['gx', 'gy', 'gw', 'gh', 'ga']):
        print('e',e,ttt[i].shape, ttt[i].sum())
    input('pause')
    bboxes = np.stack([gx, gy, gw, gh, ga], axis=-1)
    print('bboxes', bboxes.shape, bboxes.sum())
    #bboxes = paddle.to_tensor(bboxes)
    input('pause')
    return bboxes


def bbox_decode(bbox_preds,
                anchors,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1]):
    """decode bbox from deltas
    Args:
        bbox_preds: [N,5,H,W]
        anchors: [H*W,5]
    return:
        bboxes: [N,H,W,5]
    """
    print('bbox_preds', bbox_preds.shape)
    num_imgs, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred
        bboxes = delta2rbox_np(
            anchors,
            bbox_delta,
            means,
            stds,
            wh_ratio_clip=1e-6)
        #bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return np.stack(bboxes_list, axis=0)


'''
bbox_preds = np.load('demo/fam_reg_128.npy')
anchors = np.load('demo/init_anchors_rect2rbox_128.npy')
print('bbox_preds', bbox_preds.shape, bbox_preds.sum())
print('anchors', anchors.shape, anchors.sum())

# TODO: 8457096
res = bbox_decode(bbox_preds, anchors)
'''

g_torch = None
g_np = None

def anchor2offset_torch(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        print('torch, xx yy', xx.shape, yy.shape)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        print('torch, xx yy', xx.shape, yy.shape)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        print('xc', xc.shape, xx.shape)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        print('x_conv', x_conv.shape)

        # get sampling locations of anchors
        print('before unbind', anchors.shape)
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw[:, None]*xx, dh[:, None]*yy
        global g_torch
        g_torch = x
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
        print('offset', offset.shape)
        offset = offset.reshape(anchors.size(0), -1)
        print('offset', offset.shape)
        offset = offset.permute(1, 0).reshape(-1, feat_h, feat_w)
        print('offset', offset.shape)
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


def anchor2offset_np(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = np.arange(-pad, pad + 1)
        yy, xx = np.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = np.arange(0, feat_w)
        yc = np.arange(0, feat_h)
        yc, xc = np.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)

        print('np xc', xc.shape, xx.shape)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy
        print('np x_conv', x_conv.shape, np.sum(x_conv-g_torch.numpy()))
        np.save('demo/2021_x_conv.npy', x_conv)
        np.save('demo/2021_y_conv.npy', y_conv)

        # get sampling locations of anchors
        # x_ctr, y_ctr, w, h, a = np.unbind(anchors, dim=1)
        x_ctr = anchors[:, 0]
        y_ctr = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]
        a = anchors[:, 4]

        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = np.cos(a), np.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw[:, None]*xx, dh[:, None]*yy
        print(np.sum(x - g_torch.numpy()))
        xr = cos[:, None]*x-sin[:, None]*y
        yr = sin[:, None]*x+cos[:, None]*y
        x_anchor, y_anchor = xr+x_ctr[:, None], yr+y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = np.stack([offset_y, offset_x], axis=-1)
        # NA,ks*ks*2
        print('now offset', offset.shape)
        #offset = offset.reshape(anchors.size(
        #    0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        offset = offset.reshape(offset.shape[0], -1)
        offset = np.transpose(offset, [1,0]).reshape(-1, feat_h, feat_w)
        return offset

    num_imgs, H, W = anchors.shape[:3]
    featmap_size = (H, W)
    offset_list = []
    for i in range(num_imgs):
        anchor = anchors[i].reshape(-1, 5)  # (NA,5)
        offset = _calc_offset(anchor, kernel_size, featmap_size, stride)
        offset_list.append(offset)  # [2*ks**2,H,W]
    offset_tensor = np.stack(offset_list, axis=0)
    return offset_tensor


def anchor2offset_paddle(anchors, kernel_size, stride):
    """
    Args:
        anchors: [N,H,W,5]
        kernel_size: int
        stride: int
    """
    def _calc_offset(anchors, kernel_size, featmap_size, stride):
        dtype = anchors.dtype
        feat_h, feat_w = featmap_size
        pad = (kernel_size - 1) // 2
        idx = paddle.arange(-pad, pad + 1, dtype=dtype)
        yy, xx = paddle.meshgrid(idx, idx)
        xx = paddle.reshape(xx, [-1])
        yy = paddle.reshape(yy, [-1])

        # get sampling locations of default conv
        xc = paddle.arange(0, feat_w, dtype=dtype)
        yc = paddle.arange(0, feat_h, dtype=dtype)
        yc, xc = paddle.meshgrid(yc, xc)
        xc = paddle.reshape(xc, [-1, 1])
        yc = paddle.reshape(yc, [-1, 1])
        x_conv = xc + xx
        y_conv = yc + yy

        # get sampling locations of anchors
        # x_ctr, y_ctr, w, h, a = np.unbind(anchors, dim=1)
        x_ctr = anchors[:, 0]
        y_ctr = anchors[:, 1]
        w = anchors[:, 2]
        h = anchors[:, 3]
        a = anchors[:, 4]

        x_ctr = paddle.reshape(x_ctr, [x_ctr.shape[0], 1])
        y_ctr = paddle.reshape(y_ctr, [y_ctr.shape[0], 1])
        w = paddle.reshape(w, [w.shape[0], 1])
        h = paddle.reshape(h, [h.shape[0], 1])
        a = paddle.reshape(a, [a.shape[0], 1])

        x_ctr = x_ctr / stride
        y_ctr = y_ctr / stride
        w_s = w / stride
        h_s = h / stride
        cos, sin = paddle.cos(a), paddle.sin(a)
        dw, dh = w_s / kernel_size, h_s / kernel_size
        x, y = dw*xx, dh*yy
        xr = cos*x-sin*y
        yr = sin*x+cos*y
        x_anchor, y_anchor = xr+x_ctr, yr+y_ctr
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = paddle.stack([offset_y, offset_x], axis=-1)
        # NA,ks*ks*2
        print('now offset', offset.shape)
        #offset = offset.reshape(anchors.size(
        #    0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        offset = paddle.reshape(offset, [offset.shape[0], -1])
        offset = paddle.transpose(offset, [1,0])
        offset = paddle.reshape(offset, [-1, feat_h, feat_w])
        return offset

    num_imgs, H, W = anchors.shape[:3]
    featmap_size = (H, W)
    offset_list = []
    for i in range(num_imgs):
        print('anchors[i]', anchors[i].shape)
        anchor = paddle.reshape(anchors[i], [-1, 5])  # (NA,5)
        offset = _calc_offset(anchor, kernel_size, featmap_size, stride)
        offset_list.append(offset)  # [2*ks**2,H,W]
    offset_tensor = paddle.stack(offset_list, axis=0)
    return offset_tensor



# in  torch.Size([1, 128, 76, 5]) ten     sum (8457096., device='cuda:0')
# result 2021_debug_offset anchor2offset torch.Size([1, 18, 128, 76]) tensor(93547.1562, device='cuda:0')

# 使用本地np.save 和 np.load之后
# input:  (1, 128, 76, 5) 8457097.0
# output: torch.Size([1, 18, 128, 76]) tensor(93547.1484)
anchor_np = np.load('/Users/liuhui29/Downloads/npy_2021/2021_debug_anchors_128.npy')
print(anchor_np.shape, anchor_np.sum())

import torch
anchor_torch = torch.from_numpy(anchor_np)
result_torch = anchor2offset_torch(anchor_torch, 3, 8)
print(result_torch.shape, result_torch.sum(), result_torch.mean())


result_np = anchor2offset_np(anchor_np, 3, 8)
print(result_np.shape, result_np.sum(), result_np.mean())

print('\n\n\n')
anchor_paddle = paddle.to_tensor(anchor_np)
result_paddle = anchor2offset_paddle(anchor_paddle, 3, 8)
print(result_paddle.shape, result_paddle.sum(), result_paddle.mean())