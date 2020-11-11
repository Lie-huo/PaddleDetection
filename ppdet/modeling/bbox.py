import numpy as np
import paddle.fluid as fluid
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from . import ops


@register
class Anchor(object):
    __inject__ = ['anchor_generator', 'anchor_target_generator']

    def __init__(self, anchor_generator, anchor_target_generator):
        super(Anchor, self).__init__()
        self.anchor_generator = anchor_generator
        self.anchor_target_generator = anchor_target_generator

    def __call__(self, rpn_feats):
        anchors = []
        num_level = len(rpn_feats)
        for i, rpn_feat in enumerate(rpn_feats):
            anchor, var = self.anchor_generator(rpn_feat, i)
            anchors.append((anchor, var))
        return anchors

    def _get_target_input(self, rpn_feats, anchors):
        rpn_score_list = []
        rpn_delta_list = []
        anchor_list = []
        for (rpn_score, rpn_delta), (anchor, var) in zip(rpn_feats, anchors):
            rpn_score = fluid.layers.transpose(rpn_score, perm=[0, 2, 3, 1])
            rpn_delta = fluid.layers.transpose(rpn_delta, perm=[0, 2, 3, 1])
            rpn_score = fluid.layers.reshape(x=rpn_score, shape=(0, -1, 1))
            rpn_delta = fluid.layers.reshape(x=rpn_delta, shape=(0, -1, 4))

            anchor = fluid.layers.reshape(anchor, shape=(-1, 4))
            var = fluid.layers.reshape(var, shape=(-1, 4))

            rpn_score_list.append(rpn_score)
            rpn_delta_list.append(rpn_delta)
            anchor_list.append(anchor)

        rpn_scores = fluid.layers.concat(rpn_score_list, axis=1)
        rpn_deltas = fluid.layers.concat(rpn_delta_list, axis=1)
        anchors = fluid.layers.concat(anchor_list)
        return rpn_scores, rpn_deltas, anchors

    def generate_loss_inputs(self, inputs, rpn_head_out, anchors):
        assert len(rpn_head_out) == len(
            anchors
        ), "rpn_head_out and anchors should have same length, but received rpn_head_out' length is {} and anchors' length is {}".format(
            len(rpn_head_out), len(anchors))
        rpn_score, rpn_delta, anchors = self._get_target_input(rpn_head_out,
                                                               anchors)

        score_pred, roi_pred, score_tgt, roi_tgt, roi_weight = self.anchor_target_generator(
            bbox_pred=rpn_delta,
            cls_logits=rpn_score,
            anchor_box=anchors,
            gt_boxes=inputs['gt_bbox'],
            is_crowd=inputs['is_crowd'],
            im_info=inputs['im_info'])
        outs = {
            'rpn_score_pred': score_pred,
            'rpn_score_target': score_tgt,
            'rpn_rois_pred': roi_pred,
            'rpn_rois_target': roi_tgt,
            'rpn_rois_weight': roi_weight
        }
        return outs


@register
class AnchorYOLO(object):
    __inject__ = ['anchor_generator']

    def __init__(self, anchor_generator):
        super(AnchorYOLO, self).__init__()
        self.anchor_generator = anchor_generator

    def __call__(self):
        return self.anchor_generator()


@register
class Proposal(object):
    __inject__ = ['proposal_generator', 'proposal_target_generator']

    def __init__(self, proposal_generator, proposal_target_generator):
        super(Proposal, self).__init__()
        self.proposal_generator = proposal_generator
        self.proposal_target_generator = proposal_target_generator

    def generate_proposal(self, inputs, rpn_head_out, anchor_out):
        rpn_rois_list = []
        rpn_prob_list = []
        rpn_rois_num_list = []
        for (rpn_score, rpn_delta), (anchor, var) in zip(rpn_head_out,
                                                         anchor_out):
            rpn_prob = fluid.layers.sigmoid(rpn_score)
            rpn_rois, rpn_rois_prob, rpn_rois_num, post_nms_top_n = self.proposal_generator(
                scores=rpn_prob,
                bbox_deltas=rpn_delta,
                anchors=anchor,
                variances=var,
                im_info=inputs['im_info'],
                mode=inputs['mode'])
            if len(rpn_head_out) == 1:
                return rpn_rois, rpn_rois_num
            rpn_rois_list.append(rpn_rois)
            rpn_prob_list.append(rpn_rois_prob)
            rpn_rois_num_list.append(rpn_rois_num)

        start_level = 2
        end_level = start_level + len(rpn_head_out)
        rois_collect, rois_num_collect = ops.collect_fpn_proposals(
            rpn_rois_list,
            rpn_prob_list,
            start_level,
            end_level,
            post_nms_top_n,
            rois_num_per_level=rpn_rois_num_list)
        return rois_collect, rois_num_collect

    def generate_proposal_target(self, inputs, rois, rois_num, stage=0):
        outs = self.proposal_target_generator(
            rpn_rois=rois,
            rpn_rois_num=rois_num,
            gt_classes=inputs['gt_class'],
            is_crowd=inputs['is_crowd'],
            gt_boxes=inputs['gt_bbox'],
            im_info=inputs['im_info'],
            stage=stage)
        rois = outs[0]
        rois_num = outs[-1]
        targets = {
            'labels_int32': outs[1],
            'bbox_targets': outs[2],
            'bbox_inside_weights': outs[3],
            'bbox_outside_weights': outs[4]
        }
        return rois, rois_num, targets

    def refine_bbox(self, rois, bbox_delta, stage=0):
        out_dim = bbox_delta.shape[1] / 4
        bbox_delta_r = fluid.layers.reshape(bbox_delta, (-1, out_dim, 4))
        bbox_delta_s = fluid.layers.slice(
            bbox_delta_r, axes=[1], starts=[1], ends=[2])

        refined_bbox = ops.box_coder(
            prior_box=rois,
            prior_box_var=self.proposal_target_generator.bbox_reg_weights[
                stage],
            target_box=bbox_delta_s,
            code_type='decode_center_size',
            box_normalized=False,
            axis=1)
        refined_bbox = fluid.layers.reshape(refined_bbox, shape=[-1, 4])
        return refined_bbox

    def __call__(self,
                 inputs,
                 rpn_head_out,
                 anchor_out,
                 stage=0,
                 proposal_out=None,
                 bbox_head_outs=None,
                 refined=False):
        if refined:
            assert proposal_out is not None, "If proposal has been refined, proposal_out should not be None."
            return proposal_out
        if stage == 0:
            roi, rois_num = self.generate_proposal(inputs, rpn_head_out,
                                                   anchor_out)
            self.proposals_list = []
            self.targets_list = []

        else:
            bbox_delta = bbox_head_outs[stage][0]
            roi = self.refine_bbox(proposal_out[0], bbox_delta, stage - 1)
            rois_num = proposal_out[1]
        if inputs['mode'] == 'train':
            roi, rois_num, targets = self.generate_proposal_target(
                inputs, roi, rois_num, stage)
            self.targets_list.append(targets)
        self.proposals_list.append((roi, rois_num))
        return roi, rois_num

    def get_targets(self):
        return self.targets_list

    def get_proposals(self):
        return self.proposals_list


# [x, y, w, h] to [xmin, ymin, xmax, ymax]
def center_form_to_corner_form(locations):
    return paddle.concat([locations[..., :2] - locations[..., 2:] / 2,
                       locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


# [xmin, ymin, xmax, ymax] to [x, y, w, h]
def corner_form_to_center_form(boxes):
    return paddle.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


@register
class S2ANetAnchor(object):
    def __init__(self, aspect_ratios, anchor_start_size, stride):
        self.features_maps = [(75, 75), (38, 38), (19, 19), (10, 10), (5, 5)]
        self.anchor_sizes = [32, 64, 128, 256, 512]
        self.ratios = np.array([0.5, 1, 2])
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        self.image_size = 600
        self.clip = True
    
    def __call__(self, *args, **kwargs):
        anchor_list = []
        for k, (feature_map_w, feature_map_h) in enumerate(self.features_maps):
            for i in range(feature_map_w):
                for j in range(feature_map_h):
                    cx = (j + 0.5) / feature_map_w
                    cy = (i + 0.5) / feature_map_h
                    
                    size = self.anchor_sizes[k] / self.image_size  # 将框体长宽转为 比例形式
                    
                    sides_square = self.scales * size  # 计算方形检测框边长
                    for side_square in sides_square:
                        anchor_list.append([cx, cy, side_square, side_square])  # 添加方形检测框
                    
                    sides_long = sides_square * 2 ** (1 / 2)  # 计算长形检测框长边
                    for side_long in sides_long:
                        anchor_list.append([cx, cy, side_long, side_long / 2])  # 添加长形检测框,短边为长边的一半
                        anchor_list.append([cx, cy, side_long / 2, side_long])
        
        print('anchor_list ddd: ', type(anchor_list), len(anchor_list))
        anchor_list = paddle.fluid.layers.assign(anchor_list) #paddle.tensor(anchor_list)
        print('anchor_list shape:',anchor_list.shape)
        if self.clip:  # 对超出图像范围的框体进行截断
            anchor_list = center_form_to_corner_form(anchor_list)  # 截断时,先转为 [xmin, ymin, xmin, xmax]形式
            anchor_list.clamp_(max=1, min=0)
            anchor_list = corner_form_to_center_form(anchor_list)  # 转回 [x, y, w, h]形式
        return anchor_list