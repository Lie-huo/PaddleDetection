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

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import Layer
from paddle.nn import Conv2D
from paddle.nn.initializer import XavierUniform
from paddle.fluid.regularizer import L2Decay
from ppdet.core.workspace import register, serializable


@register
@serializable
class FPN(Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 min_level=0,
                 max_level=4,
                 spatial_scale=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 num_outs=5):

        super(FPN, self).__init__()
        self.lateral_convs = []
        self.fpn_convs = []
        self.has_extra_convs = has_extra_convs
        self.num_outs = num_outs

        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale

        self.extra_convs_on_inputs = True
        
        fan = out_channel * 3 * 3
        
        # first make len(self.fpn_convs) == self.num_outs, len(self.lateral_convs) == self.num_outs
        self.fpn_convs = []
        self.lateral_convs = []

        for i in range(min_level, max_level):
            if i == 3:
                lateral_name = 'fpn_inner_res5_sum'
            else:
                lateral_name = 'fpn_inner_res{}_sum_lateral'.format(i + 2)
            in_c = in_channels[i]
            lateral = self.add_sublayer(
                lateral_name,
                Conv2D(
                    in_channels=in_c,
                    out_channels=out_channel,
                    kernel_size=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=in_c)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            
            # add lateral conv
            self.lateral_convs.append(lateral)

            fpn_name = 'fpn_res{}_sum'.format(i + 2)
            fpn_conv = self.add_sublayer(
                fpn_name,
                Conv2D(
                    in_channels=out_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    padding=1,
                    weight_attr=ParamAttr(
                        initializer=XavierUniform(fan_out=fan)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            self.fpn_convs.append(fpn_conv)


        for i in range(len(self.fpn_convs)):
            print('i==', i, self.fpn_convs[i])

        # has_extra_convs == True
        highest_backbone_level = 4
        extra_levels = self.num_outs - highest_backbone_level + self.min_level
        print('extra_levels====', extra_levels)
        if self.has_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_c = in_channels[self.max_level - 1]
                else:
                    in_c = out_channel
                    
                fan = in_c * 3 * 3
                #fpn_blob_in = fpn_blob
                fpn_name = 'fpn_res{}_sum_extra'.format(i)
                fpn_conv = self.add_sublayer(
                    fpn_name,
                    Conv2D(
                        in_channels=in_c,
                        out_channels=out_channel,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        weight_attr=ParamAttr(
                            initializer=XavierUniform(fan_out=fan)),
                        bias_attr=ParamAttr(
                            learning_rate=2., regularizer=L2Decay(0.))))
                self.fpn_convs.append(fpn_conv)
                
            for k in range(len(self.lateral_convs)):
                if self.lateral_convs[k] is None:
                    print('k== none', k)
                    continue
                print('k==', k, self.lateral_convs[k].weight.shape)
        

    def forward(self, body_feats):
        for e in body_feats:
            print('forward of fpn', e.shape)
        laterals = []
        print('fpn min_level', self.min_level, self.max_level)
        
        for i in range(len(self.lateral_convs)):
            laterals.append(self.lateral_convs[i](body_feats[i+self.min_level]))

        print('len(laterals)', len(laterals))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            print('now i', i)
            upsample = F.interpolate(
                laterals[i],
                scale_factor=2.,
                mode='nearest', )
            laterals[i - 1] = laterals[i - 1] + upsample

        for i in range(len(self.fpn_convs)):
            print('fpn_convs before last i=', i, self.fpn_convs[i].weight.shape)

        # fpn_output: part1: from original levels
        fpn_output = []
        for i in range(used_backbone_levels):
            fpn_output.append(self.fpn_convs[i](laterals[i]))
            
        # fpn_output: part2: add extra levels
        self.extra_convs_on_inputs = True
        if self.num_outs > len(fpn_output):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.has_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    fpn_output.append(F.max_pool2d(fpn_output[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = body_feats[self.max_level - 1]
                    fpn_output.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    fpn_output.append(self.fpn_convs[used_backbone_levels](fpn_output[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    fpn_output.append(self.fpn_convs[i](fpn_output[-1]))
            
        # make self.spatial_scale same length as fpn_output
        for _ in range(len(fpn_output) - len(self.spatial_scale)):
            self.spatial_scale.append(self.spatial_scale[-1])
            
        for i in range(len(fpn_output)):
            print('fpn out i', i, fpn_output[i].shape, self.spatial_scale[i])
        return fpn_output, self.spatial_scale
