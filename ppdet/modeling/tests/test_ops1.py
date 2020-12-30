#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.dygraph import base

import ppdet.modeling.ops as ops
from ppdet.modeling.tests.test_base import LayerTest


class TestAnchorGenerator(LayerTest):
    def test_anchor_generator(self):
        b, c, h, w = 2, 48, 16, 16
        input_np = np.random.rand(2, 48, 16, 16).astype('float32')
        with self.static_graph():
            input = paddle.static.data(
                name='input', shape=[b, c, h, w], dtype='float32')

            anchor, var = ops.anchor_generator(
                input=input,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
            
            anchor_np, var_np = self.get_static_graph_result(
                feed={'input': input_np, },
                fetch_list=[anchor, var],
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = base.to_variable(input_np)

            anchor_dy, var_dy = ops.anchor_generator(
                input=inputs_dy,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
            anchor_dy_np = anchor_dy.numpy()
            var_dy_np = var_dy.numpy()
        print(anchor_np.shape)
        print(anchor_np[:,:,1,:])
        self.assertTrue(np.array_equal(anchor_np, anchor_dy_np))
        self.assertTrue(np.array_equal(var_np, var_dy_np))


if __name__ == '__main__':
    unittest.main()
