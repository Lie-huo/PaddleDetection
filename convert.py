import os
import sys
import pickle
import paddle
import paddle.fluid as fluid
from ppdet.utils.download import get_weights_path
import numpy as np


class Load(paddle.nn.Layer):
     def __init__(self):
         super(Load, self).__init__()

     def forward(self, filename):
         weight = self.create_parameter(
         shape=[1],
         dtype='float32',
         default_initializer=fluid.initializer.ConstantInitializer(0.0))
         self._helper.append_op(
         type='load',
         inputs={},
         outputs={'Out': [weight]},
         attrs={'file_path': filename})
         return weight

def convert(weights, weight_name_map_file, target_name):
    weight_name_map = {}
    with open(weight_name_file) as f:
        for line in f.readlines():
            fields = line.split()
            weight_name_map[fields[0]] = fields[1]

    dst = {}
    weights = get_weights_path(weights)
    if os.path.isdir(weights):
        for k, v in weight_name_map.items():
            k_path = os.path.join(weights, k)
            if os.path.exists(k_path):
                load = Load()
                weiht_path = os.path.join(weights, k)
                if weiht_path.find('_conv_offset.') > 0:
                    print(weiht_path)
                weight = load(os.path.join(weights, k))
                weight = weight.numpy()
                print('weight:', k, weight.shape)
                dst[v] = weight
                if True:#weiht_path.find('_conv_offset.') > 0:
                    print(weiht_path)
                    save_path = weiht_path + '.npy'
                    print('save_path', save_path)
                    np.save(save_path, weight)
                    #input('xxxx')
            else:
                print("warning: static weight file {} not found".format(k))
    else:
        src = pickle.load(open(weights))
        for k, v in weight_name_map.items():
            dst[v] = src[k]
    pickle.dump(dst, open(target_name, 'wb'), protocol=2)


if __name__ == "__main__":
    weight_path = sys.argv[1]
    weight_name_file = sys.argv[2]
    target_name = sys.argv[3]
    convert(weight_path, weight_name_file, target_name)
