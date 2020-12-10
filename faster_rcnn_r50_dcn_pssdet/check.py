import os,sys


dy_name_lst = []

for line in open('faster_rcnn_r50_vd_fpn_1x_coco_pssdet_list.txt'):
    line = line.strip()

    if line.find('state_dict') >= 0:
        emels = line.split(' ')
        name_lst = emels[2:]
        for tensor_name in name_lst:
            tensor_name = tensor_name.replace("['", "")
            tensor_name = tensor_name.replace("',", "")
            tensor_name = tensor_name.replace("'", "")
            tensor_name = tensor_name.replace("']", "")
            tensor_name = tensor_name.replace("]", "")
            tensor_name = tensor_name.replace("[", "")
            if len(tensor_name) <= 0:continue
            print(tensor_name)
            dy_name_lst.append(tensor_name)
 
print(len(dy_name_lst))
input('xx')

print('xxx')
weights_map_lst = [e.strip().split(' ')[-1] for e in open('weight_name_map_faster_rcnn_dcn_r50_vd_fpn.txt')]
for x in weights_map_lst:
    if x not in dy_name_lst:
        print('x=\t{}'.format(x))
        
print('*'*24 + '\n')
for x in dy_name_lst:
    if x not in weights_map_lst:
        print('x=\t{}'.format(x))