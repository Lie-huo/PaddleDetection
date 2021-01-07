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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np
from PIL import Image
import paddle
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.checkpoint import load_weight
import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    args = parser.parse_args()
    return args


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def visual_rbox(image, catid2name, bbox_res, threshold):
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(image)
    
    catid2color = {}
    from ppdet.utils.colormap import colormap
    color_list = colormap(rgb=True)[:40]
    
    for pred_bbox in bbox_res:
        # label, score, x1,y1,x2,y2 ...
        catid = int(pred_bbox[0])
        score = pred_bbox[1]
        if score < threshold:
            continue
        bbox = pred_bbox[2:]
        pt_lst = [int(e) for e in bbox]
        x1, y1, x2, y2, x3, y3, x4, y4 = pt_lst[0], pt_lst[1], pt_lst[2], pt_lst[3], pt_lst[4], \
                                         pt_lst[5], pt_lst[6], pt_lst[7]

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])
        
        draw.line(
            [(x1, y1), (x2, y2), (x2, y2), (x3, y3),
                 (x3, y3), (x4, y4), (x4, y4), (x1, y1)],
            width=2,
            fill=color)

        text = "{} {:.2f}".format(catid2name[catid], score)
        tw, th = draw.textsize(text)
        draw.rectangle([(x1 + 1, y1 - th), (x1 + tw + 1, y1)], fill=color)
        draw.text((x1 + 1, y1 - th), text, fill=(255, 255, 255))
    
    return image

def run(FLAGS, cfg, place):

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # data
    dataset = cfg.TestDataset
    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)
    test_loader, _ = create('TestReader')(dataset, cfg['worker_num'], place)

    # TODO: support other metrics
    imid2path = dataset.get_imid2path()

    from ppdet.utils.coco_eval import get_category_info
    anno_file = dataset.get_anno()
    with_background = cfg.with_background
    use_default_label = dataset.use_default_label
    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # Init Model
    load_weight(model, cfg.weights)

    # Run Infer 
    for iter_id, data in enumerate(test_loader):
        # forward
        model.eval()
        outs = model(data, cfg.TestReader['inputs_def']['fields'], 'infer')

        logger.info('Infer iter {}'.format(iter_id))
        print('outs', outs)
        im_ids = outs['im_id']
        bboxes = outs['bbox']
        bbox_num = outs['bbox_num']
        print('im_ids', im_ids.shape, im_ids, 'bbox_num', bbox_num.shape, bbox_num, bboxes.shape)
        input('vis rbox')
        start = 0
        for i, im_id in enumerate(im_ids):
            im_id = im_ids[i]
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            end = start + bbox_num[i]

            # use VisualDL to log original image
            if FLAGS.use_vdl:
                original_image_np = np.array(image)
                vdl_writer.add_image(
                    "original/frame_{}".format(vdl_image_frame),
                    original_image_np, vdl_image_step)
            

            catid2name1 = {}
            name_lst = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
                        'ground track field', 'harbor', 'bridge', 'large vehicle', 'small vehicle', 'helicopter',
                        'roundabout', 'soccer ball field', 'swimming pool']
            for (ii,v) in enumerate(name_lst):
                catid2name1[ii] = v
            image = visual_rbox(image, catid2name1, bboxes[i, :, :], threshold=0.1)
            #image = visualize_results(image, bbox_res, mask_res,
            #                          int(im_id), catid2name,
            #                          FLAGS.draw_threshold)

            # use VisualDL to log image with bbox
            if FLAGS.use_vdl:
                infer_image_np = np.array(image)
                vdl_writer.add_image("bbox/frame_{}".format(vdl_image_frame),
                                     infer_image_np, vdl_image_step)
                vdl_image_step += 1
                if vdl_image_step % 10 == 0:
                    vdl_image_step = 0
                    vdl_image_frame += 1

            # save image with detection
            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)
            start = end


def main():
    FLAGS = parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()

    place = 'gpu:{}'.format(ParallelEnv().dev_id) if cfg.use_gpu else 'cpu'
    place = paddle.set_device(place)
    run(FLAGS, cfg, place)


if __name__ == '__main__':
    main()
