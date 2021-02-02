#!/usr/bin/env python

import os
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import PIL.Image     as Image


import sys
sys.path.insert(1, '../panopticapi/')


import json
from collections import defaultdict
try:
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

from panopticapi.utils import IdGenerator, id2rgb, save_json


def show_original_image(outdir, outname):
    # read a image and show it
    #outdir = 'outdir'
    #i = 0
    #outname = 'predict_mask_' + str(i) + '.png'
    mmask = mpimg.imread(os.path.join(outdir, outname))
    plt.imshow(mmask)
    plt.show()


def show_semseg_result(semseg_json_file, categories_json_file, out_image_file):
    sem_by_image = defaultdict(list)

    with open(semseg_json_file, 'r') as f:
        sem_results = json.load(f)

    print("Semantic segmentation:")
    print("\tJSON file: {}".format(semseg_json_file))


    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {el['id']: el for el in categories_list}


    for sem in sem_results:
        img_id = sem['image_id']
        sem_by_image[sem['image_id']].append(sem)

    id_generator = IdGenerator(categories)

    #pan_segm_id = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint32)
    pan_segm_id = np.zeros((480, 640), dtype=np.uint32)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # To Do: change logic of multiple annotations case, for now, it is override
    # but we can easily learn it from panoptic combine script
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    for ann in sem_by_image[img_id]:
        mask = COCOmask.decode(ann['segmentation'])

        #print(mask.shape())
        plt.imshow(mask)
        plt.show()
        segment_id = id_generator.get_id(ann['category_id'])
        print("id: ") 
        print(ann['category_id']) 
        pan_segm_id[mask==1] = segment_id
        print("segment_id: ") 
        print(segment_id)

        print(id2rgb(pan_segm_id).shape)
    #print(sem_by_image)
    Image.fromarray(id2rgb(pan_segm_id)).save(
        os.path.join("/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/data/predictions/test_result_for_vox/", out_image_file) 
    )


if __name__ == "__main__":
    folder = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/data/predictions/"
    folder2 = "/home/zhiliu/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/panopticapi/"
    #file_name = "predictions_test_semantic_segmentation_0126_with92_54_0.json"
    file_name = "predictions_test_semantic_segmentation_0126_without92_54_0.json"
    #doen't matter although your image id is a fake one, we just need a placeholder when we use
    #this script to visualize random image from other dataset



    #parser.add_argument('--categories_json_file', type=str,
    #                    help="JSON file with Panoptic COCO categories information",

    outdir = "/home/zhiliu/Documents/catkin_ws_VoSM/outputs/test_images_for_panoptic_0127/" 
    outname = "scenenn_camera_frame_94.012380_image.jpg"

    show_original_image(outdir, outname)


    show_semseg_result(os.path.join(folder, file_name), folder2 + "panoptic_coco_categories.json", "result.png")





