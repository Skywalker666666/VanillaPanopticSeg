# VanillaPanopticSeg

1. Instance Segmentation
predict_instance.py:  
it is instance prediction script from source code of github, but it use the 90 categories label for all DATA 2017, it can save .json annotation result, it use sample dataset from author which is     
def __init__(self, images_info_path: str = '../../data/annotations/mini_test.json', no_cuda: bool = True):

And result is saved in here:
    predictor.save_predictions(destination_dir='../../data/predictions', filename='predictions_test_instance.json')

And this result is used to combine for Panoptic result.



2. Just for fine tuning semantic segmentation for thing object 
Instance_segmentation_test.py:
This is used to evaluate mIoU or generate segmented result mask image by switching eval option. Dataset used is all test 2017.

This script is modified from 
/home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/test.py
Because this is an instance segmentation problem, but no score. If we use PSPNet
Use PSPNet to do instance segmentation for thing. NOT for STUFF.
(that time, my stuff segmentation script is not working, so I use this one to help us for debugging.)
We definitely can detect and segment thing object in a stuff fashion.

For detection with object segments (instance segmentation), it belongs to part of Object Detection.
It use model from:
/home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/runs/coco/psp/default/PSPNet_1epoch_21class_instance.params

And this model is from the training of:
/home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/train.py
Python train.py --model psp --dataset coco --batch-size 4 --epochs 1 --lr 1e-4
Alias as:
Semantic segmentation for STUFF category.
Instance segmentation without score.



Instance_segmentation_test_2.py
Script is from predict_semantic.py, same function with semantic side.
In order to generate result file for model from /home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/runs/coco/psp/default/PSPNet_1epoch_21class_instance.params

This is also only have  21 classes and can be used to generate annotation file of instance segmentation.

Alias as:
Semantic segmentation for STUFF category.
Instance segmentation without score.


predict.py
Just a parent CLASS

3. semantic segmentation for stuff
predict_semantic.py
Used to generate annotation file for semantic segmentation.


semantic_segmentation_test.py
Used to eval mIoU value or generate mask segments, switched by eval argument.

predict_semantic_single_img.py
Used to test single image semantic segmentation inference. Prepare for ros integration. So picture id used in script is just a placeholder


The script to generate ground truth semantic segmentation result is embedded in the  dataloaders/cocosemantic.py temporally.


Multi_purp_visualize.py
Visulization and image save.

