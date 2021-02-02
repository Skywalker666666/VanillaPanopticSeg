#python semantic_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/models/PSPNet_resnet50_1_epoch.params --test-batch-size 1 --eval --aux



#python semantic_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/models/PSP_Resnet50_1epoch_auxon_batchs1_lr1en5.params --test-batch-size 1 --eval --aux

# test model zoo
#python semantic_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --test-batch-size 1 --eval --aux --model-zoo psp_resnet101_coco
# this one is failed because of instance segmentation(21) and semantic segmentation(53)


#18-01-2021
#python semantic_segmentation_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/models/PSP_Resnet50_1epoch_auxon_batchs1_lr1en5.params --test-batch-size 4 --eval --aux

#19-01-2021 noon
#python semantic_segmentation_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --test-batch-size 4 --eval --aux



#python instance_segmentation_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume /home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/runs/coco/psp/default/PSPNet_1epoch_21class_instance.params --test-batch-size 4 --eval --aux

# for viz
#turned off eval
#python semantic_segmentation_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --test-batch-size 1 --aux

#19-01-2021 Eve
#python predict_semantic.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0119_with92_demo.json

# without cuda, so slow
#python predict_semantic.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0119_with92.json  --no-cuda


# with cuda, faster
#python predict_semantic.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0119_with92.json


# 20-01-2021
#python semantic_segmentation_test.py  --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-3 --resume     ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_54c_epoch2of2_auxon_batchs4_lr1en3.params --test-batch-size 4 --eval --aux

#python predict_semantic.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_54c0_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0122_without92_54_0.json




#22-01-2021
#python semantic_segmentation_test.py  --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-3 --resume     ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_54c0_epoch2of2_auxon_batchs4_lr1en3.params --test-batch-size 4 --eval --aux


#26-01-2021
# for single image, test run for ros node integration
# for 54 category with 'thing' as 0
#python predict_semantic_single_img.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_54c0_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0126_without92_54_0.json

# for 53 categories
#python predict_semantic_single_img.py --resume ~/Documents/Panoptic_Segement/Cocopanopticapi/VanillaPanopticSeg/src/trainers/runs/coco/psp/default/PSP_Resnet50_epoch2of2_auxon_batchs4_lr1en3.params --sem-seg-result-file predictions_test_semantic_segmentation_0126_without92_54_0.json --no-cuda


python instance_segmentation_test.py --model psp --dataset coco --batch-size 1 --epochs 1 --lr 1e-5 --resume /home/zhiliu/Documents/Weapons/gluon-cv/scripts/segmentation/runs/coco/psp/default/PSPNet_1epoch_21class_instance.params --test-batch-size 4 --eval --aux

