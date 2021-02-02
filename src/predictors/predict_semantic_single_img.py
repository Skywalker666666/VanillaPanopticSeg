"""Semantic Segmentation performed by a PSPNet."""
import argparse
import os
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import mxnet as mx
import numpy as np
from mxnet import image
from tqdm import tqdm

from gluoncv.model_zoo.segbase import *
from gluoncv import model_zoo

# for DataParallelModel
from gluoncv.utils.parallel import *



import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')

#from src.models.pspnet import PSPNet
from src.predictors.predict import Predict



def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascalaug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')


    # NEW: for inputfile

    # NEW: for outputfile
    parser.add_argument('--sem-seg-result-file', type=str, default='predictions_test_semantic.json',
                        help='put the file name for segmantic segmentation result if needed')

    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args




class SemanticSegmentation(Predict):
    """Perform the semantic segmentation on a dataset by using a PSPNet."""

    #def __init__(self, model_path: str = '../../models/PSPNet_resnet50_1_epoch.params',
    #def __init__(self, model_path: str = '../../models/PSP_Resnet50_1epoch_auxon_batchs1_lr1en5.params',
    def __init__(self, args, 
                 images_info_path: str = '../../data/annotations/mini_test.json', no_cuda: bool = False):
                 #images_info_path: str = '/home/zhiliu/.mxnet/datasets/coco/annotations/stuff_val2017.json', no_cuda: bool = True):

        Predict.__init__(self, images_info_path, no_cuda)
        self.model_path = args.resume
        self.model = self._load_model()
        self.args = args

    @property
    def classes_name(self) -> list:
        return ['banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower',
                'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
                'railroad',
                'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
                'wall-stone',
                'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
                'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
                'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
                'food-other-merged',
                'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
        #return ['thing', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower',
        #        'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        #        'railroad',
        #        'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick',
        #        'wall-stone',
        #        'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
        #        'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged',
        #        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
        #        'food-other-merged',
        #        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

    @property
    def classes(self) -> list:
        """Category coco index"""
        return [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
                149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]
        #return [0, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
        #        149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
        #        187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    def _load_model(self) -> model_zoo.PSPNet:
        """Load PSPNet and the trained parameters.

        Returns
        -------

        The model with the associate parameters.

        """
        model = model_zoo.PSPNet(nclass=53, backbone='resnet50', pretrained_base=False, ctx=args.ctx)
        model.load_parameters(self.model_path)
        return model

    def predict(self, img_path: str = '/home/zhiliu/Documents/catkin_ws_VoSM/outputs/test_images_for_panoptic_0127/'):
        """Predict the semantic segmentation on a dataset.

        Parameters
        ----------
        img_path : str
            The directory path where the images are stored.

        Returns
        -------

        """
        coco = self._coco
        coco_mask = self._coco_mask

        model = self.model

        # The results will be stored in a list, it needs memory...
        # TODO: flush the memory periodically
        semantic_segmentation = []

        #print("image list: ")
        #print(self._imgs_idx)
        # self._imgs_idx demo: [466319, 523573, 308929, 57540]
        #tbar = tqdm(self._imgs_idx)

        #tbar = tqdm([466319, 523573])
        tbar = tqdm([466319])

        for idx in tbar:
            #img_metadata = coco.loadImgs(idx)[0]
            #path = img_metadata['file_name']
            #img_metadata = {'file_name': 'scenenn_camera_frame_9.887220_image.jpg'}
            #img_metadata = {'file_name': 'scenenn_camera_frame_3.150504_image.jpg'}
            #img_metadata = {'file_name': '000000266409.jpg'}
            #img_metadata = {'file_name': 'scenenn_camera_frame_14.680008_image.jpg'}
            #img_metadata = {'file_name': 'scenenn_camera_frame_181.556172_image.jpg'}
            #img_metadata = {'file_name': 'scenenn_camera_frame_159.636708_image.jpg'}
            img_metadata = {'file_name': 'scenenn_camera_frame_76.550544_image.jpg'}

            path = img_metadata['file_name']
            img = image.imread(os.path.join(img_path, path))
            img = self.transform(img)


            #method 1:
            # comes with original reference code, demo()
            img = img.expand_dims(0).as_in_context(args.ctx[0])
            output = model.demo(img)
            predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

            #method 2: (prefered)
            # support multiple input image size
            #img = img.expand_dims(0).as_in_context(args.ctx[0])
            #evaluator = MultiEvalModel(model, 53, ctx_list=args.ctx)
            #output = evaluator.parallel_forward(img)
            #predict = mx.nd.squeeze(mx.nd.argmax(output[0], 1)).asnumpy()
            
            #method 3: # a quick version for no cuda
            # but this one is so slow
            # after read source code, we figure out this one
            #img2 = img.as_in_context(args.ctx[0])
            #print(args.ctx)
            #evaluator = MultiEvalModel(model, 53, ctx_list=args.ctx)
            #output = evaluator(img2)
            #predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()


            #method 4:
            # this method can only generate 480 x 480, cropped version
            #img = img.expand_dims(0).as_in_context(args.ctx[0])
            #evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
            #outputs = evaluator(img.astype(args.dtype, copy=False))
            #output = [x[0] for x in outputs]
            #predict = mx.nd.squeeze(mx.nd.argmax(output[0], 1)).asnumpy()


            print('output shape: ')
            print(len(output)) 
            #print(output[0].shape) 
            #print(output)
            #continue
            predicted_categories = list(np.unique(predict))

            
            #print(idx)
            #print(predicted_categories)

            for category in predicted_categories:
                print('Category: ')
                print(category)
                print("category_id: self.classes[int(category)]")
                print(self.classes[int(category)])
                print(self.classes_name[int(category)])
                # TODO: I think the category 0 is not 'banner' as expected... Need to look at the training.
                if category == 0.0: continue
                binary_mask = (np.isin(predict, category) * 1)
                binary_mask = np.asfortranarray(binary_mask).astype('uint8')
                segmentation_rle = coco_mask.encode(binary_mask)
                result = {"image_id": int(idx),
                          "category_id": self.classes[int(category)],
                          "segmentation": segmentation_rle,
                          }
                semantic_segmentation.append(result)
                #print('for category')
                #print(len(result['segmentation']))

            #print('for idx in tbar: result[segmentation]')
            #print(result['segmentation'])
            #print(len(semantic_segmentation))
        self.predictions = semantic_segmentation


if __name__ == "__main__":
    args = parse_args()
    predictor = SemanticSegmentation(args)
    predictor.predict()
    predictor.save_predictions(destination_dir='../../data/predictions', filename=args.sem_seg_result_file)
