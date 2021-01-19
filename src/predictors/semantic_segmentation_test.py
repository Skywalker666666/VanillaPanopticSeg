import os
from tqdm import tqdm
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete

from gluoncv.utils.parallel import *

from gluoncv import model_zoo

import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from src.dataloaders.cocosemantic import COCOSemantic

from trainers.PSPNet_trainer import parse_args

def test(args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    if args.eval:
        #testset = get_segmentation_dataset(
        #    args.dataset, split='val', mode='testval', transform=input_transform)
        #testset = COCOSemantic(split='val', mode='testval', transform=input_transform)
        testset = COCOSemantic(split='val', mode='testval', transform=input_transform)

        total_inter, total_union, total_correct, total_label = \
            np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    else:
        #testset = get_segmentation_dataset(
        #    args.dataset, split='test', mode='test', transform=input_transform)
        testset = COCOSemantic(split='test', mode='testval', transform=input_transform)

    print("testset.num_class: ")
    print(testset.num_class)

    #test_data = gluon.data.DataLoader(
    #    testset, args.test_batch_size, shuffle=False, last_batch='keep',
    #    batchify_fn=ms_batchify_fn, num_workers=args.workers)
    # what is this one used for? ms_batchify_fn     """Multi-size batchfy function"""

    test_data = gluon.data.DataLoader(testset, args.test_batch_size,
                                               last_batch='keep', num_workers=args.workers)




    # create network
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        #model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
        #                               backbone=args.backbone, norm_layer=args.norm_layer,
        #                               norm_kwargs=args.norm_kwargs, aux=args.aux,
        #                               base_size=args.base_size, crop_size=args.crop_size)

        model = model_zoo.PSPNet(nclass=testset.num_class, 
                                 backbone=args.backbone, 
                                 aux=args.aux, 
                                 pretrained_base=False, 
                                 #norm_layer=args.norm_layer, 
                                 #norm_kwargs=args.norm_kwargs, 
                                 #crop_size=args.crop_size, 
                                 ctx=args.ctx)
        model.cast(args.dtype)
        # load pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            #model.load_parameters(args.resume, ctx=args.ctx)
            model.load_parameters(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))

    print(model)

    evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)
    # this is used in the val, not test
    #evaluator2 = DataParallelModel(SegEvalModel(model), args.ctx)
    
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        if args.eval:
            #predicts = [ pred[0] for pred in evaluator.parallel_forward(data)]
            #targets = [target.as_in_context(predicts[0].context) \
            #           for target in dsts]
            #predicts = evaluator(data.astype(args.dtype, copy=False))
            #predicts = [x[0] for x in predicts]

            # method 3
            predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
            #predicts = [[x for x in predicts1]]

            print(len(predicts))
            print(len(predicts[0]))
            print(len(predicts[0][0]))
            #predicts = mx.nd.expand_dims(mx.nd.array(predicts1), axis=0)
            predicts2 = mx.nd.expand_dims(predicts[0], axis=0)
            predicts3 = [predicts2.as_in_context(predicts[0].context)]

            targets = mx.gluon.utils.split_and_load(dsts, args.ctx, even_split=False)

            print("i: ")
            print(i)
            #print(predicts2)
            #print(len(targets[0]))
            #print(targets) 
            #bbb = np.array([[[1],[2],[3]],[[7],[6],[0]]])
            #bbb = np.array(predicts)
            targets2 = targets[0]
            print(predicts2.shape)
            #print(predicts3)
            print(targets2.shape)
            #print(targets2)
            metric.update(targets, predicts3)
            pixAcc, mIoU = metric.get()
            tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            im_paths = dsts
            print("im_paths: ")
            print(im_paths)
            predicts = evaluator.parallel_forward(data)
            for predict, impath in zip(predicts, im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                    testset.pred_offset
                mask = get_color_pallete(predict, args.dataset)
                print("outname: ")
                print(os.path.splitext(impath)[0])
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))


#def validation(self, epoch):
#    # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
#    self.metric.reset()
#    tbar = tqdm(self.eval_data)
#    for i, (data, target) in enumerate(tbar):
#        outputs = self.evaluator(data.astype(args.dtype, copy=False))
#        outputs = [x[0] for x in outputs]
#        targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
#        self.metric.update(targets, outputs)
#        pixAcc, mIoU = self.metric.get()
#        tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f' % \
#                             (epoch, pixAcc, mIoU))
#        self.sw.add_scalar(tag='Pixel Accuray (on valset)', value=pixAcc, global_step=epoch)
#        self.sw.add_scalar(tag='mIoU (on valset)', value=mIoU, global_step=epoch)
#        mx.nd.waitall()



if __name__ == "__main__":
    args = parse_args()
    args.test_batch_size = args.ngpus
    print('Testing model: ', args.resume)
    test(args)
