"""Load MSCOCO stuff dataset."""
import os
import pickle

import numpy as np
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset
from tqdm import trange
from gluoncv.utils.viz import get_color_pallete


class COCOSemantic(SegmentationDataset):
    """COCO Semantic Segmentation Dataset for the Panoptic Segmentation task.

    Parameters
    ----------
    root : string
        Path to COCO dataset folder. Default is './mscoco'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image

    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.COCOSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    #CAT_LIST = [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
    #            149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
    #            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    #NUM_CLASS = 53


    #CAT_LIST = [92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
    #            149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 183, 184, 185, 186,
    #            187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    #NUM_CLASS = 54


    CAT_LIST = [0, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138, 141, 144, 145, 147, 148,
                149, 151, 154, 155, 156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186,
                187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200]

    NUM_CLASS = 54




    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/coco'),
                 split='train', mode=None, transform=None, **kwargs):
        super(COCOSemantic, self).__init__(root, split, mode, transform, **kwargs)
        # lazy import pycocotools
        from pycocotools.coco import COCO
        from pycocotools import mask
        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/stuff_train2017.json')
            ids_file = os.path.join(root, 'annotations/sem_train_ids_54_0.mx')
            self.root = os.path.join(root, 'train2017')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/stuff_val2017.json')
            ids_file = os.path.join(root, 'annotations/sem_val_ids_54_0.mx')
            self.root = os.path.join(root, 'val2017')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        
        """
        # only used for generating mask ground truth
        outdir = 'gt_outdir'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outname = 'gt_mask_' + str(index) + '_' + str(img_id) + '.png'
        #print(mask)
        mask_out = get_color_pallete(mask.asnumpy(), 'coco')
        mask_out.save(os.path.join(outdir, outname))
        print(str(img_id) + ' is saved.')
        """

        return img, mask

    def __len__(self):
        return len(self.ids)

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            #print(instance)
            #print(instance['segmentation'][0])
            #--------------------------------------------------------------------
            # right one for this version (1atest version)
            rle = self.coco.annToRLE(instance)
            m = coco_mask.decode(rle)
            #--------------------------------------------------------------------
            # For original github version.
            #m = coco_mask.decode(instance['segmentation'])
            #--------------------------------------------------------------------
            #print('decode is successful')
            # Here is anoter recommendation from gluoncv/data/mscoco/segmentation.py
            #rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            #m = coco_mask.decode(rle)
            #--------------------------------------------------------------------

            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while." + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            #print(len(cocotarget))
            img_metadata = self.coco.loadImgs(img_id)[0]
            #print(len(img_metadata))
            #print(img_metadata['height'])
            #print(img_metadata['width'])
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    @property
    def classes(self):
        """Category names."""
        #return ('banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood',
        #        'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
        #        'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs',
        #        'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
        #        'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged',
        #        'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
        #        'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
        #        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged')


        #return ('banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood',
        #        'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
        #        'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs',
        #        'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
        #        'window-blind', 'window-other', 'other', 'tree-merged', 'fence-merged', 'ceiling-merged',
        #        'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
        #        'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
        #        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged')


        return ('thing', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood',
                'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
                'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs',
                'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
                'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged',
                'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
                'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
                'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged')

