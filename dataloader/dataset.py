import os
import sys
import json
from pathlib import Path
from multiprocessing.pool import ThreadPool

import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import (GENERAL_MEAN, GENERAL_STD, IMG_FORMATS,
                           NUM_THREADS, TQDM_BAR_FORMAT, xcycwh_to_x1y1wh)


class Dataset:
    def __init__(self, opt, phase):
        self.phase = phase
        self.transform = None
        opt.mean, opt.std = GENERAL_MEAN, GENERAL_STD
        self.class_list = getattr(opt, 'class_list')
        self.mAP_evalfile = getattr(opt, 'val_file')

        self.image_paths = []
        image_dir = opt.train_dir if phase.lower() == 'train' else opt.val_dir
        image_dir = Path(opt.dataroot_dir) / image_dir
        for fpath in sorted(glob.glob(str(image_dir / '*'))):
            if fpath.lower().endswith(IMG_FORMATS):
                self.image_paths.append(fpath)

        self.label_paths = self.replace_image2label_path(image_paths=self.image_paths)
        self.generate_no_label(label_paths=self.label_paths)
        self.num_classes = len(self.class_list)
        if phase.lower() == 'val' and not Path(self.mAP_evalfile).is_file():
            self.generate_mAP_evalfile()

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, index):
        image, label, filename, shape = self.get_GT(index)
        image, boxes, cls = self.transform(image=image, boxes=label[:, 1:5], labels=label[:, 0])
        label = np.concatenate((cls[:, None], boxes), axis=1)
        if not len(label):
            label = np.array([[-1, 0., 0., 0., 0.]], dtype=np.float32)
        return image, label, filename, shape
    
    def get_GT(self, index):
        image, filename, shape = self.get_image(index)
        label = self.get_label(index)
        label = self.check_no_label(label)
        return image, label, filename, shape
    
    def check_no_label(self, label):
        if not len(label):
            label = np.array([[-1, 0., 0., 0., 0.]], dtype=np.float32)
        return label
    
    def get_image(self, index):
        filename = self.image_paths[index].split(os.sep)[-1]
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        shape = image.shape
        return image, filename, shape

    def get_label(self, index):
        with open(self.label_paths[index], mode='r') as f:
            item = [x.split() for x in f.read().splitlines()]
        lbl = np.array(item, dtype=np.float32)
        _, i = np.unique(lbl, axis=0, return_index=True)
        if len(i) < len(lbl):
            lbl = lbl[i]
        return lbl
    
    def replace_image2label_path(self, image_paths):
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in image_paths]
    
    def generate_no_label(self, label_paths):
        for label_path in label_paths:
            if not os.path.isfile(label_path):
                print(f'no such label file : {label_path}, generate empty label file')
                f = open(str(label_path), mode='w')
                f.close()

    @staticmethod
    def collate_fn(batch):
        images, labels, filenames, shapes = zip(*batch)

        batch_size = len(labels)
        max_len = len(max(labels, key=len))
        new_labels = np.zeros((batch_size, max_len, 5), dtype=np.float32)
        new_labels[..., 0] = -1
        for i in range(batch_size):
            label = labels[i]
            if -1 not in label[:, 0]:
                new_labels[i, :len(label)] = label
        return torch.stack(images, dim=0), new_labels, filenames, shapes

    def generate_mAP_evalfile(self):
        mAP_eval_format = {}
        mAP_eval_format['imageToid'] = {}
        mAP_eval_format['images'] = []
        mAP_eval_format['annotations'] = []
        mAP_eval_format['categories'] = []
        
        lb_id = 0
        results = ThreadPool(NUM_THREADS).imap(self.get_GT, range(len(self)))
        for i, (_, lb, fn, sh) in tqdm(enumerate(results), desc='Generating annotaion files for evaluation...', 
                                    total=len(self), bar_format=TQDM_BAR_FORMAT):
            img_h, img_w = sh[:2]
            mAP_eval_format['imageToid'][fn] = i
            mAP_eval_format['images'].append({'id':i, 'width':img_w, 'height':img_h})
            
            lb[:, 1:5] = xcycwh_to_x1y1wh(lb[:, 1:5])
            lb[:, [1,3]] *= img_w
            lb[:, [2,4]] *= img_h
            for j in range(len(lb)):
                x = {}
                x['image_id'] = i
                x['id'] = lb_id
                x['bbox'] = [round(item, 2) for item in lb[j][1:5].tolist()]       
                x['area'] = round((x['bbox'][2] * x['bbox'][3]), 2)
                x['iscrowd'] = 0
                x['category_id'] = int(lb[j][0])
                x['segmentation'] = []
                mAP_eval_format['annotations'].append(x)
                lb_id += 1

        for idx, cls in self.class_list.items():
            mAP_eval_format['categories'].append({'id':idx, 'supercategory':'', 'name':cls})

        with open(self.mAP_evalfile, 'w') as file:
            json.dump(mAP_eval_format, file)


if __name__ == "__main__":
    import sys

    import cv2
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))

    from transforms import TrainTransform, ValidTransform
    from utils.args import ConfigParser
    from utils.general import init_seeds
    from utils.viz import visualize_dataset

    # init_seeds(seed=0)

    dataset = 'toy'
    data_dir = ROOT / 'cfg'
    opt = ConfigParser(data_dir=data_dir, dataset=dataset).update()
    train_dataset = Dataset(opt=opt, phase='train')
    val_dataset = Dataset(opt=opt, phase='val')

    # for i in range(len(train_dataset)):
    #     image, label, filename, shape = train_dataset.get_GT(i)
    #     print(filename)
    #     print(image.shape == shape, shape)
    #     print(label)
    
    mosaic = True
    train_transform = TrainTransform(input_size=640, mean=opt.mean, std=opt.std,
                                     dataset=train_dataset, mosaic=mosaic)
    train_dataset.transform = train_transform
    val_transform = ValidTransform(input_size=640, mean=opt.mean, std=opt.std)
    val_dataset.transform = val_transform
    # image, label, filename, shape = val_dataset[0]
    # cv2.imwrite(str(ROOT / 'assets' / f'{dataset}_letterbox.jpg' ), image)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, 
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                            collate_fn=train_dataset.collate_fn)
    
    for idx, batch in enumerate(train_loader):
        images, labels, fnames, shapes = batch
        print(idx, '-', images.shape, shapes)
        print(labels)
        print(labels.shape)
        print(torch.min(images), torch.max(images))
        if idx == 0:
            break

    vis_image = visualize_dataset(img_loader=train_loader, 
                                  class_list=opt.class_list, 
                                  mean=opt.mean, std=opt.std)
    cv2.imwrite(str(ROOT / 'assets' / f'{dataset}_train.jpg' ), vis_image)
    # cv2.imwrite(str(ROOT / 'assets' / f'{dataset}_val.jpg' ), vis_image)