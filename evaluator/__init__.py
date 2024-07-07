__reference__ = 'pycocotools' # https://github.com/cocodataset/cocoapi

import json

import torch
import numpy as np
from .coco import COCO
from .cocoeval import COCOeval

from utils.general import x1y1x2y2_to_x1y1wh, scale_coords


class Evaluator:
    def __init__(self, annoFile):
        self.annoGt = COCO(annotation_file=annoFile)
        with open(annoFile, mode='r') as f:
            data = json.load(f)
        self.imageToid = data['imageToid']
        self.annoDt = []

    def accumulate(self, prediction, filename, shape0, shape1):
        if len(prediction):
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            cls_id = prediction[:, 0:1]
            bbox = prediction[:, 1:5]
            conf = prediction[:, 5:6]
            box_x1y1x2y2 = scale_coords(coords=bbox, img0_shape=shape0, 
                                        img1_shape=shape1, scaleup=True)
            box_x1y1wh = x1y1x2y2_to_x1y1wh(box_x1y1x2y2)
            img_id = np.array((self.imageToid[filename],) * len(cls_id))[:, np.newaxis]
            self.annoDt.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    def aggregate(self):
        annoDt = np.concatenate(self.annoDt, axis=0) if len(self.annoDt) else np.array([[0, ] * 7])
        annoEval = COCOeval(self.annoGt, self.annoGt.loadRes(annoDt), iouType='bbox')
        annoEval.evaluate()
        annoEval.accumulate()
        annoEval.summarize()
        self.annoDt.clear()

        cls_score = {}
        for catId in annoEval.params.catIds:
            x = {}
            x['AP'] = self.get_classAP(scores=annoEval.eval, catId=catId)
            x['AP50'] = self.get_classAP(scores=annoEval.eval, catId=catId, iouThr=.50)
            x['AP75'] = self.get_classAP(scores=annoEval.eval, catId=catId, iouThr=.75)
            cls_score[catId] = x
        return annoEval.stats, cls_score

    def get_classAP(self, scores, catId, iouThr=None, areaRng='all', maxDets=100):
        p = scores['params']
        s = scores['precision']
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]

        s = s[..., catId, aind, mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s