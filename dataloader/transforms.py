import sys
import math
from pathlib import Path

import cv2
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.general import xcycwhn_to_x1y1x2y2, x1y1x2y2_to_xcycwhn


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.01, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def mixup(inputs, targets, alpha):
    if np.random.randint(2) and alpha > 0.0:
        m = np.random.beta(alpha, alpha)
        permutation = np.random.permutation(inputs.shape[0])
        inputs = m * inputs + (1.0 - m) * inputs[permutation]
        targets = np.concatenate((targets, targets[permutation]), axis=1)
    return inputs, targets


class TrainTransform:
    """augmentation class for model training
    """
    def __init__(self, input_size, mean, std, **kwargs):
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.degrees = kwargs.get('degrees', 0.0)
        self.translate = kwargs.get('translate', 0.1)
        self.scale = kwargs.get('scale', 0.5)
        self.shear = kwargs.get('shear', 0.0)
        self.perspective = kwargs.get('perspective', 0.0)
        self.h_gain = kwargs.get('h_gain', 0.015)
        self.s_gain = kwargs.get('s_gain', 0.7)
        self.v_gain = kwargs.get('v_gain', 0.4)
        dataset = kwargs.get('dataset', None)
        mosaic = kwargs.get('mosaic', False)
        
        if mosaic:
            assert dataset is not None, f'Mosaic requires dataset, but got {dataset}'
        border = (-input_size // 2, -input_size // 2) if mosaic else (0, 0)

        self.transforms = Compose([
            Mosaic(dataset, new_shape=self.input_size, border=border) \
                if mosaic else LetterBox(new_shape=(self.input_size, self.input_size)),
            ToXminYminXmaxYmax(),
            ToAbsoluteCoords(),
            RandomPerspective(degrees=self.degrees, translate=self.translate, 
                              scale=self.scale, shear=self.shear, perspective=self.perspective, 
                              border=border),
            HorizontalFlip(),
            ToPercentCoords(),
            ToXcenYcenWH(),
            AugmentHSV(h_gain=self.h_gain, s_gain=self.s_gain, v_gain=self.v_gain),
            Normalize(mean=self.mean, std=self.std),
            ToTensor()
        ])
    
    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.transforms(image, boxes, labels)
        return image, boxes, labels

    def close_mosaic(self):
        self.__init__(input_size=self.input_size, mean=self.mean, std=self.std,
                      degrees=self.degrees, translate=self.translate, scale=self.scale, 
                      shear=self.shear, perspective=self.perspective, mosaic=False)
        

class ValidTransform:
    """augmentation class for model evaluation
    """
    def __init__(self, input_size, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            LetterBox(new_shape=(input_size, input_size), scaleup=True),
            Normalize(mean=mean, std=std),
            ToTensor()
        ])
    
    def __call__(self, image, boxes, labels):
        image, boxes, labels = self.tfs(image, boxes, labels)
        return image, boxes, labels


class TestTransform:
    """augmentation class for model evaluation
    """
    def __init__(self, input_size, mean, std):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        self.tfs = Compose([
            LetterBox(new_shape=(input_size, input_size), scaleup=True),
            Normalize(mean=mean, std=std),
            ToTensor()
        ])
    
    def __call__(self, image):
        image, *_ = self.tfs(image, None, None)
        return image


class Compose:
    """compositor for augmentation combination
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for tf in self.transforms:
            image, boxes, labels = tf(image, boxes, labels)
        return image, boxes, labels


class Mosaic:
    def __init__(self, dataset, new_shape=640, border=None, fill=114):
        self.dataset = dataset
        self.new_shape = new_shape
        self.border = (-new_shape // 2, -new_shape // 2) if border is None else border
        self.fill = fill
        self.indices = range(len(self.dataset))

    def __call__(self, image, boxes, labels):
        image4, boxes4, labels4 = self._mosaic4(image=image, boxes=boxes, labels=labels)
        return image4, boxes4, labels4

    def _mosaic4(self, image, boxes, labels):
        boxes4, labels4 = [], []
        s = self.new_shape
        image = scale_image(image=image, img0_shape=image.shape[1::-1], img1_shape=(s, s))
        
        h, w, c = image.shape
        yc, xc = (int(np.random.uniform(-x, 2 * s + x)) for x in self.border)
        indices = np.random.choice(self.indices, size=3)
        image4 = np.full((s * 2, s * 2, c), self.fill, dtype=np.uint8)
        x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc # for mosaic image
        x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h # for each image

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b] # top-left
        padw = x1a - x1b
        padh = y1a - y1b
        boxes = self._transform_boxes(boxes, w, h, padw, padh)
        boxes4.append(boxes)
        labels4.append(labels)
        
        for i, index in enumerate(indices):
            image, labels, *_ = self.dataset.get_GT(index)
            image = scale_image(image=image, img0_shape=image.shape[1::-1], img1_shape=(s, s))
            boxes, labels = labels[:, 1:], labels[:, 0]
            h, w = image.shape[:2]

            if i == 0: # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 1: # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 2: # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            boxes = self._transform_boxes(boxes, w, h, padw, padh)
            boxes4.append(boxes)
            labels4.append(labels)

        boxes4 = np.concatenate(boxes4, axis=0)
        labels4 = np.concatenate(labels4, axis=0)
        boxes4, labels4 = self._remove_zero_area_boxes(boxes=boxes4, labels=labels4)
        return image4, boxes4, labels4

    def _transform_boxes(self, x, w, h, padw, padh):
        x = xcycwhn_to_x1y1x2y2(x, w=w, h=h, padw=padw, padh=padh)
        x = x1y1x2y2_to_xcycwhn(x, w=self.new_shape * 2, h=self.new_shape * 2, clip=True)
        return x

    def _remove_zero_area_boxes(self, boxes, labels):
        areas = boxes[:, 2] * boxes[:, 3]
        good = areas > 0
        return boxes[good], labels[good]
    

class RandomPerspective:
    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, 
                 perspective=0.0, border=(0,0), fill=(114, 114, 114)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.fill = fill
    
    def __call__(self, image, boxes, labels):
        self.height = image.shape[0] + self.border[0] * 2
        self.width = image.shape[1] + self.border[0] * 2
        new_image, M, scale = self.affine_transform(image)
        new_boxes, new_labels = self.apply_bboxes(boxes=boxes, labels=labels, M=M)
        i = box_candidates(box1=boxes.T * scale, box2=new_boxes.T, area_thr=0.1)
        return new_image, new_boxes[i], new_labels[i]
        
    def affine_transform(self, image):
        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -image.shape[1] / 2
        C[1, 2] = -image.shape[0] / 2
        
        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = np.random.uniform(-self.perspective, self.perspective)
        P[2, 1] = np.random.uniform(-self.perspective, self.perspective)
        
        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = np.random.uniform(-self.degrees, self.degrees)
        s = np.random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        
        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        S[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        
        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.width
        T[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.height
    
        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                image = cv2.warpPerspective(image, M, dsize=(self.width, self.height), borderValue=self.fill)
            else:
                image = cv2.warpAffine(image, M[:2], dsize=(self.width, self.height), borderValue=self.fill)
        return image, M, s

    def apply_bboxes(self, boxes, labels, M):
        if -1 in labels:
            return boxes, labels
        
        n = len(boxes)
        xy = np.ones((n * 4, 3), dtype=boxes.dtype)
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        boxes[:, [0,2]] = boxes[:, [0,2]].clip(min=0, max=self.width)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(min=0, max=self.height)
        return boxes, labels


class ToAbsoluteCoords:
    def __call__(self, image, boxes, labels):
        height, width = image.shape[:2]
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes, labels):
        height, width = image.shape[:2]
        boxes[:, [0, 2]] /= width
        boxes[:, [1, 3]] /= height
        return image, boxes, labels


class ToXminYminXmaxYmax:
    def __call__(self, image, boxes, labels):
        x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
        x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
        boxes = np.concatenate((x1y1, x2y2), axis=1).clip(min=0, max=1)
        return image, boxes, labels


class ToXcenYcenWH:
    def __call__(self, image, boxes, labels):
        wh = boxes[:, 2:] - boxes[:, :2]
        xcyc = boxes[:, :2] + wh / 2
        boxes = np.concatenate((xcyc, wh), axis=1)
        return image, boxes, labels


class AugmentHSV:
    def __init__(self, h_gain=0.5, s_gain=0.5, v_gain=0.5):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, image, boxes, labels):
        if self.h_gain or self.s_gain or self.v_gain:
            r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(image.dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(image.dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(image.dtype)
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        return image, boxes, labels


class HorizontalFlip:
    def __call__(self, image, boxes, labels):
        width = image.shape[1]
        if np.random.randint(2):
            image = image[:, ::-1, :]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels


def scale_image(image, img0_shape, img1_shape, scaleup=True):
    r = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    new_shape = int(round(img0_shape[0] * r)), int(round(img0_shape[1] * r)) # width, height
    if img0_shape != new_shape: # resize
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
    return image


class LetterBox:
    def __init__(self, new_shape=(640, 640), fill=(114, 114, 114), border=(8, 8), scaleup=True):
        self.new_shape = new_shape # (w, h)
        self.scaleup = scaleup
        self.fill = fill
        self.img_shape = (new_shape[0] - 2 * border[0], new_shape[1] - 2 * border[1]) # (w, h)

    def __call__(self, image, boxes, labels):
        image = scale_image(image=image, img0_shape=image.shape[1::-1], 
                            img1_shape=self.img_shape, scaleup=self.scaleup)
        shape = image.shape[:2]
        dw, dh = (self.new_shape[0] - shape[1]) / 2, (self.new_shape[1] - shape[0]) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)

        if boxes is not None:
            boxes[:, :2] = (boxes[:, :2] * (shape[1], shape[0]) + (left, top))
            boxes[:, :2] /= (self.new_shape)
            boxes[:, 2:] /= (self.new_shape[0] / shape[1], self.new_shape[1] / shape[0])
        return image, boxes, labels


class Normalize:
    """normalize a tensor image with mean and standard deviation.
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes, labels):
        if not isinstance(image.dtype, np.float32):
            image = image.astype(np.float32)
        image /= 255
        image -= self.mean
        image /= self.std
        return image, boxes, labels


class ToTensor:
    def __call__(self, image, boxes, labels):
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        return image, boxes, labels
