import os
import time
import math
import glob
import random
import inspect
from enum import Enum
from pathlib import Path

import torch
import torchvision
import numpy as np
from PIL import Image
from numba import jit

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
TQDM_BAR_FORMAT = '{l_bar}{bar:12}{r_bar}'
GENERAL_MEAN = (0.485, 0.456, 0.406)
GENERAL_STD = (0.229, 0.224, 0.225)
IMG_FORMATS = 'jpeg', 'jpg', 'png'  # include image suffixes


def non_max_suppression(preds, conf_thres, nms_thres, multi_label=False, agnostic=False, max_det=300):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= nms_thres <= 1, f"Invalid IoU {nms_thres}, valid values are between 0.0 and 1.0"

    device = preds.device
    bs = preds.shape[0]  # batch size
    nc = preds.shape[1] - 4  # number of classes
    xc = preds[:, 4:].amax(1) > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 2.0 + 0.05 * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    preds = preds.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    preds[..., :4] = xcycwhn_to_x1y1x2y2(preds[..., :4], w=1, h=1)  # xywh to xyxy
    
    t = time.time()
    output = [torch.zeros((0, 6), device=device)] * bs
    for xi, x in enumerate(preds):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue
        
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls = x.split((4, nc), 1)
                
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((j[:, None].float(), box[i], x[i, 4 + j, None], ), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((j.float(), box, conf), 1)[conf.view(-1) > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 5].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 0:1] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, 1:5] + c, x[:, 5]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, nms_thres)  # NMS
        i = i[:max_det]  # limit detections
        
        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            print(f"WARNING ! NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    
    return output

def print_args(args = None, show_file=True, include_keys=(), exclude_keys=()):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, *_ = inspect.getframeinfo(x)
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '')
    if len(include_keys):
        s += ', '.join(f'{k}={v}' for k, v in args.__dict__.items() if k in include_keys)
    if len(exclude_keys):
        s += ', '.join(f'{k}={v}' for k, v in args.__dict__.items() if k not in exclude_keys)
    print(s)
    

class LoadImages:
    def __init__(self, path, input_size=640, transforms=None):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        self.files = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        self.transforms = transforms  # optional
        self.input_size = input_size
        
    def __len__(self): return len(self.files)

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == len(self.files):
            raise StopIteration

        path = self.files[self.count]
        im0 = np.array(Image.open(path).convert('RGB'))
            
        s = f'[{self.count+1}/{len(self.files)}] {path}: '
        
        if self.transforms:
            im = self.transforms(image=im0)  # transforms
        else:
            im = im0

        self.count += 1
        return path, im, im0, s
    
def init_seeds(seed=0, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = False
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def scale_coords(coords, img0_shape, img1_shape, border=(8, 8), scaleup=True):
    unborder_shape = (img1_shape[0] - 2 * border[0], img1_shape[1] - 2 * border[1]) # (w, h)
    gain = min(unborder_shape[0] / img0_shape[0], unborder_shape[1] / img0_shape[1])  # gain  = old / new
    if not scaleup:
        gain = min(gain, 1.0)
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x paddingprocess_nms
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_boxes(coords, img0_shape)
    return coords

def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def x1y1x2y2_to_x1y1wh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] # top left x
    y[..., 1] = x[..., 1] # top left y
    y[..., 2] = x[..., 2] - x[..., 0] # width
    y[..., 3] = x[..., 3] - x[..., 1] # height
    return y

def xcycwh_to_x1y1wh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2 # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2 # top left y
    y[..., 2] = x[..., 2] # width
    y[..., 3] = x[..., 3] # height
    return y

def x1y1x2y2_to_xcycwhn(x, w=640, h=640, clip=False, eps=0.0):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right."""
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y

def xcycwhn_to_x1y1x2y2(x, w=640, h=640, padw=0, padh=0):
    """Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.

    Input shapes are box1(1,4) to box2(n,4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} : {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} : {avg' + self.fmt + '}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} : {sum' + self.fmt + '}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} : {count' + self.fmt + '}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        return fmtstr.format(**self.__dict__)


def report_per_class(save_dir, src, filename='eval_per_class.csv'):
    keys = ('Class Index', 'AP@50:95', 'AP@50', 'AP@75')
    c, ap, ap50, ap75 = 0, 0, 0, 0

    with open(save_dir / filename, 'w') as f:
        f.write(('%14s,' * len(keys) % keys).rstrip(',') + '\n')
        
        for k, v in src.items():
            c += 1
            ap += v['AP']
            ap50 += v['AP50']
            ap75 += v['AP75']

            f.write((('%14s,' + '%14.4g,' * (len(keys) - 1)) % 
                    (k, v['AP'], v['AP50'], v['AP75'])).rstrip(',') + '\n')
        f.write((('%14s,' + '%14.4g,' * (len(keys) - 1)) % 
                    ('Total', ap / c, ap50 / c, ap75 / c)).rstrip(',') + '\n')