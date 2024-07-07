import os
from pathlib import Path

import cv2
import torch
import numpy as np

from dataloader.transforms import TestTransform
from model import build_model
from utils.args import build_parser
from utils.general import (GENERAL_MEAN, GENERAL_STD, LoadImages, 
                           AverageMeter, non_max_suppression, scale_coords)
from utils.torch_utils import time_sync
from utils.viz import visualize_prediction

ROOT = Path(__file__).resolve().parents[0]


@torch.no_grad()
def test(loader, model, class_list, device, **kwargs):
    conf_thres = kwargs.get('conf_thres')
    nms_thres = kwargs.get('nms_thres')
    result_dir = kwargs.get('result_dir')
    process_time = AverageMeter('Time', ':5.3f')

    model.eval()
    for fpath, im, im0, s in loader:
        im = im.to(device, non_blocking=True)

        t1 = time_sync()
        preds = model(im.unsqueeze(0).half())
        preds = non_max_suppression(preds, conf_thres, nms_thres, multi_label=False)
        preds = preds[0].cpu().numpy()
        preds[:, 1:5] = scale_coords(coords=preds[:, 1:5], img0_shape=im0.shape[:2], 
                                        img1_shape=im.shape[1:], scaleup=True)
        process_time.update(time_sync() - t1)

        if len(preds):
            for c in np.unique(preds[:, 0]):
                n = (preds[:, 0] == c).sum()
                s += f"{n} {class_list[int(c)]}{'s' if n > 1 else ''}, "
            s = s.rstrip(', ')
        else:
            s += 'no detection'
        
        print(f"{s.rstrip(', ')} (Time: {process_time.val * 1e+3:.3f} ms)")
        visualize_prediction(image=im0, prediction=preds, class_list=class_list)
        cv2.imwrite(str(result_dir / f'{fpath.split(os.sep)[-1]}' ), im0)
    del im, preds
    torch.cuda.empty_cache()


def main(opt, device):
    assert opt.test_dir, f'input "--test-dir" argument, got "{opt.test_dir}"'
    assert Path(opt.test_dir).is_dir(), f'given "{opt.test_dir}" directory not exists'

    ckpt = torch.load(opt.ckpt_path)
    arch = ckpt.get('arch')
    img_size = ckpt.get('img_size')
    class_list = ckpt.get('class_list')
    model_state = ckpt.get('model_state')
    device = torch.device(device)
    conf_thres = 0.20
    nms_thres = 0.45

    transformer = TestTransform(input_size=img_size, mean=GENERAL_MEAN, std=GENERAL_STD)
    loader = LoadImages(path=opt.test_dir, transforms=transformer)
    model = build_model(arch_name=arch, num_classes=len(class_list))
    model.load_state_dict(state_dict=model_state, strict=True)
    model = model.to(device)
    model.half()
    
    x = torch.randn((1, 3, img_size, img_size), device=device)
    model(x.half())

    test(loader=loader, model=model, class_list=class_list, device=device, 
         conf_thres=conf_thres, nms_thres=nms_thres, result_dir=opt.result_dir)
    

if __name__ == "__main__":
    try:
        opt, _ = build_parser(root_dir=ROOT)
        main(opt=opt, device='cuda')
    except Exception as e:
        raise RuntimeError(e)