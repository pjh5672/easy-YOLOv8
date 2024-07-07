from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm

from dataloader import build_dataloader
from model import build_model
from evaluator import Evaluator

from utils.args import build_parser
from utils.general import (TQDM_BAR_FORMAT, print_args,
                           non_max_suppression, report_per_class)
from utils.viz import visualize_prediction, to_image

ROOT = Path(__file__).resolve().parents[0]


@torch.no_grad()
def validate(loader, model, evaluator, device, **kwargs):
    s = ('%15s' + '%14s' * 6) % ('Validation', 'AP@50:95', 'AP@50', 
                                 'AP@75', 'AP@S', 'AP@M', 'AP@L')
    pbar = tqdm(enumerate(loader), desc=s, total=len(loader), bar_format=TQDM_BAR_FORMAT)
    conf_thres = kwargs.get('conf_thres')
    nms_thres = kwargs.get('nms_thres')
    class_list = kwargs.get('class_list')
    check_images, check_preds, check_results = [], [], []

    model.eval()
    for _, batch in pbar:
        images, filenames, shapes = batch[0].to(device, non_blocking=True), batch[2], batch[3]
        preds = model(images)
        preds = non_max_suppression(preds, conf_thres, nms_thres, multi_label=True)
        
        for j in range(len(filenames)):
            evaluator.accumulate(prediction=preds[j], filename=filenames[j], 
                                 shape0=shapes[j][:2], shape1=images.shape[2:])
    
    #         if len(check_images) < 12:
    #             check_images.append(images[j])
    #             check_preds.append(preds[j])
            
    # for k in range(len(check_images)):
    #     check_image = to_image(check_images[k]).copy()
    #     check_pred = check_preds[k].cpu().numpy().copy()
    #     check_result = visualize_prediction(image=check_image, prediction=check_pred, class_list=class_list)
    #     check_results.append(check_result)
    # concat_result = np.concatenate(check_results, axis=1)
    # cv2.imwrite(f'assets/val_results.jpg', concat_result)
    
    summ_result, class_result = evaluator.aggregate()
    print(('%15s' + '%14.4g' * 6) % ('Result', *summ_result[:6]))
    del images, preds
    torch.cuda.empty_cache()
    return summ_result, class_result


def save_result(keys, vals, save_dir):
    result_csv = save_dir / 'result.csv'
    keys = tuple(x.strip() for x in keys)
    n = len(keys)
    s = '' if result_csv.exists() else (('%14s,' * n % keys).rstrip(',') + '\n')
    with open(result_csv, 'a') as f:
        f.write(s + ('%14.5g,' * n % vals).rstrip(',') + '\n')


def main(opt, parser, device):
    conf_thres = getattr(opt, 'conf_thres')
    nms_thres = getattr(opt, 'nms_thres')
    ckpt = torch.load(opt.ckpt_path)
    dataset = ckpt.get('dataset')
    arch = ckpt.get('arch')
    img_size = ckpt.get('img_size')
    class_list = ckpt.get('class_list')
    model_state = ckpt.get('model_state')
    device = torch.device(device)
    opt.dataset = dataset
    opt.img_size = img_size

    opt = parser.change_dataset(to=dataset)
    print_args(opt, include_keys=('project', 'val_file', 'dataset', 'img_size',
                                  'batch_size', 'conf_thres', 'nms_thres'))
    _, val_loader = build_dataloader(opt=opt)
    model = build_model(arch_name=arch, num_classes=len(class_list))
    model.load_state_dict(state_dict=model_state, strict=True)
    model.to(device)
    evaluator = Evaluator(annoFile=opt.val_file)

    _, class_result = validate(loader=val_loader, model=model, 
                               evaluator=evaluator, device=device, 
                               conf_thres=conf_thres, nms_thres=nms_thres, 
                               img_size=img_size, class_list=class_list)
    
    report_per_class(save_dir=opt.project_dir, src=class_result, filename='val_eval_per_class.csv')


if __name__ == "__main__":
    try:
        opt, parser = build_parser(root_dir=ROOT)
        main(opt=opt, parser=parser, device='cuda')
    except Exception as e:
        raise RuntimeError(e)