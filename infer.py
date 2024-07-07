from pathlib import Path

import cv2
import torch
import numpy as np

from dataloader.transforms import TestTransform
from model import build_model
from utils.args import build_parser
from utils.general import (GENERAL_MEAN, GENERAL_STD, AverageMeter,
                           non_max_suppression, scale_coords)
from utils.torch_utils import time_sync
from utils.viz import visualize_prediction

ROOT = Path(__file__).resolve().parents[0]


@torch.no_grad()
def inference(image, transform, model, device, conf_thres, nms_thres):
    im = transform(image).to(device, non_blocking=True)
    preds = model(im.unsqueeze(0).half())
    preds = non_max_suppression(preds, conf_thres, nms_thres, multi_label=False, agnostic=True)
    preds = preds[0].cpu().numpy()
    preds[:, 1:5] = scale_coords(coords=preds[:, 1:5], img0_shape=image.shape[:2], 
                                img1_shape=im.shape[1:], scaleup=True)
    return preds


def run(video_path, transform, model, class_list, device, **kwargs):
    conf_thres = kwargs.get('conf_thres')
    nms_thres = kwargs.get('nms_thres')
    result_dir = kwargs.get('result_dir')

    vid_cap = cv2.VideoCapture(video_path)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_file = result_dir / Path(video_path).name
    # force *.mp4 suffix on results videos
    vid_writer = cv2.VideoWriter(str(save_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h))

    process_time = AverageMeter('Time', ':5.3f')
    model.eval()
    while (vid_cap.isOpened()):
        ret, frame = vid_cap.read()

        if ret:
            t1 = time_sync()
            preds = inference(image=frame, transform=transform, model=model, device=device,
                              conf_thres=conf_thres, nms_thres=nms_thres)
            process_time.update(time_sync() - t1)
            text= f"{(process_time.val) * 1e+3:.1f}ms/frame"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (40, 40, 40), 2)

            if len(preds):
                visualize_prediction(image=frame, prediction=preds, class_list=class_list)[..., ::-1]
            
            ### only for road-traffic ###
            line_x, line_y = int(vid_w/2), int(vid_h/2)
            cv2.line(frame, (line_x, line_y), (line_x, vid_h), (40, 200, 40), 3)
            cv2.line(frame, (0, line_y), (vid_w, line_y), (40, 200, 40), 3)

            l_num, r_num = 0, 0
            if len(preds):
                center = (preds[:, 1:3] + preds[:, 3:5]) / 2 # x1y1x2y2 -> xcyc format 
                l_num = ((center[:, 0] < line_x) & (center[:, 1] > line_y)).sum()
                r_num = ((center[:, 0] > line_x) & (center[:, 1] > line_y)).sum()
            s = f'Left side: {l_num}   Right side: {r_num}'
            cv2.putText(frame, s, (int(vid_w/2)-220, 40), cv2.FONT_HERSHEY_PLAIN, 2, (40, 40, 40), 2)
            #############################
                
            cv2.imshow('camera', frame)
            vid_writer.write(frame)

            key = cv2.waitKey(5)
            if key == 27:
                break
            if key == ord('s'):
                cv2.waitKey()
    
        else:
            break
    vid_cap.release()
    vid_writer.release()
    cv2.destroyAllWindows()


def main(opt, device):
    assert opt.vid_path, f'input "--vid_path" argument, got "{opt.vid_path}"'
    assert Path(opt.vid_path).is_file(), f'given "{opt.vid_path}" file not exists'
    
    ckpt = torch.load(opt.ckpt_path)
    arch = ckpt.get('arch')
    img_size = ckpt.get('img_size')
    class_list = ckpt.get('class_list')
    model_state = ckpt.get('model_state')
    device = torch.device(device)
    conf_thres = 0.40
    nms_thres = 0.45

    transform = TestTransform(input_size=img_size, mean=GENERAL_MEAN, std=GENERAL_STD)
    model = build_model(arch_name=arch, num_classes=len(class_list))
    model.load_state_dict(state_dict=model_state, strict=True)
    model = model.to(device)
    model.half()

    x = torch.randn((1, 3, img_size, img_size), device=device)
    model(x.half())
    
    run(video_path=opt.vid_path, transform=transform, model=model, class_list=class_list, 
        device=device, conf_thres=conf_thres, nms_thres=nms_thres, result_dir=opt.result_dir)
    

if __name__ == "__main__":
    try:
        opt, _ = build_parser(root_dir=ROOT)
        main(opt=opt, device='cuda')
    except Exception as e:
        raise RuntimeError(e)