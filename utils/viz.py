import random

import cv2
import numpy as np
from torch import Tensor

from utils.general import xcycwhn_to_x1y1x2y2, GENERAL_MEAN, GENERAL_STD


def generate_random_color(num_colors):
    color_list = []
    for _ in range(num_colors):
        hex_color = ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        rgb_color = tuple(int(hex_color[k:k+2], 16) for k in (0, 2, 4))
        color_list.append(rgb_color)
    return color_list

random.seed(0)
COLOR_LIST = generate_random_color(100)


def visualize_prediction(image, prediction, class_list, 
                         font_scale=0.6, thickness=2, text_color=(250, 250, 250)):
    for det in prediction:
        image = visualize_box(class_list=class_list, image=image, prediction=det, show_class=True, 
                              show_score=True, text_color=text_color, fontscale=font_scale, thickness=thickness)
    return image[..., ::-1]

def visualize_box(class_list, image, target=None, prediction=None, 
                  show_class=False, show_score=False, text_color=(10, 250, 10), fontscale=0.7, thickness=2):
    label = target if target is not None else prediction
    class_id = int(label[0])
    box = label[1:5].astype(int)
    if class_id >= 0:
        color = COLOR_LIST[class_id]
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        if show_class:
            class_name = class_list[class_id]
            if show_score:
                class_name += f'({label[-1]*100:.0f}%)'
            ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale, 2)
            if target is not None:
                cv2.rectangle(image, (x_min, y_min - int(fontscale * text_height)), (int(x_min + text_width), y_min), color, -1)
            else:
                cv2.rectangle(image, (x_min, y_min - int(fontscale * text_height * 2.5)), (int(x_min + text_width), y_min), color, -1)
            cv2.putText(image, text=class_name, org=(x_min, y_min - int(0.3 * text_height)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale, color=text_color, thickness=thickness, lineType=cv2.LINE_AA)
    return image

def visualize_dataset(img_loader, class_list, mean, std, show_nums=6, 
                      font_scale=1.5, thickness=4, text_color=(250, 250, 250)):
    batch = next(iter(img_loader))
    images, labels = batch[0], batch[1]
    check_images = []
    for i in range(len(images)):
        image = to_image(images[i], mean, std).copy()
        if isinstance(labels, Tensor):
            label = labels[labels[:, 0] == i][:, 1:].numpy().copy()
        else:
            label = labels[i].copy()
        label[:, 1:5] = xcycwhn_to_x1y1x2y2(label[:, 1:5], w=image.shape[1], h=image.shape[0])
        for lbl in label:
            image = visualize_box(class_list=class_list, image=image, target=lbl, show_class=True, 
                                  text_color=text_color, fontscale=font_scale, thickness=thickness)
        check_images.append(image)
        if len(check_images) >= show_nums:
            return np.concatenate(check_images, axis=1)
    return np.concatenate(check_images, axis=1)

def to_image(tensor, mean=GENERAL_MEAN, std=GENERAL_STD):
    denorm_tensor = tensor.cpu().clone()
    for t, m, s in zip(denorm_tensor, mean, std):
        t.mul_(s).add_(m)
    denorm_tensor.clamp_(min=0, max=1.)
    denorm_tensor *= 255
    return denorm_tensor.permute(1,2,0).numpy().astype(np.uint8)