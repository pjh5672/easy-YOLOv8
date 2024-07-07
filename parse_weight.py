from pathlib import Path
from collections import OrderedDict

import torch

ROOT = Path(__file__).resolve().parents[0]


if __name__ == '__main__':
    arch = 'yolov8'
    num_classes = 80
    ckpt = torch.load(ROOT / f'experiments/voc-test/weight/best.pt', map_location='cpu')
    model_state = ckpt.get('model_state')
    pretrained_state = OrderedDict()

    for k, v in model_state.items():
        if k.startswith('head'):
            continue

        pretrained_state[k] = v

    torch.save({'state_dict': pretrained_state}, f'./pretrained/{arch}.pt')