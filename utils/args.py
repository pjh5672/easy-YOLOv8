import os
import argparse
from pathlib import Path

import yaml


def yaml_load(file):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


class ConfigParser:
    def __init__(self, data_dir, dataset):
        self.opt = argparse.Namespace()
        self.data_dir = data_dir
        self.cfg = yaml_load(data_dir / f'{dataset}.data.yaml')
    
    def update(self):
        self.opt.__dict__.update(self.cfg)
        return self.opt

    def change_dataset(self, to):
        self.cfg = yaml_load(self.data_dir / f'{to}.data.yaml')
        return self.update()


def build_parser(root_dir):
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, required=True, 
                        help='Name to project')
    
    parser.add_argument('--dataset', type=str, default='toy', 
                        help='Dataset')
    
    parser.add_argument('--arch', type=str, default='yolov8n', 
                        help='Model architecture')
    
    parser.add_argument('--img-size', type=int, default=448,
                        help='Input size (default:448)')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Number of total epochs to run')
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    
    parser.add_argument('--lr-decay', type=float, default=0.01,
                        help='Epoch to learning rate decay')
    
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warming up learning rate on early training stage')
    
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='Momentum for gradient optimization')
    
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for gradient optimization')

    parser.add_argument('--hsv-h', type=float, default=0.015,
                        help='Hue in HSV augmentation(default:0.015)')
    
    parser.add_argument('--hsv-s', type=float, default=0.7,
                        help='Saturation in HSV augmentation(default:0.7)')
    
    parser.add_argument('--hsv-v', type=float, default=0.4,
                        help='Value in HSV augmentation(default:0.4)')
    
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Rotation augmentation(default:0.0)')
    
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Translate augmentation(default:0.1)')
    
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale augmentation(default:0.5)')
    
    parser.add_argument('--shear', type=float, default=0.0,
                        help='Shear augmentation(default:0.0)')
    
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='Perspective augmentation(default:0.0)')

    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Alpha value for mixup augmentation(recommend:1.0)')
    
    parser.add_argument('--mosaic', action='store_true',
                        help='Mosaic augmentation with 4 images')
    
    parser.add_argument('--close-mosaic', type=int, default=0,
                        help='Epoch to close mosaic augmentation')
    
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='threshold to filter confidence score (default:0.001)')

    parser.add_argument('--nms-thres', type=float, default=0.7,
                        help='threshold to filter Box IoU of NMS process (default:0.7)')
    
    parser.add_argument('--evolve', type=int, nargs='?', const=300,
                        help='Evolve parameters for x generations (default:300)')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of workers for dataloader')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='Global training seed')

    parser.add_argument('--model-ema', action='store_true',
                        help='Enable tracking Exponential Moving Average of model parameters')
    
    parser.add_argument('--cos-lr', action='store_true',
                        help='One-cycle cosine annealing for learning rate')
    
    parser.add_argument('--no-amp', action='store_true',
                        help='Use of FP32 training (default:AMP training)')
    
    parser.add_argument('--scratch', action='store_true',
                        help='Scratch training without pretrained weight')
    
    parser.add_argument('--test-dir', type=str, nargs='?',
                        help='Directory to test data')
    
    parser.add_argument('--vid-path', type=str, nargs='?',
                        help='Video file path for inference')
    
    args = parser.parse_args()
    args.project_dir = root_dir / 'experiments' / args.project
    args.weight_dir = args.project_dir / 'weight'
    args.evolve_dir = args.project_dir / 'evolve'
    args.result_dir = args.project_dir / 'test_result'
    args.ckpt_path = args.weight_dir / 'best.pt'

    parser = ConfigParser(data_dir=root_dir / 'cfg', dataset=args.dataset)
    opt = parser.update()
    
    os.makedirs(args.weight_dir, exist_ok=True)
    if args.test_dir:
        os.makedirs(args.result_dir, exist_ok=True)
    if args.evolve:
        os.makedirs(args.evolve_dir, exist_ok=True)

    for k, v in vars(args).items():
        setattr(opt, k, v)
    return opt, parser


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    opt, parser = build_parser(root_dir=ROOT)
    print(opt)
    print(opt.dataroot_dir)
    opt = parser.change_dataset(to='catdog')
    print(opt.dataroot_dir)

    