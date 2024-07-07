import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

if __package__:
    from .dataset import Dataset
    from .transforms import TrainTransform, ValidTransform, mixup
    from utils.general import seed_worker
else:
    from dataset import Dataset
    from transforms import TrainTransform, ValidTransform, mixup
    from utils.general import seed_worker


def build_dataset(opt):
    """build method for classification dataset via calling predefined <data>.yaml
    """
    train_dataset = Dataset(opt=opt, phase='train')
    val_dataset = Dataset(opt=opt, phase='val')
    train_transformer = TrainTransform(input_size=opt.img_size, mean=opt.mean, std=opt.std,
                                       degrees=opt.degrees, translate=opt.translate, scale=opt.scale,
                                       shear=opt.shear, perspective=opt.perspective,
                                       h_gain=opt.hsv_h, s_gain=opt.hsv_s, v_gain=opt.hsv_v,
                                       dataset=train_dataset, mosaic=opt.mosaic)
    val_transformer = ValidTransform(input_size=opt.img_size, mean=opt.mean, std=opt.std)
    train_dataset.transform =train_transformer
    val_dataset.transform = val_transformer
    return train_dataset, val_dataset


def build_dataloader(opt):
    seed = getattr(opt, 'seed')
    workers = getattr(opt, 'workers')
    batch_size = getattr(opt, 'batch_size')

    workers = min([os.cpu_count() // max(torch.cuda.device_count(), 1), 
                   batch_size if batch_size > 1 else 0, workers])
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset, val_dataset = build_dataset(opt=opt)
    train_sampler = RandomSampler(train_dataset, generator=generator)
    val_sampler = SequentialSampler(val_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              sampler=train_sampler, pin_memory=True, 
                              num_workers=workers, worker_init_fn=seed_worker,
                              generator=generator, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            sampler=val_sampler, pin_memory=True, 
                            num_workers=workers, worker_init_fn=seed_worker, 
                            generator=generator, collate_fn=val_dataset.collate_fn)
    return train_loader, val_loader


if __name__ == "__main__":
    import time
    from pathlib import Path
    from utils.args import build_parser

    ROOT = Path(__file__).resolve().parents[1]
    opt, parser = build_parser(root_dir=ROOT)
    train_loader, val_loader = build_dataloader(opt)

    avg_time = 0.0
    max_count = 30
    test_loader = iter(train_loader)
    for idx in range(max_count):
        tic = time.time()
        _ = next(test_loader)
        toc = time.time()
        elapsed_time = (toc - tic) * 1000
        avg_time += elapsed_time
        if idx == max_count:
            break
    print(f'avg time : {avg_time/max_count:.3f} ms')