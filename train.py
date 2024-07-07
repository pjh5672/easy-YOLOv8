import argparse
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from torch.cuda import amp

from dataloader import build_dataloader, mixup
from model import build_model
from loss import build_criterion
from evaluator import Evaluator
from val import validate, save_result
from utils.args import build_parser
from utils.general import (print_args, init_seeds, AverageMeter, 
                           TQDM_BAR_FORMAT, report_per_class)
from utils.torch_utils import (build_optimizer, build_scheduler, ModelEMA, 
                               model_info, time_sync)
from utils.evolve import ParamSearcher

ROOT = Path(__file__).resolve().parents[0]


def train_one_epoch(loader, model, criterion, optimizer, device, **kwargs):
    pbar = tqdm(enumerate(loader), total=len(loader), bar_format=TQDM_BAR_FORMAT)

    nw = kwargs.get('nw')
    no_amp = kwargs.get('no_amp')
    epoch = kwargs.get('epoch')
    num_epochs = kwargs.get('num_epochs')
    model_ema = kwargs.get('model_ema')
    scaler = kwargs.get('scaler')
    scheduler = kwargs.get('scheduler')
    batch_size = kwargs.get('batch_size')
    lf = kwargs.get('lf')
    momentum = kwargs.get('momentum')
    mixup_alpha = kwargs.get('mixup_alpha')
    epoch_time = kwargs.get('epoch_time')
    batch_time = kwargs.get('batch_time')
    total_loss = kwargs.get('total_loss')
    box_loss = kwargs.get('box_loss')
    cls_loss = kwargs.get('cls_loss')
    dfl_loss = kwargs.get('dfl_loss')
    warmup_bias_lr = 0.1
    warmup_momentum = 0.8

    batch_time.reset()
    total_loss.reset()
    box_loss.reset()
    cls_loss.reset()
    dfl_loss.reset()

    model.train()
    optimizer.zero_grad()
    t1 = time_sync()
    for i, batch in pbar:
        t2 = time_sync()
        ni = i + len(loader) * (epoch - 1)
        if ni <= nw:
            xi = [0, nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * lf(epoch - 1)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [warmup_momentum, momentum])

        images, targets = batch[0].to(device, non_blocking=True), batch[1]
        images, targets = mixup(inputs=images, targets=targets, alpha=mixup_alpha)

        with amp.autocast(enabled=not no_amp):
            preds = model(images)
            tot_loss, loss = criterion(preds=preds, targets=targets)

        total_loss.update(tot_loss.item() / batch_size, images.size(0))
        box_loss.update(loss[0].item(), images.size(0))
        cls_loss.update(loss[1].item(), images.size(0))
        dfl_loss.update(loss[2].item(), images.size(0))

        scaler.scale(tot_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
        if model_ema:
            model_ema.update_parameters(model)
        
        batch_time.update(time_sync() - t2)
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
        pbar.set_description(('%15s' + '%14s' + '%14.4g' * 6) % 
                                (f'{epoch}/{num_epochs}', mem, epoch_time.val, batch_time.val, 
                                total_loss.avg, box_loss.avg, cls_loss.avg, dfl_loss.avg))

    scheduler.step()
    epoch_time.update(time_sync() - t1)
    del images, targets, preds
    torch.cuda.empty_cache()


def train(opt, device):
    seed = getattr(opt, 'seed')
    dataset = getattr(opt, 'dataset')
    arch = getattr(opt, 'arch')
    img_size = getattr(opt, 'img_size')
    lr = getattr(opt, 'lr')
    batch_size = getattr(opt, 'batch_size') 
    class_list = getattr(opt, 'class_list')
    momentum = getattr(opt, 'momentum')
    weight_decay = getattr(opt, 'weight_decay')
    lr_decay = getattr(opt, 'lr_decay')
    num_epochs = getattr(opt, 'num_epochs')
    no_amp = getattr(opt, 'no_amp')
    cos_lr = getattr(opt, 'cos_lr')
    warmup = getattr(opt, 'warmup')
    project_dir = getattr(opt, 'project_dir')
    weight_dir = getattr(opt, 'weight_dir')
    evolve = getattr(opt, 'evolve')
    is_ema = getattr(opt, 'model_ema')
    mixup_alpha = getattr(opt, 'mixup_alpha')
    close_mosaic = getattr(opt, 'close_mosaic')
    conf_thres = getattr(opt, 'conf_thres')
    nms_thres = getattr(opt, 'nms_thres')
    eval_file = getattr(opt, 'val_file')
    scratch = getattr(opt, 'scratch')

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    init_seeds(seed + 1, deterministic=True)

    train_loader, val_loader = build_dataloader(opt=opt)
    model = build_model(arch_name=arch, num_classes=len(class_list))

    if not evolve:
        model_info(model=model, input_size=img_size)

    if not scratch and Path(f'./pretrained/{arch}.pt').exists():
        print(f'Start training from pretrained {arch} model...')
        pretrained = torch.load(f'./pretrained/{arch}.pt', map_location='cpu')
        model.load_state_dict(pretrained['state_dict'], strict=False)

    criterion = build_criterion(model=model, device=device)
    optimizer = build_optimizer(model=model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler, lf = build_scheduler(optimizer=optimizer, cos_lr=cos_lr, lr_decay=lr_decay, num_epochs=num_epochs)
    scaler = amp.GradScaler(enabled=not no_amp)
    evaluator = Evaluator(annoFile=eval_file)
    model.to(device)
    
    nw = max(round(warmup * len(train_loader)), 100)
    model_ema = ModelEMA(model=model) if is_ema else None

    start_epoch = 1
    epoch_time = AverageMeter('Epoch', ':5.3f')
    batch_time = AverageMeter('Batch', ':5.3f')
    total_loss = AverageMeter('TotalLoss', ':5.4f')
    box_loss = AverageMeter('BoxLoss', ':5.4f')
    cls_loss = AverageMeter('ClsLoss', ':5.4f')
    dfl_loss = AverageMeter('DflLoss', ':5.4f')
    best_epoch, best_ap, best_ap50, best_ap75, best_aps, best_apm, best_apl = [0] * 7

    for epoch in range(start_epoch, num_epochs + 1):
        print(('\n' + '%15s' + '%14s' * 7) % ('Epoch', 'GPU_mem', 'Time/Epoch', 'Time/Batch', 
                                              'Total_Loss', 'Box_Loss', 'Cls_Loss', 'Dfl_Loss'))
        
        if (epoch - 1) == (num_epochs - close_mosaic):
            train_loader.dataset.transform.close_mosaic()

        train_one_epoch(loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, 
                        device=device, nw=nw, no_amp=no_amp, epoch=epoch, num_epochs=num_epochs, 
                        model_ema=model_ema, scaler=scaler, scheduler=scheduler, batch_size=batch_size, 
                        lf=lf, momentum=momentum, mixup_alpha=mixup_alpha, epoch_time=epoch_time, 
                        batch_time=batch_time, total_loss=total_loss, box_loss=box_loss, 
                        cls_loss=cls_loss, dfl_loss=dfl_loss)
        
        if epoch > warmup:
            val_model = deepcopy(model_ema) if model_ema else deepcopy(model)
            summ_result, class_result = validate(loader=val_loader, model=val_model, 
                                                 evaluator=evaluator, device=device, 
                                                 conf_thres=conf_thres, nms_thres=nms_thres, 
                                                 class_list=class_list)

            if not evolve:
                keys = ('Epoch', 'AP@50:95', 'AP@50', 'AP75', 'AP@S', 'AP@M', 'AP@L')
                vals = (epoch, *summ_result[:6])
                save_result(keys=keys, vals=vals, save_dir=project_dir)

            save_obj = {}
            save_obj.update(dataset=dataset,
                            arch=arch,
                            img_size=img_size,
                            class_list=class_list,
                            model_state=val_model.state_dict())
            if model_ema:
                save_obj.update(model_state = val_model.module.state_dict())

            torch.save(save_obj, weight_dir / 'last.pt')
            if summ_result[0] > best_ap:
                best_result = deepcopy(class_result)
                best_epoch, best_ap, best_ap50, best_ap75, best_aps, best_apm, best_apl = \
                    epoch, *summ_result[:6]
                torch.save(save_obj, weight_dir / 'best.pt')

    if not evolve:
        report_per_class(save_dir=project_dir, src=best_result, filename='train_eval_per_class.csv')
    
    print()
    print(('%15s' + '%14s' * 7) % ('Final', 'Best Epoch', 'AP@50:95', 'AP@50',
                                   'AP@75', 'AP@S', 'AP@M', 'AP@L'))
    print(('%15s' + '%14i' + '%14.4g' * 6) % ('Result', best_epoch, best_ap, best_ap50, 
                                              best_ap75, best_aps, best_apm, best_apl))
    return best_ap, best_ap50, best_ap75, best_aps, best_apm, best_apl


def main(opt, device):
    print_args(opt, exclude_keys=('class_list', 'project_dir', 'weight_dir', 
                                  'evolve_dir', 'result_dir', 'ckpt_path', 'test_dir'))
    device = torch.device(device)

    if not opt.evolve:
        _ = train(opt, device)
    else:
        searcher = ParamSearcher(save_dir=opt.evolve_dir)
        for _ in range(opt.evolve):
            hyp = {k: vars(opt)[k] for k in list(searcher.params.keys())}
            searcher.run(hyp=hyp)
            opt = argparse.Namespace(**dict(vars(opt), **hyp))
            results = train(opt, device)
            keys = ('AP@50:95', 'AP@50', 'AP@75', 'AP@S', 'AP@M', 'AP@L')
            searcher.write_results(hyp=hyp, keys=keys, results=results)

        print(f'Hyperparameter evolution finished {opt.evolve} generations\n'
              f"Results saved to '{opt.evolve_dir}'.")


if __name__ == "__main__":
    try:
        opt, _ = build_parser(root_dir=ROOT)
        main(opt=opt, device='cuda')
    except Exception as e:
        raise RuntimeError(e)