import sys
from pathlib import Path

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if __package__:
    from .tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist
else:
    from tal import TaskAlignedAssigner, make_anchors, dist2bbox, bbox2dist

from utils.general import bbox_iou, xcycwhn_to_x1y1x2y2


class YOLOv8Loss:
    def __init__(self, model, device):
        m = model.head
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.dtype = torch.float32
        self.use_dfl = m.reg_max > 1
        self.lam_box = 7.5
        self.lam_cls = 0.5
        self.lam_dfl = 1.5
        self.topk = 10
        self.alpha = 0.5
        self.beta = 6.0
        
        self.proj = torch.arange(m.reg_max, dtype=self.dtype, device=self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.assigner = TaskAlignedAssigner(topk=self.topk, num_classes=self.nc, 
                                            alpha=self.alpha, beta=self.beta)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device=self.device)

    def __call__(self, preds, targets):
        loss = torch.zeros(3, device=self.device)
        batch_size = len(targets)
        
        pred = torch.cat([pred.view(batch_size, self.no, -1) for pred in preds], dim=-1)
        pred_distri, pred_scores = pred.split((self.reg_max * 4, self.nc), dim=1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        
        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        imgsz = np.array(preds[0].shape[2:], dtype=np.float32) * int(self.stride[0])
        targets = self.preprocess(targets=targets, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pd_scores = pred_scores.detach().sigmoid(), 
            pd_bboxes = (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points = anchor_points * stride_tensor, 
            gt_labels = gt_labels, 
            gt_bboxes = gt_bboxes,
            mask_gt = mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls_loss
        loss[1] = self.bce(pred_scores, target_scores.to(self.dtype)).sum() / target_scores_sum 

        # box_loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_dist = pred_distri, 
                                              pred_bboxes = pred_bboxes, 
                                              anchor_points = anchor_points, 
                                              target_bboxes = target_bboxes, 
                                              target_scores = target_scores, 
                                              target_scores_sum = target_scores_sum, 
                                              fg_mask = fg_mask)
        
        loss[0] *= self.lam_box
        loss[1] *= self.lam_cls
        loss[2] *= self.lam_dfl
        return loss.sum() * batch_size, loss.detach()
        
    def preprocess(self, targets, scale_tensor):
        targets[..., 1:5] = xcycwhn_to_x1y1x2y2(targets[..., 1:5] * scale_tensor, w=1, h=1)
        return torch.tensor(targets).to(self.device)
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
    
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


if __name__ == "__main__":
    from dataloader import build_dataloader
    from model import build_model
    from loss import build_criterion
    from utils.args import build_parser
    from utils.general import init_seeds

    opt, _ = build_parser(root_dir=ROOT)
    init_seeds(opt.seed + 1)

    arch = opt.arch
    input_size = opt.img_size
    num_classes = len(opt.class_list)
    opt.batch_size = 4
    device = torch.device('cuda')

    train_loader, val_loader = build_dataloader(opt=opt)
    model = build_model(arch_name=arch, num_classes=num_classes)
    criterion = build_criterion(model=model, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(device)
    
    for j in range(50):
        avg_box_loss, avg_cls_loss, avg_dfl_loss = 0., 0., 0.

        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            images, targets, filenames, shapes = batch[0], batch[1], batch[2], batch[3]
            images = images.to(device, non_blocking=True)
            preds = model(images)
            total_loss, loss = criterion(preds=preds, targets=targets)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            avg_box_loss += loss[0]
            avg_cls_loss += loss[1]
            avg_dfl_loss += loss[2]

        avg_box_loss /= len(train_loader)
        avg_cls_loss /= len(train_loader)
        avg_dfl_loss /= len(train_loader)
        print(f'ep{j} - box_loss: {avg_box_loss:.4f}, cls_loss: {avg_cls_loss:.4f}, dfl_loss: {avg_dfl_loss:.4f}')
