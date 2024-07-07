from loss.v8loss import YOLOv8Loss


def build_criterion(**kwargs):
    model = kwargs.get('model')
    device = kwargs.get('device')
    return YOLOv8Loss(model=model, device=device)