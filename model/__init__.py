if __package__:
    from .yolov8 import YOLOv8
else:
    from yolov8 import YOLOv8


MODEL_LIST = ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
              

def build_model(arch_name, num_classes):
    """build method for defined architectures

    Args:
        arch_name (str): classifier name
        num_classes (int): number of classes in prediction

    Returns:
        torch.nn.Module: classifier architecture
    """
    
    arch_name = arch_name.lower()
    assert arch_name in MODEL_LIST, \
        f'not support such architecture, got {arch_name}.'
    return YOLOv8(scale=arch_name[-1], num_classes=num_classes)


if __name__ == '__main__':
    from utils.torch_utils import model_info

    model = build_model(arch_name='yolov8l', num_classes=80)
    model_info(model, input_size=640)