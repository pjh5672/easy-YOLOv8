import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if __package__:
    from .element import Conv, C2f, SPPF, Detect
else:
    from element import Conv, C2f, SPPF, Detect

from utils.torch_utils import make_divisible


CFG_SCALE = {
    # [depth, width, max_channels]
    'n': [0.33, 0.25, 1024],
    's': [0.33, 0.50, 1024],
    'm': [0.67, 0.75, 768],
    'l': [1.00, 1.00, 512],
    'x': [1.00, 1.25, 512],
}
CFG_WIDTH = [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 512, 256, 256, 512, 512, 1024]
CFG_DEPTH = [3, 6, 6, 3, 3, 3, 3, 3]


class YOLOv8(nn.Module):
    def __init__(self, scale='n', num_classes=80, in_channels=3):
        super().__init__()
        depth, width, max_channels = CFG_SCALE[scale]
        ch = [make_divisible(min(x, max_channels) * width, 8) for x in CFG_WIDTH]
        n = [max(round(x * depth), 1) for x in CFG_DEPTH]

        # backbone
        self.layer1 = Conv(in_channels, ch[0], k=3, s=2, p=1, act=True)
        self.layer2 = Conv(ch[0], ch[1], k=3, s=2, p=1, act=True)
        self.layer3 = C2f(ch[1], ch[2], n=n[0], shortcut=True)
        self.layer4 = Conv(ch[2], ch[3], k=3, s=2, p=1, act=True)
        self.layer5 = C2f(ch[3], ch[4], n=n[1], shortcut=True)
        self.layer6 = Conv(ch[4], ch[5], k=3, s=2, p=1, act=True)
        self.layer7 = C2f(ch[5], ch[6], n=n[2], shortcut=True)
        self.layer8 = Conv(ch[6], ch[7], k=3, s=2, p=1, act=True)
        self.layer9 = C2f(ch[7], ch[8], n=n[3], shortcut=True)
        self.sppf = SPPF(ch[8], ch[9])
        
        # head
        self.upsample = nn.Upsample(None, 2, 'nearest')
        self.layer10 = C2f(ch[6] + ch[9], ch[10], n=n[4], shortcut=False)
        self.layer11 = C2f(ch[4] + ch[10], ch[11], n=n[5], shortcut=False)
        self.layer12 = Conv(ch[11], ch[12], k=3, s=2, p=1, act=True)
        self.layer13 = C2f(ch[10] + ch[12], ch[13], n=n[6], shortcut=False)
        self.layer14 = Conv(ch[13], ch[14], k=3, s=2, p=1, act=True)
        self.layer15 = C2f(ch[9] + ch[14], ch[15], n=n[7], shortcut=False)
        self.head = Detect(nc=num_classes, ch=(ch[11], ch[13], ch[15]))

        s = 256
        m = self.head
        m.inplace = True
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))])
        m.bias_init()
        self.initialize_weights()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        C1 = self.layer5(x) # out
        x = self.layer6(C1)
        C2 = self.layer7(x) # out
        x = self.layer8(C2)
        x = self.layer9(x)
        P1 = self.sppf(x) # skip-conn
        
        x = self.upsample(P1)
        x = torch.cat((x, C2), dim=1)
        P2 = self.layer10(x) # skip-conn
        x = self.upsample(P2)
        x = torch.cat((x, C1), dim=1)
        P3 = self.layer11(x) # head(s)
        x = self.layer12(P3)
        x = torch.cat((x, P2), dim=1)
        P4 = self.layer13(x)
        x = self.layer14(P4)
        x = torch.cat((x, P1), dim=1)
        P5 = self.layer15(x)
        y = self.head([P3, P4, P5])
        return y
    
    def initialize_weights(self):
        """Initialize model weights to random values."""
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True


if __name__ == "__main__":
    import torch
    from utils.torch_utils import model_info
    
    model = YOLOv8(scale='s', num_classes=80)
    # print(model)
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    for yi in y:
        print(yi.shape)

    model_info(model, input_size=640)