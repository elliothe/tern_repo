import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn import init

from .tern_threshold_trainable_5 import quanConv2d, quanLinear

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, fp_fl=True, fp_ll=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False) if fp_fl
            else quanConv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
            quanConv2d(96, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
            quanConv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            quanConv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            quanConv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            quanLinear(256 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            quanLinear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes, bias=True) if fp_ll
            else quanLinear(4096, num_classes, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()            
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def quan_alexnet_ff_lf(num_classes=1000):
    model = AlexNet(fp_fl=True, fp_ll=True)
    return model

def quan_alexnet_fq_lq(num_classes=1000):
    model = AlexNet(fp_fl=False, fp_ll=False)
    return model