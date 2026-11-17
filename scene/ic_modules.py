import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return x + h

class ICModule(nn.Module):
    def __init__(self, in_ch=3, c=64, num_blocks=2):
        super().__init__()
        self.head = nn.Conv2d(in_ch, c, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(c) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(c, in_ch, 3, padding=1)

        with torch.no_grad():
            self.tail.weight.mul_(0.01)
            self.tail.bias.zero_()

    def forward(self, I_sr):
        feat = self.head(I_sr)
        feat = self.body(feat)
        res  = self.tail(feat)
        return I_sr + res  , res# I_IMSR

