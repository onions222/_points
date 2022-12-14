from torchvision import models
from torch import nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            models.resnet50()
        )
        self.out = nn.Linear(1000, 10)

    def forward(self, x):
        return self.out(self.layer(x))


if __name__ == '__main__':
    net = Net()
    x = torch.randn(1, 3, 100, 100)
    print(net(x).shape)
