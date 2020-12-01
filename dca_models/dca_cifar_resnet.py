import torch.nn as nn
from torchvision.models import ResNet
from .dca_module import dca_layer as DCALayer

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class CifarDCABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, k_size=3):
        super(CifarDCABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dca = DCALayer(planes, k_size)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dca(out)

        out += residual
        out = self.relu(out)

        return out


class CifarDCAResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=100, k_size=[3,3,3]):
        super(CifarDCAResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, k_size=int(k_size[0]))
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, k_size=int(k_size[1]))
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, k_size=int(k_size[2]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, k_size):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, k_size))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def dca_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarDCAResNet(CifarDCABasicBlock, 3, k_size=[3,3,5], **kwargs)
    return model


def dca_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarDCAResNet(CifarDCABasicBlock, 5, k_size=[3,3,5],**kwargs)
    return model


def dca_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarDCAResNet(CifarDCABasicBlock, 9, k_size=[3,3,5], **kwargs)
    return model
