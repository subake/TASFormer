import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride= 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SemSegResNet50(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.1,
    ):
        super().__init__()
        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Load pretrained resnet
        self.resnet_50 = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            'fcn_resnet50', 
            pretrained=True,
        )

        # Change classifier to the desired number of classes
        in_features = 512
        self.resnet_50.backbone.layer4 = self.resnet_50.backbone.layer4[0]
        self.resnet_50.bottlenecks1 = nn.Sequential(Bottleneck(2048, 512,  1, None, 1, 64, 4),
                                                Bottleneck(2048, 512,  1, None, 1, 64, 4))
        
        self.resnet_50.bottlenecks2 = nn.Sequential(Bottleneck(2048, 512,  1, None, 1, 64, 4),
                                                Bottleneck(2048, 512,  1, None, 1, 64, 4))   

        self.resnet_50.classifier1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )
        self.resnet_50.classifier2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )
        


    def forward(self, inp):
        # Normalization
        inp = self.norm_input(inp)
        # Predict mask
        input_shape = inp.shape[-2:]

        x = self.resnet_50.backbone(inp)
        x1 = self.resnet_50.bottlenecks1(x['out'])
        x2 = self.resnet_50.bottlenecks2(x['out'])

        x1 = self.resnet_50.classifier1(x1)        
        x2 = self.resnet_50.classifier2(x2)

        x1 = F.interpolate(x1, size=input_shape, mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=input_shape, mode="bilinear", align_corners=False)

        return {'out': [x1, x2]}
