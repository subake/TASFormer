import torch
import torch.nn as nn
import torchvision

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
        self.resnet_50.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.resnet_50.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inp):
        # Normalization
        inp = self.norm_input(inp)
        # Predict mask
        logits = self.resnet_50(inp)['out']

        return {'out': [logits]}
