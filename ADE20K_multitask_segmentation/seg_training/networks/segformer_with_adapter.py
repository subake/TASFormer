import torch
import torch.nn as nn
import torchvision
from transformers import SegformerForSemanticSegmentation
from transformers import SegformerDecodeHead

class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes,
    ):
        super().__init__()
        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Load pretrained segformer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
        )

    def forward(self, inp, task_num):
        # Normalization
        inp = self.norm_input(inp)

        # Predict mask
        output = self.segformer(inp, task_num)

        # Resize masks
        logits = nn.functional.interpolate(output.logits, size=inp.shape[-2:], mode="bilinear", align_corners=True)

        return {'out': [logits]}