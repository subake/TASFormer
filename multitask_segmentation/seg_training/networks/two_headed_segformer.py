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
            output_hidden_states=True,
        )

        # Remove decoder
        self.segformer = torch.nn.Sequential(*(list(self.segformer.children())[:-1]))

        # Add two decoders
        self.decode_head1 = SegformerDecodeHead.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
        )
        self.decode_head2 = SegformerDecodeHead.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
        )

    def forward(self, inp):
        # Normalization
        inp = self.norm_input(inp)

        # Predict mask
        outputs = self.segformer(inp)
        logits1 = self.decode_head1(outputs[1])
        logits2 = self.decode_head2(outputs[1])

        # Resize masks
        logits1 = nn.functional.interpolate(logits1, size=inp.shape[-2:], mode="bilinear", align_corners=True)
        logits2 = nn.functional.interpolate(logits2, size=inp.shape[-2:], mode="bilinear", align_corners=True)

        return {'out': [logits1, logits2]}
