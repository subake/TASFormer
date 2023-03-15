import torch
import torch.nn as nn
import torchvision

# For tasformer_with_adapter you additionally have to manually update files inside 
# anaconda3/envs/tasformer/lib/python3.8/site-packages/transformers/models/segformer/ 
# with files from transformers_update_for_adapters/hf/ or transformers_update_for_adapters/hf++/
from transformers import SegformerForSemanticSegmentation
from transformers import SegformerDecodeHead

class TASFormerAdapter(nn.Module):
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

        # Load pretrained model
        self.tasformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
        )

    def forward(self, inp, task_num):
        # Normalization
        inp = self.norm_input(inp)

        # Predict mask
        output = self.tasformer(inp, task_num)

        # Resize masks
        logits = nn.functional.interpolate(output.logits, size=inp.shape[-2:], mode="bilinear", align_corners=True)

        return {'out': [logits]}