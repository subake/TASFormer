from transformers import SegformerModel
from transformers import SegformerDecodeHead
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import math


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
        self.segformer_model = SegformerModel.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            output_hidden_states=True
        )

        self.segformer_decoder = SegformerDecodeHead.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
        )
        self.linear_c = torch.nn.ModuleList(list(self.segformer_decoder.children())[0])

        config = self.segformer_decoder.config

        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks + 10,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        
        self.task_emb_1 = nn.parameter.Parameter(data=torch.zeros(1, 10, 320//4, 320//4), requires_grad=True)
        self.task_emb_2 = nn.parameter.Parameter(data=torch.zeros(1, 10, 320//4, 320//4), requires_grad=True)

        self.task_emb_1.data.uniform_(-1.3, 1.3)
        self.task_emb_2.data.uniform_(-1.3, 1.3)

        self.config = config


    def forward(self, inp, task_num):
        # Normalization
        inp = self.norm_input(inp)
        input_shape = inp.shape[-2:]
        # Predict mask
        output = self.segformer_model(inp)
        
        encoder_hidden_states = output[1]
        
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        
        if task_num == 1:
            all_hidden_states += (self.task_emb_1.repeat(batch_size, 1, 1, 1),)
        else:
            all_hidden_states += (self.task_emb_2.repeat(batch_size, 1, 1, 1),)
        
        hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
        
        hidden_states = self.linear_fuse(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        logits = self.classifier(hidden_states)

        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        return {'out': [logits]}