import torch
import torch.nn as nn

import timm

class TimmModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone = args.model
        num_classes = args.num_classes
        
        self.model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output