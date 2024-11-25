import torch
import torch.nn as nn
from torchvision.models import swin_t  # Swin Transformer Tiny model

class SwinTransformer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = swin_t(pretrained=pretrained)  # Swin Transformer Tiny model with pretrained weights
            self.extractor = model # Remove classification head
            self.embed_size = model.head.in_features  # Get the embedding size from the final layer
            self.extractor.head = nn.Identity()
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        
        print(f"SwinTransformer - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)  # Flatten the output
        logits = self.fc(out)  # Forward through the classification head

        return logits, out

    def concat_forward(self, x, f):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        out = out + 10*f  # Combine the extracted features with the auxiliary features
        logits = self.fc(out)

        return logits, out
