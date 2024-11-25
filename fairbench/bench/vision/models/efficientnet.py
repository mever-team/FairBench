import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0  # EfficientNetB0 model

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = efficientnet_b0(pretrained=pretrained)  # EfficientNetB0 with pretrained weights
            self.extractor = nn.Sequential(*list(model.children())[:-1])  # Remove classification head
            self.embed_size = 1280  # EfficientNetB0 output features
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            self.extractor = nn.Sequential(*list(model.children())[:-1])  # Same as before
            self.embed_size = 1280
            self.num_classes = num_classes
            self.fc = model.classifier[1]  # Use the classifier from the pre-loaded model
        
        print(f"EfficientNetB0 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)  # Flatten the output
        logits = self.fc(out)  # Forward through the classification head

        return logits, out

    def concat_forward(self, x, f):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        out = out + f  # Combine the extracted features with the auxiliary features
        logits = self.fc(out)

        return logits, out