import torch
import torch.nn as nn
from torchvision.models import vit_b_16  # ViT base model
import torch.nn.functional as F

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            model = vit_b_16(pretrained=pretrained)  # ViT base model with 16x16 patches
            self.extractor = model  # Remove classification head
           
            self.embed_size = model.heads.head.in_features  # Get the embedding size from the final layer of ViT\
            self.extractor.heads.head = nn.Identity()
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        
        print(f"VisionTransformer - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)  # Flatten the output for the classifier
        logits = self.fc(out)  # Forward through the classification head

        return logits, out

    def concat_forward(self, x, f):
        out = self.extractor(x)
        # print(out.shape)
        out = out.squeeze(-1).squeeze(-1)
        # print(out.shape)
        # print(torch.norm(out),torch.norm(x))
        # out = F.normalize(out, dim=1)
        # f = F.normalize(f, dim=1)
        out = out + 10*f # Combine the extracted features with the auxiliary features
        logits = self.fc(out)

        return logits, out
