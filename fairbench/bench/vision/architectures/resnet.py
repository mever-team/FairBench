import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import Bottleneck
import torch.nn.functional as F
import torchvision.models as models
import torch


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        model = resnet18()
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)

        return logits


class BAddResNet50(models.ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, dim=1)
        x = self.fc(x)

        return x
