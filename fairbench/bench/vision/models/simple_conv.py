import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, kernel_size=7, **kwargs):
        super(SimpleConvNet, self).__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.extracter = nn.Sequential(*layers)
        # self.dropout = nn.Dropout2d(p=0.1)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        # self.bn4 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.dim_in = 128

        print(f"SimpleConvNet: kernel_size {kernel_size}")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_last_shared_layer(self):
        return self.fc

    def forward(self, x):
        feats = [None] * 5
        x = self.extracter[0](x)
        x = self.extracter[1](x)
        x = self.extracter[2](x)
        # x[:, 12, :, :] = torch.zeros_like(x[:, 12, :, :]).cuda()
        feats[0] = x
        # x = self.dropout(x)
        x = self.extracter[3](x)
        x = self.extracter[4](x)
        x = self.extracter[5](x)
        feats[1] = x
        x = self.extracter[6](x)
        x = self.extracter[7](x)
        x = self.extracter[8](x)
        feats[2] = x
        x = self.extracter[9](x)
        x = self.extracter[10](x)
        x = self.extracter[11](x)
        feats[3] = x
        x = self.avgpool(x)
        feats[4] = torch.flatten(x, 1)
        # feats[4] = F.normalize(feats[4], dim=1)
        logits = self.fc(feats[4])

        return logits, feats

    def concat_forward(self, x, f1, f2):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        # feat = F.normalize(feat, dim=1)
        feat = feat + (f1 + f2) / 2
        logits = self.fc(feat)

        return logits, feat

    # def concat_forward1(self, x, f1):
    #     x = self.extracter(x)
    #     x = self.avgpool(x)
    #     feat = torch.flatten(x, 1)
    #     feat = F.normalize(feat, dim=1)
    #     feat = feat + f1
    #     logits = self.fc(feat)

    #     return logits, feat

    def concat_forward1(self, x, f1):
        x = self.extracter[0](x)
        x = self.extracter[1](x)
        x = self.extracter[2](x)
        # x[:, 12, :, :] = torch.zeros_like(x[:, 12, :, :]).cuda()
        # x = self.dropout(x)
        x = self.extracter[3](x)
        x = self.extracter[4](x)
        x = self.extracter[5](x)
        x = self.extracter[6](x)
        x = self.extracter[7](x)
        x = self.extracter[8](x)
        x = self.extracter[9](x)
        x = self.extracter[10](x)
        x = self.extracter[11](x)
        x = self.avgpool(x)
        feats = torch.flatten(x, 1)
        # feats = F.normalize(feats, dim=1)
        # f1 = F.normalize(f1, dim=1)
        logits = self.fc(feats + f1)

        return logits, feats

    def get_feature(self, x):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat
