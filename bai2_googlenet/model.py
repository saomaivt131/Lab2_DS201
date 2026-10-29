import torch
import torch.nn as nn


def n_params(module):
    return sum(p.numel() for p in module.parameters())


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()

        # Nhánh 1: Conv 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Nhánh 2: Conv 1x1 -> Conv 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3r, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Nhánh 3: Conv 1x1 -> Conv 5x5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5r, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # Nhánh 4: MaxPool -> Conv 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # [B, 3, 64, 64] -> [B, 32, 32, 32]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # 16 + 24 + 8 + 8 = 56 channels
        self.inc1 = InceptionBlock(
            in_ch=32,
            f1=16,
            f3r=16,
            f3=24,
            f5r=4,
            f5=8,
            fp=8
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        # 32 + 48 + 16 + 16 = 112 channels
        self.inc2 = InceptionBlock(
            in_ch=56,
            f1=32,
            f3r=32,
            f3=48,
            f5r=8,
            f5=16,
            fp=16
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        # 64 + 64 + 16 + 16 = 160 channels
        self.inc3 = InceptionBlock(
            in_ch=112,
            f1=64,
            f3r=48,
            f3=64,
            f5r=8,
            f5=16,
            fp=16
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(160, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.inc1(x)
        x = self.pool1(x)

        x = self.inc2(x)
        x = self.pool2(x)

        x = self.inc3(x)

        x = self.gap(x)
        x = x.flatten(1)

        return self.fc(x)


if __name__ == "__main__":
    model = GoogLeNet(num_classes=21)

    x = torch.rand(2, 3, 64, 64)
    logits = model(x)

    print("input :", tuple(x.shape))
    print("logits:", tuple(logits.shape))
    print("params:", f"{n_params(model):,}")