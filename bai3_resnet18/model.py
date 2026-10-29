import torch
import torch.nn as nn


def n_params(module):
    return sum(p.numel() for p in module.parameters())


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Khi shape không đổi, shortcut là identity.
        self.shortcut = nn.Identity()

        # Khi đổi channel hoặc giảm kích thước, dùng projection.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Cộng nhánh chính với shortcut trước ReLU.
        out = out + self.shortcut(x)

        return torch.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()

        # Ảnh 96x96 -> feature map 24x24.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # ResNet-18 có số block [2, 2, 2, 2].
        self.layer1 = self._make_stage(
            in_ch=64,
            out_ch=64,
            stride=1
        )

        self.layer2 = self._make_stage(
            in_ch=64,
            out_ch=128,
            stride=2
        )

        self.layer3 = self._make_stage(
            in_ch=128,
            out_ch=256,
            stride=2
        )

        self.layer4 = self._make_stage(
            in_ch=256,
            out_ch=512,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, stride):
        return nn.Sequential(
            # Block đầu đổi channel/kích thước nếu cần.
            BasicBlock(in_ch, out_ch, stride=stride),

            # Block thứ hai giữ nguyên shape.
            BasicBlock(out_ch, out_ch, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)

        return self.fc(x)


if __name__ == "__main__":
    model = ResNet18(num_classes=21)

    x = torch.rand(2, 3, 96, 96)
    logits = model(x)

    print("input :", tuple(x.shape))
    print("logits:", tuple(logits.shape))
    print("params:", f"{n_params(model):,}")

    assert logits.shape == (2, 21)
    assert n_params(model) == 11_187_285