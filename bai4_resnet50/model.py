import torch
import torch.nn as nn
from transformers import ResNetForImageClassification


MODEL_ID = "microsoft/resnet-50"


def n_params(module, trainable_only=False):
    params = module.parameters()

    if trainable_only:
        params = (p for p in params if p.requires_grad)

    return sum(p.numel() for p in params)


class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()

        # Tải model ResNet-50 đã học trên ImageNet.
        base = ResNetForImageClassification.from_pretrained(MODEL_ID)

        # Chỉ lấy phần trích xuất đặc trưng.
        self.resnet = base.resnet

        # Thay classifier 1000 lớp bằng classifier 21 lớp.
        self.classifier = nn.Linear(2048, num_classes)

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        # Khi feature extraction, giữ BatchNorm của backbone ở eval mode.
        if mode and self.freeze_backbone:
            self.resnet.eval()

        return self

    def forward(self, images):
        # [B, 2048, 1, 1]
        features = self.resnet(images).pooler_output

        # [B, 2048]
        features = features.flatten(1)

        # [B, 21]
        return self.classifier(features)


if __name__ == "__main__":
    model = PretrainedResNet50(
        num_classes=21,
        freeze_backbone=True
    )

    x = torch.rand(2, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    print("input             :", tuple(x.shape))
    print("logits            :", tuple(logits.shape))
    print("total params      :", f"{n_params(model):,}")
    print("trainable params  :", f"{n_params(model, trainable_only=True):,}")

    assert logits.shape == (2, 21)
    assert n_params(model) == 23_551_061
    assert n_params(model, trainable_only=True) == 43_029

    print("Model ResNet-50 feature extraction: OK")