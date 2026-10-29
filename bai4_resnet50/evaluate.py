import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data import build_test_loader
from engine import evaluate
from model import PretrainedResNet50


def get_args():
    p = argparse.ArgumentParser(
        description="Đánh giá pretrained ResNet-50 trên tập test"
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(
        range(len(classes)),
        classes,
        rotation=90,
    )
    ax.set_yticks(
        range(len(classes)),
        classes,
    )

    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")

    fig.colorbar(image)
    fig.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(
        args.ckpt,
        map_location=device,
    )

    classes = ckpt["classes"]
    img_size = ckpt["img_size"]
    mode = ckpt.get("mode", "finetune")
    freeze_backbone = mode == "feature"

    print(f"device          : {device}")
    print(f"checkpoint mode : {mode}")
    print(f"checkpoint epoch: {ckpt['epoch']}")
    print(f"image size      : {img_size}")

    model = PretrainedResNet50(
        num_classes=len(classes),
        freeze_backbone=freeze_backbone,
    ).to(device)

    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(
        data_root=args.data_root,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    assert test_classes == classes, (
        "Thứ tự lớp ở tập test khác tập train"
    )

    loss, acc, engine_f1, y_true, y_pred = evaluate(
        model,
        test_dl,
        nn.CrossEntropyLoss(),
        device,
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print()
    print("===== TEST RESULTS =====")
    print(f"Mode             : {mode}")
    print(f"Checkpoint epoch : {ckpt['epoch']}")
    print(f"Test loss        : {loss:.4f}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision-macro  : {precision:.4f}")
    print(f"Recall-macro     : {recall:.4f}")
    print(f"F1-macro         : {macro_f1:.4f}")
    print()
    print(report)

    output_dir = Path(args.ckpt).parent

    report_path = output_dir / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Model: pretrained ResNet-50\n")
        file.write("Source: microsoft/resnet-50\n")
        file.write(f"Mode: {mode}\n")
        file.write(f"Image size: {img_size}x{img_size}\n")
        file.write(f"Checkpoint epoch: {ckpt['epoch']}\n")
        file.write(f"Test loss: {loss:.4f}\n")
        file.write(f"Accuracy: {acc:.4f}\n")
        file.write(f"Precision-macro: {precision:.4f}\n")
        file.write(f"Recall-macro: {recall:.4f}\n")
        file.write(f"F1-macro: {macro_f1:.4f}\n\n")
        file.write(report)

    confusion_path = output_dir / "confusion_matrix.png"

    save_confusion_matrix(
        confusion_matrix(y_true, y_pred),
        classes,
        confusion_path,
    )

    print("Đã lưu:", report_path)
    print("Đã lưu:", confusion_path)


if __name__ == "__main__":
    main()