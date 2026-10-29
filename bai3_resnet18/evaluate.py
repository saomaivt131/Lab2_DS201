import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from data import build_test_loader
from engine import evaluate
from model import ResNet18


def get_args():
    p = argparse.ArgumentParser(description="Đánh giá ResNet18 trên tập test")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="outputs/best.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
    fig.colorbar(im)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    classes, img_size = ckpt["classes"], ckpt["img_size"]

    model = ResNet18(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(
        args.data_root, img_size, args.batch_size, args.num_workers)
    assert test_classes == classes, "thứ tự lớp ở test khác train"

    loss, acc, engine_f1, y_true, y_pred = evaluate(
    model,
    test_dl,
    nn.CrossEntropyLoss(),
    device
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    print()
    print("===== TEST RESULTS =====")
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

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("Model: ResNet-18 [2, 2, 2, 2]\n")
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
        confusion_path
    )

    print("đã lưu", report_path)
    print("đã lưu", confusion_path)


if __name__ == "__main__":
    main()
