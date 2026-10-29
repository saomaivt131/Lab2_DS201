import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from data import build_test_loader
from engine import evaluate
from model import LeNet5


def get_args():
    p = argparse.ArgumentParser(description="Đánh giá LeNet-5 trên tập test")
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
    pool = ckpt.get("pool", "avg")

    model = LeNet5(in_ch=3, img=img_size, num_classes=len(classes), pool=pool).to(device)
    print("Pooling:", pool)
    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(
        args.data_root, img_size, args.batch_size, args.num_workers)
    assert test_classes == classes, "thứ tự lớp ở test khác train"

    loss, acc, f1, y_true, y_pred = evaluate(
        model, test_dl, nn.CrossEntropyLoss(), device)
    print(f"checkpoint epoch {ckpt['epoch']} | test loss {loss:.4f} | "
          f"acc {acc:.4f} | macro-F1 {f1:.4f}\n")
    
    report = classification_report(
    y_true,
    y_pred,
    target_names=classes,
    digits=4,
    zero_division=0
    )

    print(report)

    report_path = Path(args.ckpt).parent / "classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Pooling: {pool}\n")
        f.write(f"Checkpoint epoch: {ckpt['epoch']}\n")
        f.write(f"Test loss: {loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro-F1: {f1:.4f}\n\n")
        f.write(report)

    print("đã lưu", report_path)

    out_path = Path(args.ckpt).parent / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix(y_true, y_pred), classes, out_path)
    print("đã lưu", out_path)


if __name__ == "__main__":
    main()
