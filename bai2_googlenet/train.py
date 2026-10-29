import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import GoogLeNet, n_params


def get_args():
    p = argparse.ArgumentParser(description="Huấn luyện GoogLeNet trên VinaFood21")
    p.add_argument("--data_root", type=str, required=True,
                   help="thư mục chứa train/ (và val/, test/ nếu có)")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")             # chạy trên terminal, không có màn hình
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    plt.plot(ep, [h["train_loss"] for h in history], label="train loss")
    plt.plot(ep, [h["val_loss"] for h in history], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, args.img_size, args.batch_size,
        num_workers=args.num_workers, seed=args.seed)
    print(f"device: {device} | số lớp: {len(classes)} | "
          f"train: {len(train_dl.dataset)} | val: {len(val_dl.dataset)}")

    model = GoogLeNet(num_classes=len(classes)).to(device)
    print(f"GoogLeNet: {n_params(model):,} tham số")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history, best_f1 = [], -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1))

        mark = ""
        if va_f1 > best_f1:           # giữ checkpoint có macro-F1 trên val cao nhất
            best_f1 = va_f1
            torch.save({"model": model.state_dict(), "classes": classes,
                        "img_size": args.img_size, "epoch": ep},
                       out_dir / "best.pt")
            mark = "  *"
        print(f"epoch {ep:>2}  train {tr_loss:.4f}/{tr_acc:.4f}"
              f"   val {va_loss:.4f}/{va_acc:.4f}  F1 {va_f1:.4f}{mark}")

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    save_loss_curve(history, out_dir / "loss_curve.png")
    print(f"\nbest val macro-F1: {best_f1:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
