import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import PretrainedResNet50, n_params


def get_args():
    p = argparse.ArgumentParser(
        description="Huấn luyện pretrained ResNet-50 trên VinaFood21"
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["feature", "finetune"],
        help="feature: đóng băng backbone; finetune: cập nhật toàn bộ model",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)

    # Nếu không truyền lr, chương trình tự chọn theo mode.
    p.add_argument("--lr", type=float, default=None)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]

    plt.plot(
        epochs,
        [h["train_loss"] for h in history],
        label="train loss",
    )
    plt.plot(
        epochs,
        [h["val_loss"] for h in history],
        label="val loss",
    )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    freeze_backbone = args.mode == "feature"

    if args.lr is None:
        lr = 1e-3 if freeze_backbone else 1e-4
    else:
        lr = args.lr

    if args.out_dir is None:
        out_dir = Path(f"outputs_{args.mode}")
    else:
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, classes = build_train_val_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"device          : {device}")
    print(f"mode            : {args.mode}")
    print(f"image size      : {args.img_size}")
    print(f"learning rate   : {lr}")
    print(f"số lớp          : {len(classes)}")
    print(f"train samples   : {len(train_dl.dataset)}")
    print(f"val samples     : {len(val_dl.dataset)}")

    model = PretrainedResNet50(
        num_classes=len(classes),
        freeze_backbone=freeze_backbone,
    ).to(device)

    print(f"total params    : {n_params(model):,}")
    print(
        "trainable params:",
        f"{n_params(model, trainable_only=True):,}",
    )

    criterion = nn.CrossEntropyLoss()

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
    )

    history = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_dl,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model,
            val_dl,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        mark = ""

        if val_f1 > best_f1:
            best_f1 = val_f1

            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "epoch": epoch,
                "mode": args.mode,
                "freeze_backbone": freeze_backbone,
                "lr": lr,
                "model_id": "microsoft/resnet-50",
            }, out_dir / "best.pt")

            mark = "  *"

        print(
            f"epoch {epoch:>2}  "
            f"train {train_loss:.4f}/{train_acc:.4f}  "
            f"val {val_loss:.4f}/{val_acc:.4f}  "
            f"F1 {val_f1:.4f}{mark}"
        )

    with open(
        out_dir / "history.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    save_loss_curve(
        history,
        out_dir / "loss_curve.png",
    )

    print()
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print("Checkpoint:", out_dir / "best.pt")


if __name__ == "__main__":
    main()