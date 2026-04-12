#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_runtime, ensure_dir, evaluate, get_device, save_checkpoint, save_history_csv, save_json, seed_everything, train_one_epoch
from src.datasets import build_dogs_vs_cats_dataloaders
from src.models import build_model, maybe_freeze_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Dogs vs Cats classifier.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "dogs_vs_cats")
    parser.add_argument("--model-name", choices=["small_cnn", "resnet18"], default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--disable-train-augmentation", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-channels-last", action="store_true")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    deterministic = args.deterministic and not args.fast_mode
    seed_everything(args.seed, deterministic=deterministic)
    configure_runtime(device=device, deterministic=deterministic)
    output_dir = ensure_dir(args.output_dir)
    amp_enabled = device.type == "cuda" and not args.disable_amp
    use_channels_last = device.type == "cuda" and not args.disable_channels_last
    persistent_workers = not args.disable_persistent_workers
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    dataloaders = build_dogs_vs_cats_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_augment=not args.disable_train_augmentation,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=args.pretrained,
        cifar_stem=False,
    )
    if args.freeze_backbone and args.model_name == "resnet18":
        maybe_freeze_backbone(model)
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = []
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {
        "task": "dogs_vs_cats",
        "class_to_idx": dataloaders["class_to_idx"],
        "idx_to_class": dataloaders["idx_to_class"],
        "args": serializable_args,
        "train_size": dataloaders["train_size"],
        "val_size": dataloaders["val_size"],
        "test_size": dataloaders["test_size"],
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            scaler=scaler,
            use_channels_last=use_channels_last,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders["val_loader"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            use_channels_last=use_channels_last,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(row)
        print(row)

        save_checkpoint(
            path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_val_accuracy,
            metadata=metadata,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_accuracy,
                metadata=metadata,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "device": str(device),
        "history_length": len(history),
        "train_size": dataloaders["train_size"],
        "val_size": dataloaders["val_size"],
    }
    save_history_csv(output_dir / "history.csv", history)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "metadata.json", metadata)
    print(summary)


if __name__ == "__main__":
    main()
