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

from src.common import FocalLoss, configure_runtime, ensure_dir, evaluate, get_device, save_checkpoint, save_history_csv, save_json, seed_everything, train_one_epoch
from src.datasets import build_cifar10_dataloaders, build_class_weights
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier and imbalance experiments.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "cifar10")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "cifar10")
    parser.add_argument("--model-name", choices=["small_cnn", "resnet18"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--disable-persistent-workers", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--disable-channels-last", action="store_true")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--imbalance", choices=["none", "long_tail"], default="none")
    parser.add_argument("--imbalance-ratio", type=float, default=0.1)
    parser.add_argument(
        "--mitigation",
        choices=["none", "class_weight", "weighted_sampler", "focal_loss"],
        default="none",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def build_criterion(args: argparse.Namespace, train_targets, device: torch.device) -> nn.Module:
    if args.mitigation == "class_weight":
        weights = build_class_weights(train_targets, num_classes=10).to(device)
        return nn.CrossEntropyLoss(weight=weights)
    if args.mitigation == "focal_loss":
        weights = None
        if args.imbalance != "none":
            weights = build_class_weights(train_targets, num_classes=10).to(device)
        return FocalLoss(gamma=2.0, alpha=weights)
    return nn.CrossEntropyLoss()


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

    dataloaders = build_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_size=args.val_size,
        imbalance=args.imbalance,
        imbalance_ratio=args.imbalance_ratio,
        mitigation=args.mitigation,
        pin_memory=device.type == "cuda",
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_model(
        model_name=args.model_name,
        num_classes=10,
        pretrained=False,
        cifar_stem=args.model_name == "resnet18",
    ).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = build_criterion(args, dataloaders["train_targets"], device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = []
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    final_test_accuracy = 0.0

    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {
        "task": "cifar10",
        "class_names": dataloaders["class_names"],
        "train_distribution": dataloaders["train_distribution"],
        "args": serializable_args,
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
        test_metrics = evaluate(
            model=model,
            dataloader=dataloaders["test_loader"],
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            use_channels_last=use_channels_last,
        )
        final_test_accuracy = test_metrics["accuracy"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
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
            patience_counter = 0
            save_checkpoint(
                path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_accuracy,
                metadata=metadata,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    summary = {
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "final_test_accuracy": final_test_accuracy,
        "train_size": dataloaders["train_size"],
        "val_size": dataloaders["val_size"],
        "test_size": dataloaders["test_size"],
        "train_distribution": dataloaders["train_distribution"],
        "device": str(device),
    }
    save_history_csv(output_dir / "history.csv", history)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "metadata.json", metadata)
    print(summary)


if __name__ == "__main__":
    main()
