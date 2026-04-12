#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import copy_ranked_samples, ensure_dir, get_device, load_checkpoint, save_json, write_prediction_csv
from src.datasets import build_dogs_vs_cats_dataloaders
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export correct and incorrect validation samples.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "dogs_vs_cats" / "analysis")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
    trained_model_name = checkpoint_data.get("metadata", {}).get("args", {}).get("model_name", "resnet18")

    dataloaders = build_dogs_vs_cats_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42,
        pin_memory=device.type == "cuda",
    )
    val_loader = dataloaders["val_loader"]

    model = build_model(model_name=trained_model_name, num_classes=2, pretrained=False)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model = model.to(device)
    model.eval()

    idx_to_class = dataloaders["idx_to_class"]
    dataset = val_loader.dataset
    rows = []
    sample_offset = 0

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)

        for inner_index in range(len(targets)):
            target = int(targets[inner_index].item())
            prediction = int(predictions[inner_index].cpu().item())
            confidence = float(confidences[inner_index].cpu().item())
            sample_path, _ = dataset.samples[sample_offset + inner_index]
            rows.append(
                {
                    "path": sample_path,
                    "true_label": idx_to_class[target],
                    "pred_label": idx_to_class[prediction],
                    "confidence": confidence,
                    "correct": int(target == prediction),
                }
            )

        sample_offset += len(targets)

    correct = sorted((item for item in rows if item["correct"] == 1), key=lambda x: x["confidence"], reverse=True)
    incorrect = sorted((item for item in rows if item["correct"] == 0), key=lambda x: x["confidence"], reverse=True)

    copy_ranked_samples(correct, output_dir / "correct", limit=args.top_k)
    copy_ranked_samples(incorrect, output_dir / "incorrect", limit=args.top_k)
    write_prediction_csv(
        path=output_dir / "val_predictions.csv",
        fieldnames=["path", "true_label", "pred_label", "confidence", "correct"],
        rows=rows,
    )
    save_json(
        output_dir / "summary.json",
        {
            "total_samples": len(rows),
            "correct_samples": len(correct),
            "incorrect_samples": len(incorrect),
            "top_k_exported": args.top_k,
        },
    )
    print(f"Analysis exported to {output_dir}")


if __name__ == "__main__":
    main()
