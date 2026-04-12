#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import get_device, load_checkpoint, write_prediction_csv
from src.datasets import build_dogs_vs_cats_dataloaders
from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction for Dogs vs Cats.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "runs" / "dogs_vs_cats" / "submission.csv")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
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

    model = build_model(model_name=trained_model_name, num_classes=2, pretrained=False)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model = model.to(device)
    model.eval()

    if args.split == "test":
        dataloader = dataloaders["test_loader"]
        rows = []
        for images, _, paths in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = F.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1).cpu().tolist()
            confidences = probabilities.max(dim=1).values.cpu().tolist()
            for path, prediction, confidence in zip(paths, predictions, confidences):
                image_id = int(Path(path).stem)
                rows.append({"ID": image_id, "label": int(prediction), "confidence": round(confidence, 6)})

        rows.sort(key=lambda item: item["ID"])
        write_prediction_csv(
            path=args.output_csv,
            fieldnames=["ID", "label"],
            rows=[{"ID": item["ID"], "label": item["label"]} for item in rows],
        )
        print(f"Saved {len(rows)} test predictions to {args.output_csv}")
        return

    val_loader = dataloaders["val_loader"]
    rows = []
    idx_to_class = dataloaders["idx_to_class"]
    dataset = val_loader.dataset

    sample_offset = 0
    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1).cpu().tolist()
        confidences = probabilities.max(dim=1).values.cpu().tolist()
        targets_list = targets.tolist()

        for inner_index, (prediction, confidence, target) in enumerate(zip(predictions, confidences, targets_list)):
            sample_path, _ = dataset.samples[sample_offset + inner_index]
            rows.append(
                {
                    "path": sample_path,
                    "true_label": idx_to_class[int(target)],
                    "pred_label": idx_to_class[int(prediction)],
                    "confidence": round(confidence, 6),
                    "correct": int(prediction == target),
                }
            )
        sample_offset += len(targets_list)

    write_prediction_csv(
        path=args.output_csv,
        fieldnames=["path", "true_label", "pred_label", "confidence", "correct"],
        rows=rows,
    )
    print(f"Saved {len(rows)} validation predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
