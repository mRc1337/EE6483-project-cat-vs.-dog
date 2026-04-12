import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits.float(), targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def configure_runtime(device: torch.device, deterministic: bool = True) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def autocast_kwargs(device: torch.device, enabled: bool) -> Dict[str, object]:
    return {
        "device_type": device.type if device.type in {"cuda", "cpu"} else "cpu",
        "enabled": enabled and device.type == "cuda",
        "dtype": torch.float16,
    }


def get_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_history_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_channels_last: bool = False,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(**autocast_kwargs(device=device, enabled=amp_enabled)):
            logits = model(images)
        loss = criterion(logits.float(), targets)

        if scaler is not None and amp_enabled and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy_from_logits(logits.detach(), targets), batch_size)

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: Optional[nn.Module],
    device: torch.device,
    amp_enabled: bool = False,
    use_channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if use_channels_last and images.ndim == 4:
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(**autocast_kwargs(device=device, enabled=amp_enabled)):
            logits = model(images)

        if criterion is not None:
            loss = criterion(logits.float(), targets)
            loss_meter.update(loss.item(), images.size(0))

        acc_meter.update(accuracy_from_logits(logits, targets), images.size(0))

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.inference_mode()
def predict_logits(model: nn.Module, dataloader, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model.eval()
    outputs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    has_labels = True

    for batch in dataloader:
        if len(batch) == 3:
            images, targets, _ = batch
        elif len(batch) == 2:
            images, targets = batch
        else:
            raise ValueError("Unexpected batch format.")

        if targets is None:
            has_labels = False
        images = images.to(device, non_blocking=True)
        logits = model(images)
        outputs.append(logits.cpu())
        if has_labels and targets is not None:
            labels.append(targets.cpu())

    logits_tensor = torch.cat(outputs, dim=0)
    labels_tensor = torch.cat(labels, dim=0) if has_labels and labels else None
    return logits_tensor, labels_tensor


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_metric: float,
    metadata: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def compute_confusion_matrix(
    targets: Iterable[int],
    predictions: Iterable[int],
    num_classes: int,
) -> List[List[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, prediction in zip(targets, predictions):
        matrix[int(target)][int(prediction)] += 1
    return matrix


def copy_ranked_samples(
    records: List[Dict],
    destination: Path,
    limit: int,
) -> None:
    ensure_dir(destination)
    for item in records[:limit]:
        source = Path(item["path"])
        target_name = (
            f"score_{item['confidence']:.4f}_true_{item['true_label']}_pred_{item['pred_label']}_{source.name}"
        )
        destination.joinpath(target_name).write_bytes(source.read_bytes())
