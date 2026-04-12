import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DOG_LABELS = ["cat", "dog"]
CIFAR_COLORS = ["#1f3c88", "#3a7ca5", "#81b29a", "#f2cc8f", "#e07a5f", "#8d5a97"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_history_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        return rows


def load_prediction_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _clean_label(name: str) -> str:
    return (
        name.replace("dogs_", "")
        .replace("cifar10_", "")
        .replace("pretrained_", "")
        .replace("weighted_sampler", "sampler")
        .replace("_", "\n")
    )


def plot_dogs_training_curves(experiment_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    dog_rows = [row for row in experiment_rows if row["task"] == "dogs_vs_cats"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for row in dog_rows:
        history = load_history_csv(Path(row["output_dir"]) / "history.csv")
        epochs = [item["epoch"] for item in history]
        axes[0].plot(epochs, [item["train_accuracy"] for item in history], linestyle="--", alpha=0.6)
        axes[0].plot(epochs, [item["val_accuracy"] for item in history], label=_clean_label(row["experiment_id"]))
        axes[1].plot(epochs, [item["train_loss"] for item in history], linestyle="--", alpha=0.35)
        axes[1].plot(epochs, [item["val_loss"] for item in history], label=_clean_label(row["experiment_id"]))

    axes[0].set_title("Dogs vs Cats Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Dogs vs Cats Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = figures_dir / "dogs_training_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "Dogs vs Cats 不同模型/数据增强设置下的训练与验证曲线，可用于展示收敛速度与过拟合情况。",
    }


def plot_dogs_comparison(experiment_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    dog_rows = [row for row in experiment_rows if row["task"] == "dogs_vs_cats"]
    labels = [_clean_label(row["experiment_id"]) for row in dog_rows]
    values = [row.get("best_val_accuracy", 0.0) for row in dog_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=["#1f3c88", "#3a7ca5", "#e07a5f"])
    ax.set_ylim(0.0, min(1.0, max(values) + 0.1 if values else 1.0))
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Dogs vs Cats Model Comparison")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", fontsize=9)

    fig.tight_layout()
    path = figures_dir / "dogs_model_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "Dogs vs Cats 不同模型与数据处理策略的验证集精度对比，可直接用于回答模型与数据处理对性能影响的问题。",
    }


def plot_confusion_matrix(prediction_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    label_to_idx = {name: index for index, name in enumerate(DOG_LABELS)}
    matrix = np.zeros((2, 2), dtype=np.int32)
    for row in prediction_rows:
        true_idx = label_to_idx[row["true_label"]]
        pred_idx = label_to_idx[row["pred_label"]]
        matrix[true_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(2), DOG_LABELS)
    ax.set_yticks(range(2), DOG_LABELS)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Dogs vs Cats Validation Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / "dogs_confusion_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "Dogs vs Cats 最优模型在验证集上的混淆矩阵，可用于分析猫狗两类的主要误判方向。",
    }


def _sample_gallery_paths(folder: Path, limit: int) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png"}
    paths = [path for path in sorted(folder.iterdir()) if path.suffix.lower() in extensions]
    return paths[:limit]


def plot_sample_gallery(analysis_dir: Path, figures_dir: Path, samples_per_group: int = 6) -> Dict:
    correct_paths = _sample_gallery_paths(analysis_dir / "correct", samples_per_group)
    incorrect_paths = _sample_gallery_paths(analysis_dir / "incorrect", samples_per_group)
    rows = 2
    cols = max(len(correct_paths), len(incorrect_paths), 1)
    fig, axes = plt.subplots(rows, cols, figsize=(2.7 * cols, 5.8))
    if cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    for col in range(cols):
        for row_idx, paths in enumerate([correct_paths, incorrect_paths]):
            ax = axes[row_idx, col]
            ax.axis("off")
            if col < len(paths):
                image = Image.open(paths[col]).convert("RGB")
                ax.imshow(image)
                ax.set_title(paths[col].name[:55], fontsize=8)

    axes[0, 0].set_ylabel("Correct", fontsize=12)
    axes[1, 0].set_ylabel("Incorrect", fontsize=12)
    fig.suptitle("Dogs vs Cats Case Study Gallery", fontsize=14)
    fig.tight_layout()
    path = figures_dir / "dogs_case_gallery.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "Dogs vs Cats 验证集中的高置信度正确样本与错误样本示例，可直接用于案例分析部分。",
    }


def plot_cifar_training_curves(experiment_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    cifar_rows = [row for row in experiment_rows if row["task"] == "cifar10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for row in cifar_rows:
        history = load_history_csv(Path(row["output_dir"]) / "history.csv")
        epochs = [item["epoch"] for item in history]
        label = _clean_label(row["experiment_id"])
        axes[0].plot(epochs, [item["val_accuracy"] for item in history], label=label)
        axes[1].plot(epochs, [item["test_accuracy"] for item in history], label=label)

    axes[0].set_title("CIFAR-10 Validation Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("CIFAR-10 Test Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = figures_dir / "cifar10_training_curves.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "CIFAR-10 基线与不平衡处理方法的验证/测试精度曲线，可用于比较不同策略的收敛表现。",
    }


def plot_cifar_comparison(experiment_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    cifar_rows = [row for row in experiment_rows if row["task"] == "cifar10"]
    labels = [_clean_label(row["experiment_id"]) for row in cifar_rows]
    val_acc = [row.get("best_val_accuracy", 0.0) for row in cifar_rows]
    test_acc = [row.get("final_test_accuracy", 0.0) for row in cifar_rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, val_acc, width, label="Best Val Acc", color="#3a7ca5")
    bars2 = ax.bar(x + width / 2, test_acc, width, label="Final Test Acc", color="#e07a5f")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, min(1.0, max(val_acc + test_acc) + 0.1 if labels else 1.0))
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR-10 Experiment Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.4f}", ha="center", fontsize=8)

    fig.tight_layout()
    path = figures_dir / "cifar10_experiment_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "CIFAR-10 基线与不平衡缓解方法的验证/测试精度对比，可用于回答不同方法对结果的影响。",
    }


def plot_cifar_distribution(experiment_rows: Sequence[Dict], figures_dir: Path) -> Dict:
    baseline = next(row for row in experiment_rows if row["task"] == "cifar10" and row.get("imbalance") == "none")
    long_tail = next(row for row in experiment_rows if row["task"] == "cifar10" and row.get("imbalance") == "long_tail")

    baseline_distribution = baseline["train_distribution"]
    long_tail_distribution = long_tail["train_distribution"]
    class_ids = sorted(int(key) for key in baseline_distribution.keys())
    baseline_values = [baseline_distribution[str(class_id)] if str(class_id) in baseline_distribution else baseline_distribution[class_id] for class_id in class_ids]
    long_tail_values = [long_tail_distribution[str(class_id)] if str(class_id) in long_tail_distribution else long_tail_distribution[class_id] for class_id in class_ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    axes[0].bar(class_ids, baseline_values, color="#81b29a")
    axes[0].set_title("Balanced Training Split")
    axes[0].set_xlabel("Class ID")
    axes[0].set_ylabel("Number of Images")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(class_ids, long_tail_values, color="#e07a5f")
    axes[1].set_title("Long-Tail Training Split")
    axes[1].set_xlabel("Class ID")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("CIFAR-10 Class Distribution for Imbalance Experiment")
    fig.tight_layout()
    path = figures_dir / "cifar10_class_distribution.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "path": str(path),
        "caption": "CIFAR-10 平衡训练集与长尾训练集的类别分布对比，可用于说明不平衡设定。",
    }


def generate_report_figures(experiment_rows: Sequence[Dict], dogs_artifacts: Dict, artifacts_dir: Path) -> Dict:
    figures_dir = ensure_dir(artifacts_dir / "figures")
    manifest = {}

    manifest["dogs_training_curves"] = plot_dogs_training_curves(experiment_rows, figures_dir)
    manifest["dogs_model_comparison"] = plot_dogs_comparison(experiment_rows, figures_dir)
    manifest["dogs_confusion_matrix"] = plot_confusion_matrix(
        load_prediction_csv(Path(dogs_artifacts["val_predictions_csv"])),
        figures_dir,
    )
    manifest["dogs_case_gallery"] = plot_sample_gallery(Path(dogs_artifacts["analysis_dir"]), figures_dir)
    manifest["cifar10_training_curves"] = plot_cifar_training_curves(experiment_rows, figures_dir)
    manifest["cifar10_experiment_comparison"] = plot_cifar_comparison(experiment_rows, figures_dir)
    manifest["cifar10_class_distribution"] = plot_cifar_distribution(experiment_rows, figures_dir)

    save_json(artifacts_dir / "figures_manifest.json", manifest)
    return manifest
