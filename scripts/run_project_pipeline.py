#!/usr/bin/env python
import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import generate_report_figures


@dataclass
class Experiment:
    experiment_id: str
    task: str
    description: str
    command: List[str]
    output_dir: Path
    summary_path: Path
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full EE6483 project pipeline and collect report artifacts.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "project_pipeline")
    parser.add_argument("--profile", choices=["quick", "report"], default="report")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", default=None, help="Set CUDA_VISIBLE_DEVICES to a single GPU id, e.g. 0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-focal", action="store_true")
    parser.add_argument("--top-k-analysis", type=int, default=12)
    parser.add_argument("--dogs-epochs", type=int, default=None)
    parser.add_argument("--dogs-batch-size", type=int, default=None)
    parser.add_argument("--dogs-train-limit", type=int, default=None)
    parser.add_argument("--dogs-val-limit", type=int, default=None)
    parser.add_argument("--dogs-image-size", type=int, default=224)
    parser.add_argument("--cifar-epochs", type=int, default=None)
    parser.add_argument("--cifar-batch-size", type=int, default=None)
    parser.add_argument("--cifar-imbalance-ratio", type=float, default=0.1)
    return parser.parse_args()


def profile_defaults(profile: str) -> Dict[str, int]:
    if profile == "quick":
        return {
            "dogs_epochs": 3,
            "dogs_batch_size": 32,
            "dogs_train_limit": 2000,
            "dogs_val_limit": 500,
            "cifar_epochs": 3,
            "cifar_batch_size": 64,
        }
    return {
        "dogs_epochs": 10,
        "dogs_batch_size": 64,
        "dogs_train_limit": 16000,
        "dogs_val_limit": 5000,
        "cifar_epochs": 20,
        "cifar_batch_size": 128,
    }


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    normalized_rows: List[Dict] = []
    for row in rows:
        normalized = {}
        for key in fieldnames:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                normalized[key] = json.dumps(value, ensure_ascii=False)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def run_command(command: List[str], env: Dict[str, str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def maybe_run_experiment(experiment: Experiment, env: Dict[str, str], cwd: Path, skip_existing: bool) -> Dict:
    if skip_existing and experiment.summary_path.exists():
        print(f"Skipping existing experiment: {experiment.experiment_id}")
    else:
        run_command(experiment.command, env=env, cwd=cwd)

    summary = load_json(experiment.summary_path)
    return {
        "experiment_id": experiment.experiment_id,
        "task": experiment.task,
        "description": experiment.description,
        "output_dir": str(experiment.output_dir),
        **experiment.metadata,
        **summary,
    }


def build_dogs_experiments(args: argparse.Namespace, defaults: Dict[str, int]) -> List[Experiment]:
    root = args.project_root
    output_root = ensure_dir(args.output_root)
    data_root = root / "datasets"
    epochs = args.dogs_epochs or defaults["dogs_epochs"]
    batch_size = args.dogs_batch_size or defaults["dogs_batch_size"]
    train_limit = args.dogs_train_limit if args.dogs_train_limit is not None else defaults["dogs_train_limit"]
    val_limit = args.dogs_val_limit if args.dogs_val_limit is not None else defaults["dogs_val_limit"]

    common_args = [
        str(root / "scripts" / "train_dogs_vs_cats.py"),
        "--data-root",
        str(data_root),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(args.dogs_image_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--model-name",
        "resnet18",
    ]

    experiments = [
        Experiment(
            experiment_id="dogs_resnet18_pretrained_aug",
            task="dogs_vs_cats",
            description="ResNet18 + ImageNet 预训练 + 数据增强",
            command=[
                sys.executable,
                *common_args,
                "--output-dir",
                str(output_root / "dogs_resnet18_pretrained_aug"),
                "--pretrained",
                "--train-limit",
                str(train_limit),
                "--val-limit",
                str(val_limit),
            ],
            output_dir=output_root / "dogs_resnet18_pretrained_aug",
            summary_path=output_root / "dogs_resnet18_pretrained_aug" / "summary.json",
            metadata={"model_name": "resnet18", "pretrained": True, "train_augmentation": True},
        ),
        Experiment(
            experiment_id="dogs_resnet18_pretrained_noaug",
            task="dogs_vs_cats",
            description="ResNet18 + ImageNet 预训练 + 无训练增强",
            command=[
                sys.executable,
                *common_args,
                "--output-dir",
                str(output_root / "dogs_resnet18_pretrained_noaug"),
                "--pretrained",
                "--disable-train-augmentation",
                "--train-limit",
                str(train_limit),
                "--val-limit",
                str(val_limit),
            ],
            output_dir=output_root / "dogs_resnet18_pretrained_noaug",
            summary_path=output_root / "dogs_resnet18_pretrained_noaug" / "summary.json",
            metadata={"model_name": "resnet18", "pretrained": True, "train_augmentation": False},
        ),
        Experiment(
            experiment_id="dogs_smallcnn_aug",
            task="dogs_vs_cats",
            description="SmallCNN + 数据增强",
            command=[
                sys.executable,
                str(root / "scripts" / "train_dogs_vs_cats.py"),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_root / "dogs_smallcnn_aug"),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--image-size",
                str(args.dogs_image_size),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--model-name",
                "small_cnn",
                "--train-limit",
                str(train_limit),
                "--val-limit",
                str(val_limit),
            ],
            output_dir=output_root / "dogs_smallcnn_aug",
            summary_path=output_root / "dogs_smallcnn_aug" / "summary.json",
            metadata={"model_name": "small_cnn", "pretrained": False, "train_augmentation": True},
        ),
    ]
    return experiments


def build_cifar_experiments(args: argparse.Namespace, defaults: Dict[str, int]) -> List[Experiment]:
    root = args.project_root
    output_root = ensure_dir(args.output_root)
    data_root = root / "data" / "cifar10"
    epochs = args.cifar_epochs or defaults["cifar_epochs"]
    batch_size = args.cifar_batch_size or defaults["cifar_batch_size"]

    base_command = [
        sys.executable,
        str(root / "scripts" / "train_cifar10.py"),
        "--data-root",
        str(data_root),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--model-name",
        "resnet18",
    ]

    experiments = [
        Experiment(
            experiment_id="cifar10_baseline",
            task="cifar10",
            description="CIFAR-10 平衡数据基线",
            command=[
                *base_command,
                "--output-dir",
                str(output_root / "cifar10_baseline"),
            ],
            output_dir=output_root / "cifar10_baseline",
            summary_path=output_root / "cifar10_baseline" / "summary.json",
            metadata={"model_name": "resnet18", "imbalance": "none", "mitigation": "none"},
        ),
        Experiment(
            experiment_id="cifar10_imbalance_class_weight",
            task="cifar10",
            description="CIFAR-10 长尾不平衡 + 类别加权",
            command=[
                *base_command,
                "--output-dir",
                str(output_root / "cifar10_imbalance_class_weight"),
                "--imbalance",
                "long_tail",
                "--imbalance-ratio",
                str(args.cifar_imbalance_ratio),
                "--mitigation",
                "class_weight",
            ],
            output_dir=output_root / "cifar10_imbalance_class_weight",
            summary_path=output_root / "cifar10_imbalance_class_weight" / "summary.json",
            metadata={"model_name": "resnet18", "imbalance": "long_tail", "mitigation": "class_weight"},
        ),
        Experiment(
            experiment_id="cifar10_imbalance_weighted_sampler",
            task="cifar10",
            description="CIFAR-10 长尾不平衡 + WeightedSampler",
            command=[
                *base_command,
                "--output-dir",
                str(output_root / "cifar10_imbalance_weighted_sampler"),
                "--imbalance",
                "long_tail",
                "--imbalance-ratio",
                str(args.cifar_imbalance_ratio),
                "--mitigation",
                "weighted_sampler",
            ],
            output_dir=output_root / "cifar10_imbalance_weighted_sampler",
            summary_path=output_root / "cifar10_imbalance_weighted_sampler" / "summary.json",
            metadata={"model_name": "resnet18", "imbalance": "long_tail", "mitigation": "weighted_sampler"},
        ),
    ]

    if args.include_focal:
        experiments.append(
            Experiment(
                experiment_id="cifar10_imbalance_focal_loss",
                task="cifar10",
                description="CIFAR-10 长尾不平衡 + Focal Loss",
                command=[
                    *base_command,
                    "--output-dir",
                    str(output_root / "cifar10_imbalance_focal_loss"),
                    "--imbalance",
                    "long_tail",
                    "--imbalance-ratio",
                    str(args.cifar_imbalance_ratio),
                    "--mitigation",
                    "focal_loss",
                ],
                output_dir=output_root / "cifar10_imbalance_focal_loss",
                summary_path=output_root / "cifar10_imbalance_focal_loss" / "summary.json",
                metadata={"model_name": "resnet18", "imbalance": "long_tail", "mitigation": "focal_loss"},
            )
        )
    return experiments


def choose_best_dogs_experiment(rows: List[Dict]) -> Dict:
    candidates = [row for row in rows if row["task"] == "dogs_vs_cats"]
    return max(candidates, key=lambda row: row.get("best_val_accuracy", 0.0))


def generate_dogs_artifacts(
    args: argparse.Namespace,
    env: Dict[str, str],
    best_dog: Dict,
) -> Dict:
    root = args.project_root
    output_root = ensure_dir(args.output_root)
    checkpoint = Path(best_dog["output_dir"]) / "best.pt"
    submission_path = output_root / "submission.csv"
    val_predictions_path = output_root / "val_predictions.csv"
    analysis_dir = output_root / "analysis"

    run_command(
        [
            sys.executable,
            str(root / "scripts" / "predict_dogs_vs_cats.py"),
            "--data-root",
            str(root / "datasets"),
            "--checkpoint",
            str(checkpoint),
            "--output-csv",
            str(submission_path),
            "--split",
            "test",
            "--image-size",
            str(args.dogs_image_size),
            "--device",
            args.device,
        ],
        env=env,
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "predict_dogs_vs_cats.py"),
            "--data-root",
            str(root / "datasets"),
            "--checkpoint",
            str(checkpoint),
            "--output-csv",
            str(val_predictions_path),
            "--split",
            "val",
            "--image-size",
            str(args.dogs_image_size),
            "--device",
            args.device,
        ],
        env=env,
        cwd=root,
    )
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "analyze_dogs_vs_cats.py"),
            "--data-root",
            str(root / "datasets"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(analysis_dir),
            "--top-k",
            str(args.top_k_analysis),
            "--image-size",
            str(args.dogs_image_size),
            "--device",
            args.device,
        ],
        env=env,
        cwd=root,
    )

    analysis_summary = load_json(analysis_dir / "summary.json")
    return {
        "best_dogs_experiment_id": best_dog["experiment_id"],
        "best_dogs_checkpoint": str(checkpoint),
        "submission_csv": str(submission_path),
        "val_predictions_csv": str(val_predictions_path),
        "analysis_dir": str(analysis_dir),
        "analysis_summary": analysis_summary,
    }


def build_markdown_report(
    args: argparse.Namespace,
    rows: List[Dict],
    dogs_artifacts: Dict,
    figures_manifest: Dict,
    timestamp: str,
) -> str:
    dog_rows = [row for row in rows if row["task"] == "dogs_vs_cats"]
    cifar_rows = [row for row in rows if row["task"] == "cifar10"]
    best_dog = choose_best_dogs_experiment(rows)

    lines = [
        "# EE6483 自动实验摘要",
        "",
        f"- 生成时间：{timestamp}",
        f"- 运行配置：`profile={args.profile}`，`device={args.device}`，`seed={args.seed}`",
        f"- Dogs vs Cats 最优实验：`{best_dog['experiment_id']}`",
        f"- 最终 submission：`{dogs_artifacts['submission_csv']}`",
        f"- 验证集误分类分析目录：`{dogs_artifacts['analysis_dir']}`",
        "",
        "## Dogs vs Cats 实验对比",
        "",
        "| experiment_id | model | pretrained | augmentation | best_val_accuracy | best_epoch | train_size | val_size | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in dog_rows:
        lines.append(
            f"| {row['experiment_id']} | {row.get('model_name', '')} | {row.get('pretrained', '')} | "
            f"{row.get('train_augmentation', '')} | {row.get('best_val_accuracy', ''):.4f} | "
            f"{row.get('best_epoch', '')} | {row.get('train_size', '')} | {row.get('val_size', '')} | "
            f"{row.get('output_dir', '')} |"
        )

    lines.extend(
        [
            "",
            "## CIFAR-10 实验对比",
            "",
            "| experiment_id | imbalance | mitigation | best_val_accuracy | final_test_accuracy | train_size | val_size | output_dir |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in cifar_rows:
        lines.append(
            f"| {row['experiment_id']} | {row.get('imbalance', '')} | {row.get('mitigation', '')} | "
            f"{row.get('best_val_accuracy', 0.0):.4f} | {row.get('final_test_accuracy', 0.0):.4f} | "
            f"{row.get('train_size', '')} | {row.get('val_size', '')} | {row.get('output_dir', '')} |"
        )

    lines.extend(
        [
            "",
            "## 报告可直接引用的结论线索",
            "",
            f"- Dogs vs Cats 建议采用 `{best_dog['experiment_id']}` 作为最终模型，因为其验证集准确率最高。",
            "- `dogs_resnet18_pretrained_aug` 与 `dogs_resnet18_pretrained_noaug` 可直接用于比较数据增强对性能的影响。",
            "- `dogs_resnet18_pretrained_aug` 与 `dogs_smallcnn_aug` 可直接用于比较模型结构差异对性能的影响。",
            "- CIFAR-10 平衡基线与长尾不平衡实验可直接用于回答 PDF 中 `(g)` 与 `(h)` 两问。",
            "- 至少可使用 `class_weight` 和 `weighted_sampler` 两组结果说明类别不平衡缓解策略；若启用了 `focal_loss`，可作为额外补充。",
            "",
            "## 关键输出文件",
            "",
            f"- 实验总表：`{args.output_root / 'report_artifacts' / 'experiment_summary.csv'}`",
            f"- 机器可读汇总：`{args.output_root / 'report_artifacts' / 'report_context.json'}`",
            f"- Figure 清单：`{args.output_root / 'report_artifacts' / 'figures_manifest.json'}`",
            f"- 本 Markdown：`{args.output_root / 'report_artifacts' / 'report_summary.md'}`",
            "",
            "## 自动生成的报告图",
        ]
    )
    for figure_id, payload in figures_manifest.items():
        lines.append(f"- `{figure_id}`: `{payload['path']}`")
        lines.append(f"  建议用途：{payload['caption']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    defaults = profile_defaults(args.profile)
    output_root = ensure_dir(args.output_root)
    artifacts_dir = ensure_dir(output_root / "report_artifacts")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    experiment_rows: List[Dict] = []

    for experiment in build_dogs_experiments(args, defaults):
        row = maybe_run_experiment(experiment, env=env, cwd=args.project_root, skip_existing=args.skip_existing)
        experiment_rows.append(row)

    best_dog = choose_best_dogs_experiment(experiment_rows)
    dogs_artifacts = generate_dogs_artifacts(args=args, env=env, best_dog=best_dog)

    for experiment in build_cifar_experiments(args, defaults):
        row = maybe_run_experiment(experiment, env=env, cwd=args.project_root, skip_existing=args.skip_existing)
        experiment_rows.append(row)

    report_context = {
        "generated_at": timestamp,
        "profile": args.profile,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "output_root": str(output_root),
        "best_dogs_experiment": best_dog,
        "dogs_artifacts": dogs_artifacts,
        "experiments": experiment_rows,
    }

    figures_manifest = generate_report_figures(
        experiment_rows=experiment_rows,
        dogs_artifacts=dogs_artifacts,
        artifacts_dir=artifacts_dir,
    )
    report_context["figures_manifest"] = figures_manifest

    write_csv(artifacts_dir / "experiment_summary.csv", experiment_rows)
    write_json(artifacts_dir / "report_context.json", report_context)
    (artifacts_dir / "report_summary.md").write_text(
        build_markdown_report(
            args=args,
            rows=experiment_rows,
            dogs_artifacts=dogs_artifacts,
            figures_manifest=figures_manifest,
            timestamp=timestamp,
        ),
        encoding="utf-8",
    )
    print(f"Project pipeline finished. Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    main()
