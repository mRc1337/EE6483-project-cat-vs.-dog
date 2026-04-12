# EE6483 Mini Project 2 脚本说明

本目录提供了按 `/usr1/home/s125mdg56_02/mrc/cat_vs_dog/EE6483-Project2.pdf` 要求整理的一套可复现实验脚本，覆盖以下内容：

1. Dogs vs Cats 二分类训练、验证与测试集预测。
2. 生成符合作业要求的 `submission.csv`。
3. 导出验证集上的正确/错误样本，便于撰写误分类分析。
4. CIFAR-10 多分类训练脚本。
5. CIFAR-10 类别不平衡实验，支持至少两种改进方法：`class_weight`、`weighted_sampler`，另外额外提供 `focal_loss`。

## 1. 目录结构

```text
cat_vs_dog/
├── datasets/                     # 已提供的 Dogs vs Cats 数据
├── data/cifar10/                 # CIFAR-10 默认下载位置，首次运行自动创建
├── scripts/
│   ├── train_dogs_vs_cats.py     # Dogs vs Cats 训练
│   ├── predict_dogs_vs_cats.py   # Dogs vs Cats 验证/测试预测与 submission 导出
│   ├── analyze_dogs_vs_cats.py   # 导出正确/错误样本，辅助写报告
│   ├── train_cifar10.py          # CIFAR-10 常规训练 + 不平衡实验
│   └── run_project_pipeline.py   # 一键跑完整项目并自动汇总报告素材
├── src/
│   ├── common.py                 # 训练循环、checkpoint、指标与通用工具
│   ├── datasets.py               # 数据集构建、增强、CIFAR-10 不平衡采样
│   └── models.py                 # Small CNN / ResNet18
├── requirements.txt
└── README.md
```

## 2. 环境要求

- Python：建议 `3.10+`，当前已在 `conda` 环境 `ML` 中安装依赖。
- 当前 `ML` 环境已验证可用的关键版本：
  - `torch 2.5.1+cu121`
  - `torchvision 0.20.1+cu121`
- 脚本默认 `--device auto`，会优先使用 `CUDA`。如果你要指定某张卡，建议配合 `CUDA_VISIBLE_DEVICES` 使用。
- 训练脚本默认已启用以下提速项：
  - CUDA AMP 混合精度
  - `channels_last`
  - `pin_memory`
  - 更高的 `num_workers`
  - `persistent_workers`
  - TF32 matmul / cuDNN benchmark
- 推荐依赖：

```bash
conda activate ML
pip install -r /usr1/home/s125mdg56_02/mrc/cat_vs_dog/requirements.txt
```

如果你已经按我的操作在 `ML` 环境中安装过依赖，可以直接使用下面命令运行脚本。

例如，只使用一张指定 GPU：

```bash
conda activate ML
CUDA_VISIBLE_DEVICES=0 python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py --device cuda
```

## 2.1 一键自动化运行整个项目

如果你的目标是“自动完成训练并整理出适合写报告的结果”，优先使用：

```bash
conda activate ML
CUDA_VISIBLE_DEVICES=0 python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/run_project_pipeline.py --device cuda
```

这个脚本会自动完成：

1. Dogs vs Cats 三组实验：
   - `ResNet18 + 预训练 + 数据增强`
   - `ResNet18 + 预训练 + 无增强`
   - `SmallCNN + 数据增强`
2. 自动挑选 Dogs vs Cats 中验证集最优模型。
3. 用最优模型生成测试集 `submission.csv`。
4. 自动导出验证集正确/错误样本分析结果。
5. 运行 CIFAR-10 平衡基线实验。
6. 运行 CIFAR-10 长尾不平衡实验：
   - `class_weight`
   - `weighted_sampler`
   - 可选 `focal_loss`
7. 自动汇总所有实验结果，生成报告草稿文件。

默认输出目录：

```text
/usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/project_pipeline/
├── dogs_resnet18_pretrained_aug/
├── dogs_resnet18_pretrained_noaug/
├── dogs_smallcnn_aug/
├── cifar10_baseline/
├── cifar10_imbalance_class_weight/
├── cifar10_imbalance_weighted_sampler/
├── submission.csv
├── val_predictions.csv
├── analysis/
└── report_artifacts/
    ├── experiment_summary.csv
    ├── report_context.json
    ├── figures_manifest.json
    └── report_summary.md
```

其中最重要的三个汇总文件是：

- `experiment_summary.csv`：所有实验的关键指标总表
- `report_context.json`：机器可读的完整上下文
- `report_summary.md`：可直接作为报告草稿参考的 Markdown 摘要
- `figures_manifest.json`：自动生成的报告图清单与建议用途

此外还会自动生成：

```text
report_artifacts/figures/
├── dogs_training_curves.png
├── dogs_model_comparison.png
├── dogs_confusion_matrix.png
├── dogs_case_gallery.png
├── cifar10_training_curves.png
├── cifar10_experiment_comparison.png
└── cifar10_class_distribution.png
```

这些图可以直接用于报告中的 figure。

建议对应关系：

- `dogs_training_curves.png`
  - 用于说明 Dogs vs Cats 训练过程、收敛情况、是否过拟合
- `dogs_model_comparison.png`
  - 用于回答不同模型和数据增强对验证精度的影响
- `dogs_confusion_matrix.png`
  - 用于分析猫/狗两类主要误判情况
- `dogs_case_gallery.png`
  - 用于展示正确样本与错误样本案例分析
- `cifar10_training_curves.png`
  - 用于比较 CIFAR-10 不同方法的收敛过程
- `cifar10_experiment_comparison.png`
  - 用于回答不同不平衡处理方法对最终结果的影响
- `cifar10_class_distribution.png`
  - 用于展示不平衡设定本身，符合 `(h)` 中对数据不平衡问题的讨论需求

### 两种运行档位

#### 1. 快速验证

用于先检查流程是否能完整跑通：

```bash
CUDA_VISIBLE_DEVICES=0 python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/run_project_pipeline.py \
  --device cuda \
  --profile quick
```

`quick` 档会使用更少的数据和更少的 epoch。

默认规模：

- Dogs vs Cats：训练 `2000` 张，验证 `500` 张
- CIFAR-10：标准训练划分，训练约 `45000` 张，验证 `5000` 张

#### 2. 报告实验

用于正式生成报告所需结果：

```bash
CUDA_VISIBLE_DEVICES=0 python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/run_project_pipeline.py \
  --device cuda \
  --profile report \
  --include-focal
```

针对单张 `NVIDIA RTX A5000 24GB`，自动化脚本默认采用下面的数据规模作为报告实验参数：

- Dogs vs Cats：训练 `16000` 张，验证 `5000` 张
- CIFAR-10：训练约 `45000` 张，验证 `5000` 张

这样设置的考虑是：

- `A5000` 跑 `ResNet18 + 224x224 + batch_size=64` 显存余量充足。
- Dogs vs Cats 自动化流程一次会跑 3 组模型对比，因此训练集不直接拉满到 `20000` 张，而是默认取 `16000` 张，兼顾训练时间和报告可信度。
- 验证集默认使用全量 `5000` 张，这样验证精度更稳定，更适合在报告中引用。
- CIFAR-10 数据量本身较小，单卡 `A5000` 可以直接使用标准训练/验证划分。

### 常用参数

- `--gpu-id 0`：等价于设置 `CUDA_VISIBLE_DEVICES=0`
- `--skip-existing`：如果某个实验目录里已经存在 `summary.json`，则跳过重跑
- `--include-focal`：额外加入 `focal_loss` 不平衡实验
- `--dogs-epochs` / `--cifar-epochs`：覆盖默认 epoch
- `--dogs-train-limit` / `--dogs-val-limit`：覆盖 Dogs vs Cats 数据量
- `--cifar-imbalance-ratio`：控制 CIFAR-10 长尾比例

## 2.2 如果训练还是慢，优先这样提速

### Dogs vs Cats

优先级最高的提速方式：

1. 先用自动化脚本的 `quick` 档调流程：

```bash
CUDA_VISIBLE_DEVICES=0 python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/run_project_pipeline.py \
  --device cuda \
  --profile quick
```

2. 降低输入分辨率：

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py \
  --device cuda \
  --image-size 160
```

3. 减少训练样本量，例如：

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py \
  --device cuda \
  --train-limit 8000 \
  --val-limit 2000
```

4. 冻结预训练骨干，只训练分类头：

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py \
  --device cuda \
  --model-name resnet18 \
  --pretrained \
  --freeze-backbone
```

### CIFAR-10

- CIFAR-10 本身已经比较快，主要影响因素是 `epochs` 和不平衡实验次数。
- 若只是先产出结果，可先不加 `--include-focal`，只保留 `class_weight` 与 `weighted_sampler` 两种方法，已经满足 PDF 对“至少 2 种方法”的要求。

### 新增的提速参数

`train_dogs_vs_cats.py` 和 `train_cifar10.py` 现在都支持：

- `--disable-amp`：关闭 AMP，默认不要关
- `--disable-channels-last`：关闭 `channels_last`
- `--disable-persistent-workers`：关闭持久 worker
- `--num-workers 8`
- `--prefetch-factor 4`
- `--fast-mode`
- `--deterministic`

建议：

- 想快：直接用默认设置或加 `--fast-mode`
- 想更严格复现：加 `--deterministic`

## 3. Dogs vs Cats 对应作业要求

本部分对应 PDF 中的第 1 到 5 步，以及报告里的以下内容：

- 数据读取与预处理：`train_dogs_vs_cats.py`
- 数据增强：内置 `RandomResizedCrop`、`RandomHorizontalFlip`、`RandomRotation`、`ColorJitter`
- 模型训练与验证：`train_dogs_vs_cats.py`
- 测试集预测与 `submission.csv`：`predict_dogs_vs_cats.py`
- 正确/错误样本分析：`analyze_dogs_vs_cats.py`

### 3.1 训练 Dogs vs Cats

推荐先用预训练 `ResNet18` 作为 baseline，这个更符合课程项目里“CNN / pretrained backbone”的要求。

```bash
conda activate ML
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/datasets \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18 \
  --model-name resnet18 \
  --pretrained \
  --batch-size 64 \
  --epochs 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --image-size 224 \
  --seed 42
```

如果机器资源有限，可以只用部分数据训练，这也符合 PDF 中“可以根据计算资源决定使用多少训练/验证图像”的说明：

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_dogs_vs_cats.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/datasets \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_small \
  --model-name resnet18 \
  --pretrained \
  --train-limit 4000 \
  --val-limit 1000 \
  --epochs 8
```

训练输出内容：

- `best.pt`：验证集准确率最高的模型
- `last.pt`：最后一个 epoch 的模型
- `history.csv`：每个 epoch 的训练/验证指标
- `summary.json`：最佳验证精度、训练规模等摘要
- `metadata.json`：运行参数和类别映射

### 3.2 生成测试集 `submission.csv`

作业要求的最终提交文件是两列：`ID` 和 `label`，其中 `1=dog`，`0=cat`。本脚本默认按该格式输出。

```bash
conda activate ML
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/predict_dogs_vs_cats.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/datasets \
  --checkpoint /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18/best.pt \
  --output-csv /usr1/home/s125mdg56_02/mrc/cat_vs_dog/submission.csv \
  --split test
```

输出文件示例：

```csv
ID,label
1,0
2,1
3,1
...
```

### 3.3 导出验证集预测结果

如果你想在带标签的验证集上查看每张图像的预测结果，可以切换成 `val`：

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/predict_dogs_vs_cats.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/datasets \
  --checkpoint /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18/best.pt \
  --output-csv /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18/val_predictions.csv \
  --split val
```

这会输出：

- `path`
- `true_label`
- `pred_label`
- `confidence`
- `correct`

### 3.4 导出正确/错误样本用于报告分析

PDF 要求分析正确分类与错误分类样本。由于测试集没有公开标签，定量误分类分析应基于验证集完成。本脚本会从验证集中导出高置信度正确样本和错误样本，便于你挑选 1-2 个案例写报告。

```bash
conda activate ML
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/analyze_dogs_vs_cats.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/datasets \
  --checkpoint /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18/best.pt \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/dogs_vs_cats_resnet18/analysis \
  --top-k 12
```

输出内容：

- `analysis/correct/`：高置信度正确样本
- `analysis/incorrect/`：高置信度错误样本
- `analysis/val_predictions.csv`：验证集全量预测记录
- `analysis/summary.json`：统计摘要

## 4. CIFAR-10 对应作业要求

PDF 中的 `(g)` 和 `(h)` 由 `train_cifar10.py` 统一完成：

- `(g)` 多类别 CIFAR-10 分类：`--imbalance none`
- `(h)` 类别不平衡：`--imbalance long_tail`，并用 `--mitigation` 指定改进方法

脚本会自动下载 CIFAR-10，并把训练集划分为训练/验证集，输出验证集和测试集指标。

### 4.1 CIFAR-10 常规多分类

```bash
conda activate ML
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_cifar10.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/data/cifar10 \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/cifar10_baseline \
  --model-name resnet18 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 42
```

### 4.2 CIFAR-10 不平衡实验

`--imbalance long_tail --imbalance-ratio 0.1` 表示构造长尾训练集，尾部类别样本数约为头部类别的 10%。

#### 方法 1：Class Weight

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_cifar10.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/data/cifar10 \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/cifar10_imbalance_class_weight \
  --model-name resnet18 \
  --imbalance long_tail \
  --imbalance-ratio 0.1 \
  --mitigation class_weight \
  --epochs 20
```

#### 方法 2：Weighted Sampler

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_cifar10.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/data/cifar10 \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/cifar10_imbalance_sampler \
  --model-name resnet18 \
  --imbalance long_tail \
  --imbalance-ratio 0.1 \
  --mitigation weighted_sampler \
  --epochs 20
```

#### 方法 3：Focal Loss

```bash
python /usr1/home/s125mdg56_02/mrc/cat_vs_dog/scripts/train_cifar10.py \
  --data-root /usr1/home/s125mdg56_02/mrc/cat_vs_dog/data/cifar10 \
  --output-dir /usr1/home/s125mdg56_02/mrc/cat_vs_dog/runs/cifar10_imbalance_focal \
  --model-name resnet18 \
  --imbalance long_tail \
  --imbalance-ratio 0.1 \
  --mitigation focal_loss \
  --epochs 20
```

输出内容与 Dogs vs Cats 类似：

- `best.pt`
- `last.pt`
- `history.csv`
- `summary.json`
- `metadata.json`

其中：

- `summary.json` 会记录最佳验证精度、最终测试精度、训练集类别分布
- `metadata.json` 会记录模型参数、类别名和实验设置

## 5. 脚本参数总览

### `train_dogs_vs_cats.py`

常用参数：

- `--data-root`：Dogs vs Cats 数据目录
- `--output-dir`：模型与日志输出目录
- `--model-name`：`small_cnn` 或 `resnet18`
- `--pretrained`：是否启用 ImageNet 预训练
- `--freeze-backbone`：是否冻结 `ResNet18` 主干，只训练分类头
- `--train-limit` / `--val-limit`：限制训练/验证样本数
- `--epochs` / `--batch-size` / `--lr` / `--weight-decay`
- `--image-size`
- `--seed`

### `predict_dogs_vs_cats.py`

常用参数：

- `--checkpoint`：训练得到的 `best.pt` 或 `last.pt`
- `--split test`：导出最终 `submission.csv`
- `--split val`：导出验证集预测细节
- `--output-csv`：输出 CSV 路径

### `analyze_dogs_vs_cats.py`

常用参数：

- `--checkpoint`
- `--output-dir`
- `--top-k`：导出的高置信度正确/错误样本数量

### `train_cifar10.py`

常用参数：

- `--model-name`：`small_cnn` 或 `resnet18`
- `--imbalance`：`none` 或 `long_tail`
- `--imbalance-ratio`：长尾比例
- `--mitigation`：`none`、`class_weight`、`weighted_sampler`、`focal_loss`
- `--val-size`：从 CIFAR-10 官方训练集里切出的验证集大小
- `--epochs` / `--batch-size` / `--lr` / `--weight-decay`

## 6. 建议实验顺序

1. 先跑 `train_dogs_vs_cats.py` 得到 baseline。
2. 用 `predict_dogs_vs_cats.py --split test` 生成 `submission.csv`。
3. 用 `analyze_dogs_vs_cats.py` 导出正确/错误样本，完成报告分析部分。
4. 跑 `train_cifar10.py --imbalance none`，完成 CIFAR-10 多分类基线。
5. 跑至少两组 `--imbalance long_tail` + 不同 `--mitigation`，完成不平衡实验对比。

## 7. 报告撰写对应建议

你可以直接利用脚本输出内容完成 PDF 中的大部分实验部分：

- 数据规模：`summary.json` / `metadata.json`
- 训练策略与参数：命令行参数 + `metadata.json`
- 验证集准确率：`history.csv` 和 `summary.json`
- 测试集结果：`submission.csv`
- 正确/错误案例分析：`analysis/` 导出样本
- 不同模型/数据处理的影响：对比不同输出目录中的 `history.csv` 与 `summary.json`

## 8. 说明

- Dogs vs Cats 测试集无标签，因此“错误样本分析”建议基于验证集完成，测试集只做最终预测提交。
- 如果你要比较不同模型，只需要修改 `--model-name`、`--pretrained` 和训练参数即可。
- 默认随机种子为 `42`，脚本已显式固定随机种子，便于复现。
