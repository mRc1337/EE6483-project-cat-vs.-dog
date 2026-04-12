# EE6483 Artificial Intelligence and Data Mining
# Mini Project (Option 2) Report

## Dogs vs. Cats Image Classification and CIFAR-10 Extension

### Group Information

- Group member 1: `待填写`
- Group member 2: `待填写`
- Group member 3: `待填写`

### Contribution Statement

- Member 1: `待填写`
- Member 2: `待填写`
- Member 3: `待填写`

---

## Abstract

本报告围绕 EE6483 Mini Project Option 2 的要求，完成了 Dogs vs. Cats 二分类任务，并进一步将分类算法扩展到 CIFAR-10 多类别图像分类与类别不平衡场景。针对 Dogs vs. Cats，本文构建了三组可复现实验：`ResNet18 + ImageNet 预训练 + 数据增强`、`ResNet18 + ImageNet 预训练 + 无增强`、`SmallCNN + 数据增强`。在单卡 `NVIDIA RTX A5000 24GB` 上，最终最优配置为 `ResNet18 + ImageNet 预训练 + 数据增强`，在验证集上达到 `98.52%` 的准确率，并生成了符合要求的 `submission.csv`。随后，本文在 CIFAR-10 上实现了平衡数据基线以及长尾不平衡设定，并比较了 `class_weight`、`weighted_sampler` 与 `focal_loss` 三种不平衡缓解方法。实验表明，迁移学习与数据增强对 Dogs vs. Cats 任务均有显著作用；在 CIFAR-10 上，类别不平衡会显著降低性能，而 `class_weight` 与 `weighted_sampler` 在当前设定下均能将测试准确率稳定在 `82.67%`。报告进一步通过混淆矩阵和案例图分析模型的优势与局限，并给出对模型选择、数据处理和后续改进方向的总结。

---

## 1. Literature Survey

### 1.1 Problem Definition

本项目的核心问题属于**监督式图像分类**。输入是一张 RGB 图像，输出是离散类别标签。在 Dogs vs. Cats 任务中，标签空间为二元闭集分类：`dog` 与 `cat`；在 CIFAR-10 扩展实验中，标签空间扩展为 `10` 个互斥类别。对于本项目而言：

- Dogs vs. Cats 是**监督学习、闭集识别、同域训练与测试**的标准图像分类问题。
- CIFAR-10 也是**监督学习、闭集识别**问题，但更强调多类别判别能力。
- 若考虑实际部署，则还会涉及**开放集识别**与**域偏移**问题，例如训练集和测试集在分辨率、光照、背景或拍摄设备上不一致时，模型泛化能力会下降。

### 1.2 Main Challenges

尽管猫狗分类对人类直觉上较容易，但对计算机视觉模型而言仍存在若干挑战：

1. **类内变化大**：不同品种的猫和狗在毛色、体型、耳朵形态、脸部特征上变化很大。
2. **类间相似性强**：某些短毛犬、长毛猫、侧脸图或低分辨率图像在局部纹理上较为相似。
3. **背景偏差**：模型可能错误依赖室内场景、地毯、沙发、草地等背景信息，而非动物主体。
4. **姿态与遮挡**：侧脸、俯拍、背面或被家具遮挡的图像会降低判别稳定性。
5. **尺度与分辨率问题**：动物主体占图比例较小时，局部细节不足会导致误判。
6. **类别不平衡问题**：在 CIFAR-10 扩展实验中，长尾分布会使尾部类别难以充分学习。

### 1.3 Common Solution Types Under Different Settings

在图像分类任务中，常见方案可以分为以下几类：

- **传统机器学习方法**：先提取手工特征，如 SIFT、HOG，再结合 SVM、Random Forest 等分类器。这类方法在小规模数据集上可用，但对复杂语义模式的表达能力有限。
- **卷积神经网络（CNN）**：如 AlexNet、VGG、ResNet、DenseNet、EfficientNet 等，通过端到端学习层级视觉特征，已经成为图像分类主流方法。
- **迁移学习**：使用 ImageNet 预训练模型作为特征提取骨干，在中小数据集上通常能显著提高精度并加速收敛。
- **Transformer/混合架构**：如 ViT、Swin Transformer、ConvNeXt 等，在大规模预训练条件下取得了极强性能。
- **自监督学习与多模态预训练**：如 MAE、CLIP 等，可通过无标注或图文对数据获得更泛化的视觉表示。
- **不平衡学习方法**：包括类别加权、重采样、focal loss、class-balanced loss、two-stage re-training 等。

从设置角度看：

- **监督 vs. 无监督/自监督**：本项目为监督学习，但近期主流方法普遍利用自监督或大规模预训练表征提升下游精度。
- **闭集 vs. 开放集**：本项目假设测试图像必然属于已知类别；若测试集中出现狐狸、狼、玩具狗等未知类别，则需要开放集识别或 OOD 检测。
- **无域偏移 vs. 有域偏移**：本项目数据整体与目标任务一致；若训练集为高清宠物照片而测试集为低分辨率监控图像，则需要域泛化或域自适应方法。

### 1.4 Representative and Influential Works

以下工作对当前图像分类领域影响较大：

1. **VGGNet** 提出使用连续小卷积核构建深层网络，推动了深层 CNN 的标准化设计。
2. **ResNet** 引入残差连接，显著缓解了深层网络训练困难问题，成为图像分类最常用的骨干之一。
3. **EfficientNet** 通过复合缩放策略在精度与计算量之间取得较优平衡。
4. **Vision Transformer (ViT)** 将 Transformer 架构引入图像分类，在大规模预训练条件下取得强性能。
5. **ConvNeXt** 将现代训练策略与卷积结构结合，表明纯卷积架构在现代配方下仍具竞争力。
6. **MAE** 通过遮蔽自编码预训练提升视觉表征质量。
7. **CLIP** 利用大规模图文对训练通用视觉表征，展示了强泛化能力。
8. **Focal Loss** 在类别不平衡和难样本学习中被广泛采用。

### 1.5 Selected Works and Detailed Review

本报告重点关注两条与本项目最相关的路线：**残差网络迁移学习**与**现代预训练视觉模型**。

#### (1) ResNet for Transfer Learning

ResNet 的核心思想是在卷积块中引入恒等映射和残差学习，使网络优化目标从直接拟合映射变为拟合残差。对于本项目这种中等规模的分类任务，ResNet18 有三点优势：

- 参数量和计算量适中，适合单卡实验与多组对比。
- ImageNet 预训练权重通用性强，对猫狗这类自然图像迁移效果好。
- 网络结构成熟，便于对比数据增强、冻结骨干、不同损失函数等因素。

#### (2) Modern Large-Scale Pretraining

ViT、MAE 和 CLIP 等工作显示，大规模预训练能为下游任务提供更强鲁棒性和泛化能力。尽管这些模型通常能取得更高上限，但其计算成本、调参复杂度和推理开销也更大。对于本项目，目标不是追求 SOTA，而是在课程作业约束下实现可解释、可复现、可比较的方案，因此选择 ResNet18 作为主基线更加合理。

### 1.6 Baseline Choice and Motivation

综合以上分析，本文选择 `ResNet18 + ImageNet 预训练 + 线性分类头` 作为 Dogs vs. Cats 的主基线，原因如下：

1. 该方案与课程要求中提到的“pretrained feature extraction backbone”完全一致。
2. 在中小规模数据集上，迁移学习通常比从零训练 CNN 更稳定。
3. ResNet18 训练速度快、显存占用适中，适合在单卡 `A5000` 上完成多组实验。
4. 作为扩展到 CIFAR-10 的基础架构时，也可以通过修改 stem 和分类头实现平滑迁移。

---

## 2. Data Usage, Pre-processing and Augmentation

### 2.1 Dogs vs. Cats Dataset Usage

原始数据集结构如下：

- `train`: 每类 `10000` 张，共 `20000` 张
- `val`: 每类 `2500` 张，共 `5000` 张
- `test`: 无标签图像 `500` 张

考虑到自动化脚本需要连续完成多组对比实验，同时兼顾报告结果可靠性与单卡计算预算，本项目在 Dogs vs. Cats 上采用以下默认规模：

- 训练集：从原始 `train` 中随机抽取 `16000` 张
  - 基于固定随机种子 `42` 的实际子集分布约为 `8005` 张猫、`7995` 张狗
- 验证集：使用完整 `5000` 张验证图像
  - `2500` 张猫、`2500` 张狗
- 测试集：使用完整 `500` 张无标签图像生成 `submission.csv`

这种设置能够：

- 保证验证集评估足够稳定；
- 保留训练集的大部分样本，避免因为极小样本而影响结果可信度；
- 控制自动化流程总训练时长，使多组实验可在单卡上完成。

### 2.2 CIFAR-10 Dataset Usage

对于 CIFAR-10：

- 官方训练集：`50000` 张
- 官方测试集：`10000` 张
- 本项目从训练集中划出 `5000` 张作为验证集
- 平衡基线：训练集 `45000` 张，验证集 `5000` 张
- 长尾不平衡实验：在训练集上构造长尾分布，最终训练集大小为 `18260` 张

### 2.3 Pre-processing

#### Dogs vs. Cats

- 输入尺寸：`224 × 224`
- 标准化均值与方差：使用 ImageNet 统计量
- 验证/测试阶段：
  - `Resize(224, 224)`
  - `ToTensor()`
  - `Normalize(mean, std)`

#### CIFAR-10

- 输入尺寸：`32 × 32`
- 标准化均值与方差：使用 CIFAR-10 常用统计量
- 验证/测试阶段：
  - `ToTensor()`
  - `Normalize(mean, std)`

### 2.4 Data Augmentation

Dogs vs. Cats 训练阶段增强包括：

- `Resize(image_size + 32, image_size + 32)`
- `RandomResizedCrop(224, scale=(0.75, 1.0))`
- `RandomHorizontalFlip()`
- `RandomRotation(15°)`
- `ColorJitter(brightness, contrast, saturation)`

CIFAR-10 训练阶段增强包括：

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`

数据增强的目的在于减轻过拟合，增加模型对平移、尺度、姿态与颜色变化的鲁棒性。后续实验中也专门比较了“有增强”与“无增强”的影响。

---

## 3. Model Design and Training Strategy

### 3.1 Dogs vs. Cats Models

本项目在 Dogs vs. Cats 上比较了两类模型：

#### Model A: ResNet18 + Pretraining

输入输出关系：

- Input: `3 × 224 × 224`
- Backbone: `ResNet18`
- Global Average Pooling
- Fully Connected Layer: `512 -> 2`
- Output: `2` 维 logits，对应 `cat` 与 `dog`

简化结构示意：

```text
RGB Image (3x224x224)
    -> ResNet18 Stem
    -> Residual Block Group 1
    -> Residual Block Group 2
    -> Residual Block Group 3
    -> Residual Block Group 4
    -> Global Average Pooling
    -> Linear(512, 2)
    -> Softmax / Predicted Label
```

#### Model B: SmallCNN

输入输出关系：

- Input: `3 × 224 × 224`
- Conv(3->32) + BN + ReLU + MaxPool
- Conv(32->64) + BN + ReLU + MaxPool
- Conv(64->128) + BN + ReLU + MaxPool
- Conv(128->256) + BN + ReLU + AdaptiveAvgPool
- FC(256 -> 128 -> 2)
- Output: `2` 维 logits

该模型作为自建 CNN baseline，用于和预训练 ResNet18 做能力对比。

### 3.2 CIFAR-10 Model

在 CIFAR-10 上使用修改后的 ResNet18：

- Input: `3 × 32 × 32`
- 将初始 `7×7, stride=2` 卷积替换为 `3×3, stride=1`
- 去除初始最大池化
- 最终分类头改为 `Linear(512, 10)`

这种修改更适合 CIFAR-10 的小分辨率输入。

### 3.3 Loss Functions

- Dogs vs. Cats：`CrossEntropyLoss`
- CIFAR-10 baseline：`CrossEntropyLoss`
- CIFAR-10 imbalance：
  - `class_weight`: 带类别权重的 `CrossEntropyLoss`
  - `weighted_sampler`: 重采样 + `CrossEntropyLoss`
  - `focal_loss`: `FocalLoss(gamma=2.0, alpha=class_weights)`

### 3.4 Optimization Strategy

统一训练策略如下：

- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Scheduler: `CosineAnnealingLR`
- Early stopping:
  - Dogs vs. Cats: `patience = 3`
  - CIFAR-10: `patience = 5`
- Random seed: `42`

### 3.5 Runtime and Reproducibility

为了兼顾速度与复现性，本项目使用：

- 单卡 `NVIDIA RTX A5000 24GB`
- `batch_size = 64`（Dogs vs. Cats）
- `batch_size = 128`（CIFAR-10）
- `num_workers = 8`
- AMP mixed precision
- `channels_last`
- `persistent_workers`
- 固定随机种子 `42`

尽管开启了 AMP 和高性能数据加载设置，模型输入划分、随机种子和训练配置仍然被完整保存在 `metadata.json` 中，保证实验可追溯。

---

## 4. Parameter Selection and Rationale

### 4.1 Why `epoch = 10` for Dogs vs. Cats

Dogs vs. Cats 默认最大训练轮数设为 `10`，原因如下：

1. 主模型使用 `ImageNet` 预训练的 ResNet18，前期收敛较快；
2. 本项目需要比较三组 Dogs vs. Cats 模型，而不仅是单组训练；
3. 在单卡 A5000 上，`10` 个 epoch 已能将预训练 ResNet18 的验证精度推到接近 `98.5%`；
4. 课程项目目标是完成高质量对比分析，而不是单模型极限调参。

从训练曲线来看，在第 `10` 个 epoch 时模型仍有轻微提升空间，但性能已处于高位，因此该设置在时间与效果之间较为平衡。

### 4.2 Why `epoch = 20` for CIFAR-10

CIFAR-10 不使用 ImageNet 预训练，因此相较 Dogs vs. Cats 需要更多训练轮数以获得稳定收敛。`20` 个 epoch 在本项目设置下能够使：

- 平衡基线达到 `92.24%` 的验证准确率；
- 长尾不平衡实验达到 `81%~83%` 的测试准确率区间；
- 不同方法之间的差异充分显现，便于报告比较。

### 4.3 Why ResNet18 Instead of a Larger Model

虽然更大的模型（如 ResNet50、EfficientNet-B3、ViT）可能具有更高上限，但本项目选择 ResNet18 主要基于：

- 单卡训练时间可控；
- 在中等规模数据集上已经足够强；
- 便于完成多组模型和多种不平衡方法的对比；
- 结构成熟、结果稳定、易于解释。

### 4.4 Why These Imbalance Methods

对于 CIFAR-10 长尾设定，本报告选用三种代表性方法：

- `class_weight`：直接在损失层增加尾部类别权重；
- `weighted_sampler`：在 mini-batch 采样阶段提升尾部类别出现频率；
- `focal_loss`：强调难样本并抑制易样本损失占比。

其中前两种已经满足作业要求中的“至少两种方法”，第三种作为补充方法，用于观察更复杂损失函数在当前场景中的表现。

---

## 5. Experimental Results on Dogs vs. Cats

### 5.1 Quantitative Results

表 1 给出了 Dogs vs. Cats 三组实验的验证集结果。

| Experiment | Model | Pretrained | Augmentation | Train Size | Val Size | Best Val Accuracy |
| --- | --- | --- | --- | ---: | ---: | ---: |
| dogs_resnet18_pretrained_aug | ResNet18 | Yes | Yes | 16000 | 5000 | 98.52% |
| dogs_resnet18_pretrained_noaug | ResNet18 | Yes | No | 16000 | 5000 | 98.00% |
| dogs_smallcnn_aug | SmallCNN | No | Yes | 16000 | 5000 | 81.92% |

结果说明：

1. **ResNet18 + 预训练 + 数据增强** 达到最佳结果，是最终用于测试集提交的模型。
2. 在相同模型下，**加入数据增强将验证准确率从 98.00% 提高到 98.52%**，提升 `0.52` 个百分点。
3. 与预训练 ResNet18 相比，**SmallCNN 的验证准确率低了 16.60 个百分点**，表明预训练特征对该任务帮助极大。

### 5.2 Training Dynamics

图 1 展示了 Dogs vs. Cats 的训练曲线：

![](runs/project_pipeline/report_artifacts/figures/dogs_training_curves.png)

从图中可以观察到：

- 预训练 ResNet18 在第 1 个 epoch 就达到较高精度，说明迁移学习有效；
- 无增强版本训练精度很快逼近 `100%`，但验证精度略低于增强版本，显示出更明显的过拟合趋势；
- SmallCNN 的收敛明显更慢，且最终上限较低。

图 2 给出了三组模型在验证集上的最佳精度对比：

![](runs/project_pipeline/report_artifacts/figures/dogs_model_comparison.png)

### 5.3 Final Validation Accuracy and Test Submission

最优 Dogs vs. Cats 模型为：

- Experiment ID: `dogs_resnet18_pretrained_aug`
- Best validation accuracy: `98.52%`
- Best epoch: `10`

测试集预测文件已生成：

- `submission.csv`: `runs/project_pipeline/submission.csv`

该文件包含两列：

- `ID`
- `label` (`1 = dog`, `0 = cat`)

满足 PDF 中的提交格式要求。

### 5.4 Confusion Matrix Analysis

根据导出的验证集预测结果，最优模型的混淆矩阵为：

- True cat -> Pred cat: `2465`
- True cat -> Pred dog: `35`
- True dog -> Pred dog: `2462`
- True dog -> Pred cat: `38`

图 3 展示了对应混淆矩阵：

![](runs/project_pipeline/report_artifacts/figures/dogs_confusion_matrix.png)

可以看到：

- 两个类别的识别能力较为均衡；
- 猫误判为狗和狗误判为猫的数量都很少；
- 模型不存在明显偏向某一类的问题。

### 5.5 Case Study: Correctly and Incorrectly Classified Samples

图 4 给出了验证集中的高置信度正确样本与错误样本：

![](runs/project_pipeline/report_artifacts/figures/dogs_case_gallery.png)

结合实际样本，可得到以下结论：

#### Correct Case 1: `cat.100.jpg`

该图像中的小猫正面居中，耳朵、眼睛、胡须和面部轮廓都非常清晰，背景简单，主体占比较高，因此模型能够以接近 `100%` 的置信度正确分类。

#### Correct Case 2: `dog.10029.jpg`

该图像中狗的面部特征、耳朵形态与项圈都十分明显，构图集中，背景干扰很低，因此模型同样能以极高置信度正确分类。

#### Incorrect Case 1: `dog.9109.jpg`

该图像是一只灰色犬只躺在沙发上，但其身体轮廓与毛发纹理较为紧凑，面部较小且被环境纹理部分干扰。模型将其预测为猫，说明当主体尺度较小、背景复杂、面部信息不突出时，模型更容易受到整体纹理和轮廓误导。

#### Incorrect Case 2: `cat.7323.jpg`

该图像中的猫为近距离局部拍摄，存在较强光照、低分辨率与非常规姿态，其深色面部和局部特征使模型高置信度地误判为狗。该案例说明，模型对特殊姿态、模糊细节和局部放大图仍存在脆弱性。

### 5.6 Strengths and Weaknesses of the Model

**Strengths**

- 在大多数标准宠物照片上表现稳定；
- 对正面主体、清晰轮廓、正常光照场景有很强判别力；
- 借助预训练与增强后，对类别间差异学习较充分。

**Weaknesses**

- 对特殊品种、非典型姿态、遮挡、复杂背景和低分辨率样本仍有误判；
- 在高置信度误判样本中，模型可能过度依赖局部纹理或整体轮廓；
- 目前仍属于封闭集分类，不具备处理未知类别的能力。

### 5.7 Effect of Model Choice and Data Processing

该部分对应 PDF 中关于“不同模型和数据处理如何影响验证精度”的问题。

#### Effect of Data Augmentation

在同为 `ResNet18 + 预训练` 的前提下：

- 有增强：`98.52%`
- 无增强：`98.00%`

说明数据增强能够带来稳定提升，并降低过拟合倾向。

#### Effect of Model Choice

在均使用训练集 `16000` 张的条件下：

- `ResNet18 + 预训练 + 增强`：`98.52%`
- `SmallCNN + 增强`：`81.92%`

说明在该任务中，预训练骨干网络远强于从零训练的小型 CNN，这与视觉迁移学习的常见结论一致。

---

## 6. CIFAR-10 Multi-class Classification

### 6.1 Problem Description

CIFAR-10 是一个 `10` 类小尺寸彩色图像分类数据集，类别包括：

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

与 Dogs vs. Cats 相比，CIFAR-10 的主要变化在于：

1. 从二分类扩展为多分类；
2. 输入分辨率更低（`32×32`）；
3. 类别数更多，类间差异更复杂；
4. 不平衡实验还要求模型对尾部类别保持鲁棒性。

### 6.2 Changes Made from Dogs vs. Cats to CIFAR-10

为了适配 CIFAR-10，本项目做了如下修改：

- 分类头由 `2` 类输出改为 `10` 类输出；
- ResNet18 的 stem 修改为 `3×3, stride=1`，并移除初始 max-pooling；
- 数据增强策略改为更适合小图像的 `RandomCrop + HorizontalFlip`；
- 训练轮数从 `10` 增加为 `20`；
- 在不平衡实验中加入类别加权、重采样与 focal loss。

### 6.3 Balanced CIFAR-10 Baseline

表 2 为 CIFAR-10 平衡训练集基线结果。

| Experiment | Train Size | Val Size | Test Size | Best Val Accuracy | Final Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| cifar10_baseline | 45000 | 5000 | 10000 | 92.24% | 91.24% |

图 5 给出了 CIFAR-10 各实验的训练/测试曲线：

![](runs/project_pipeline/report_artifacts/figures/cifar10_training_curves.png)

该结果表明，所实现的 ResNet18 变体能够在标准 CIFAR-10 上取得较强的基线性能。

---

## 7. CIFAR-10 with Class Imbalance

### 7.1 Long-tail Setting

按照作业要求，本项目构造了 CIFAR-10 的长尾训练集。长尾比设为 `0.1`，即尾部类别样本量约为头部类别的 `10%`。最终训练集类别分布如下：

| Class ID | Count |
| ---: | ---: |
| 0 | 4468 |
| 1 | 3459 |
| 2 | 2678 |
| 3 | 2074 |
| 4 | 1606 |
| 5 | 1243 |
| 6 | 963 |
| 7 | 745 |
| 8 | 577 |
| 9 | 447 |

图 6 展示了平衡训练集与长尾训练集的类别分布对比：

![](runs/project_pipeline/report_artifacts/figures/cifar10_class_distribution.png)

### 7.2 Two or More Methods for Handling Imbalance

本报告比较三种方法：

1. **Class Weight**
2. **Weighted Sampler**
3. **Focal Loss**

其中前两种已经满足作业要求的“至少 2 种方法”。

### 7.3 Results

表 3 为 CIFAR-10 不平衡实验结果。

| Experiment | Imbalance | Method | Train Size | Best Val Accuracy | Final Test Accuracy |
| --- | --- | --- | ---: | ---: | ---: |
| cifar10_baseline | None | None | 45000 | 92.24% | 91.24% |
| cifar10_imbalance_class_weight | Long-tail | Class Weight | 18260 | 82.74% | 82.67% |
| cifar10_imbalance_weighted_sampler | Long-tail | Weighted Sampler | 18260 | 82.74% | 82.67% |
| cifar10_imbalance_focal_loss | Long-tail | Focal Loss | 18260 | 81.46% | 81.40% |

图 7 给出了这些方法的精度对比：

![](runs/project_pipeline/report_artifacts/figures/cifar10_experiment_comparison.png)

### 7.4 Discussion

可以观察到：

1. 从平衡数据到长尾数据，测试精度从 `91.24%` 下降到约 `82.67%`，说明类别不平衡对多分类性能有显著负面影响。
2. 在本实验设定下，`class_weight` 与 `weighted_sampler` 取得了几乎相同的验证/测试精度，说明两者都能有效缓解长尾分布带来的偏差。
3. `weighted_sampler` 的训练精度高于 `class_weight`，但最终验证/测试精度没有进一步提高，说明其对训练集拟合更强，却未明显改善泛化。
4. `focal_loss` 在当前设定下略低于前两种方法，说明更复杂的损失函数并不一定优于简单且稳定的重加权/重采样方法。

### 7.5 Why These Methods Improve Imbalanced Learning

#### Method 1: Class Weight

类别加权通过提高尾部类别样本在损失函数中的权重，使模型在优化时对少数类错误更敏感，从而减轻头部类别主导训练的问题。

#### Method 2: Weighted Sampler

重采样通过在 mini-batch 层面提升尾部类别出现频率，使模型在训练中更频繁地看到少数类样本，从而改善特征学习覆盖度。

#### Method 3: Focal Loss

Focal Loss 会降低易分类样本的损失占比，并将优化重点放在难样本上，这在类别不平衡和难样本较多时可能有效。不过在本实验中，其优势没有明显体现。

---

## 8. Discussion

综合本项目的两部分实验，可以得到以下结论：

1. **迁移学习对中等规模自然图像分类非常有效。**  
   在 Dogs vs. Cats 上，预训练 ResNet18 与自建 SmallCNN 相比优势极为明显。

2. **数据增强虽然带来的提升不如换模型那样巨大，但提升是稳定且值得的。**  
   在相同模型下，增强版本比无增强版本验证准确率更高，并表现出更好的泛化能力。

3. **模型能力和数据分布同样重要。**  
   Dogs vs. Cats 主要受益于强骨干与迁移学习；CIFAR-10 的主要问题则是多类别复杂度与长尾不平衡。

4. **类别不平衡会显著降低性能。**  
   CIFAR-10 的结果显示，仅改变训练集分布就会造成约 `8~10` 个百分点的准确率损失。

5. **在当前场景下，简单方法已经很有效。**  
   `class_weight` 与 `weighted_sampler` 的效果与更复杂的 focal loss 相当甚至更好，因此在课程项目中具有更高性价比。

---

## 9. Conclusion

本文完成了 EE6483 Mini Project Option 2 的主要任务，并得出如下结论：

1. 在 Dogs vs. Cats 上，`ResNet18 + ImageNet 预训练 + 数据增强` 是最优配置，在验证集上达到 `98.52%`。
2. 数据增强带来稳定收益，而预训练骨干相较自建小型 CNN 具有压倒性优势。
3. 测试集预测结果已按要求导出为 `submission.csv`。
4. 在 CIFAR-10 上，平衡数据基线达到 `91.24%` 的测试准确率。
5. 在 CIFAR-10 长尾不平衡设定下，`class_weight` 与 `weighted_sampler` 均达到 `82.67%` 的测试准确率，优于 `focal_loss`。
6. 项目结果表明：对于中小规模视觉分类任务，迁移学习、合理增强和简单稳定的不平衡策略，通常是最实用且高效的解决方案。

未来可以进一步尝试：

- 更强的预训练骨干，如 EfficientNet 或 ConvNeXt；
- 更细粒度的数据增强，如 RandAugment、MixUp、CutMix；
- 针对长尾问题采用 class-balanced loss、deferred re-weighting 或 two-stage fine-tuning；
- 在 Dogs vs. Cats 上引入开放集/OOD 检测，以提高真实场景鲁棒性。

---

## 10. Reproducibility and File Locations

主要输出文件如下：

- 最终 Dogs vs. Cats 提交文件：`runs/project_pipeline/submission.csv`
- Dogs vs. Cats 验证预测：`runs/project_pipeline/val_predictions.csv`
- Dogs vs. Cats 样例分析：`runs/project_pipeline/analysis/`
- 自动化实验总表：`runs/project_pipeline/report_artifacts/experiment_summary.csv`
- 自动化摘要：`runs/project_pipeline/report_artifacts/report_summary.md`
- 自动生成图清单：`runs/project_pipeline/report_artifacts/figures_manifest.json`

主要报告图位置：

- `runs/project_pipeline/report_artifacts/figures/dogs_training_curves.png`
- `runs/project_pipeline/report_artifacts/figures/dogs_model_comparison.png`
- `runs/project_pipeline/report_artifacts/figures/dogs_confusion_matrix.png`
- `runs/project_pipeline/report_artifacts/figures/dogs_case_gallery.png`
- `runs/project_pipeline/report_artifacts/figures/cifar10_training_curves.png`
- `runs/project_pipeline/report_artifacts/figures/cifar10_experiment_comparison.png`
- `runs/project_pipeline/report_artifacts/figures/cifar10_class_distribution.png`

---

## References

[1] Simonyan, K., & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv preprint arXiv:1409.1556*, 2014.

[2] He, K., Zhang, X., Ren, S., & Sun, J. Deep Residual Learning for Image Recognition. *Proceedings of CVPR*, 2016.

[3] Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. *Proceedings of CVPR*, 2009.

[4] Tan, M., & Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of ICML*, 2019.

[5] Dosovitskiy, A., et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *Proceedings of ICLR*, 2021.

[6] Liu, Z., et al. A ConvNet for the 2020s. *Proceedings of CVPR*, 2022.

[7] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. Masked Autoencoders Are Scalable Vision Learners. *Proceedings of CVPR*, 2022.

[8] Radford, A., et al. Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of ICML*, 2021.

[9] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. Focal Loss for Dense Object Detection. *Proceedings of ICCV*, 2017.

[10] He, H., & Garcia, E. A. Learning from Imbalanced Data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284, 2009.

[11] Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto, 2009.

[12] Kaggle. Dogs vs. Cats Competition Dataset. https://www.kaggle.com/c/dogs-vs-cats
