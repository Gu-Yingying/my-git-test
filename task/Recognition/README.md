# 目标识别任务

![CNN 图像分类流程示意图](../../assets/tasks/recognition-cnn-v2.png)

*任务流程示意图，非实际运行结果。*

使用 **Python + PyTorch** 实现并训练一个 CNN 模型，完成图像分类。

## CNN 简介

CNN（Convolutional Neural Network，卷积神经网络）是一类适合处理图像的神经网络。它通过训练学习图像中的特征，例如边缘、纹理和形状，再根据这些特征判断图像类别。

常见的 CNN 分类模型包含以下部分：

- **卷积层**：用可学习的卷积核扫描图像的局部区域，提取特征；同一个卷积核在不同位置共享参数。
- **激活函数**：如 ReLU，引入非线性，使网络能够学习更复杂的特征。
- **池化层**：对局部特征进行汇总，缩小特征图尺寸，减少后续计算量。
- **分类层**：综合提取到的特征，为每个类别输出一个分数，选择分数最高的类别作为预测结果。

在本任务中，输入是一张 CIFAR-10 图像，输出是 10 个类别的分数。训练时，用损失函数衡量预测与真实标签的差异，通过反向传播计算梯度，再由优化器更新网络参数；推理时，使用训练好的参数预测图像类别。

## 数据集

使用 [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)，通过 `torchvision.datasets.CIFAR10` 自动下载和加载。

## 实现要求

- 使用 PyTorch 定义 CNN 模型并编写训练代码。
- 使用 CIFAR-10 的 10 个类别完成分类，训练集用于训练，测试集只用于评估。
- 训练结束后将模型权重保存为 `.pt` 或 `.pth` 文件；评估程序须能重新创建模型、加载该权重并完成推理。
- 评估时输出测试集总样本数、正确分类数和总体准确率，并输出每个类别的准确率。准确率按实际运行结果报告，不预设统一门槛；README 中说明训练轮数、运行设备（CPU 或 GPU）及对应结果。
- 对若干测试集图像输出真实类别和模型预测类别，便于确认程序实际执行了分类推理，而不只是完成训练或保存权重。

## 代码与目录规范

- 代码放在 `task/Recognition/src/`，合理区分模型、训练和评估；任务根目录提供 `test.sh` 和 README。
- 关键步骤有必要注释，数据和权重路径、训练参数便于调整，加载失败时给出明确提示。
- 说明模型结构和数据处理方法，引用代码或模型时注明来源。

## 运行与验收

提供 `test.sh`，从仓库任务目录执行：

```bash
bash test.sh
```

脚本应运行训练和测试评估；评估须加载刚保存的权重。也可以提供独立评估命令，方便在已有权重时重复验证。一次有效的验收运行应能观察到：

1. 训练过程正常结束，并在指定位置生成非空的 `.pt` 或 `.pth` 权重文件。
2. 测试评估成功加载该权重，在 CIFAR-10 测试集上完成推理，并打印总样本数、正确数、总体准确率和各类别准确率。
3. 输出若干样本的真实类别与预测类别，例如 `true=cat, predicted=cat`；预测不必全部正确，README 如实记录运行结果。

## 提交要求

- 提交源码、依赖说明、`test.sh`、任务 README 和实际模型权重；权重通过可访问的下载链接提供。
- README 写明环境、训练与评估命令及执行目录、权重使用方法，以及实际设备、训练轮数、评估结果和未完成项。
- 用 `.gitignore` 排除数据集、权重、虚拟环境和缓存。Git 提交说明具体改动，提交前按 README 检查权重加载与评估能否复现。
- 提交时提供姓名或昵称、仓库链接、分支、提交号和权重链接，验收以该版本为准。

环境配置见[环境准备](../../ENV_SETUP.md)，神经网络与 PyTorch 教程见[学习资料与路径](../../LEARNING_RESOURCES.md)。

[返回培训考核首页](../../README.md)









训练结果：



# 目标识别任务（CIFAR-10 CNN 分类）

使用 PyTorch 实现并训练 CNN，完成 CIFAR-10 十分类。本 README 记录环境、命令与实际运行结果。

## 环境

- 操作系统：Windows（x64）
- Python 3.12.14
- PyTorch 2.14.1 + CUDA（显卡：NVIDIA RTX 5070 Ti）
- torchvision 0.29.1

依赖安装：

```
pip install torch torchvision
```

## 目录结构

```
task/Recognition/
├── src/
│   ├── model.py       # CNN 模型定义
│   ├── train.py       # 训练并保存权重
│   └── evaluate.py    # 加载权重并评估
├── test.sh            # 一键训练+评估
└── README.md
```

## 运行方法

在仓库的 `task/Recognition` 目录下执行：

```bash
bash test.sh
```

或在 `src/` 目录下分步执行：

```bash
cd src
python train.py      # 训练 10 轮，权重保存为 src/cifar10_cnn.pth
python evaluate.py   # 加载权重，在测试集上评估
```

首次运行自动下载 CIFAR-10（约 170MB）到 `src/data/`。

## 模型结构与数据处理方法

- 模型（`src/model.py`）：两层卷积（Conv 3→32→64，各接 ReLU + MaxPool）+ 两层全连接（64·8·8→128→10），约 55 万参数。结构为自行设计，未引用外部模型。
- 预处理：ToTensor 转张量，按 CIFAR-10 均值 (0.4914, 0.4822, 0.4465)、方差 (0.2470, 0.2435, 0.2616) 标准化。
- 训练：CrossEntropyLoss + Adam（lr=1e-3），batch size 128，共 10 轮，设备为 GPU（cuda）。

## 实际运行结果

- 运行设备：NVIDIA RTX 5070 Ti（cuda）
- 训练轮数：10（损失从 1.3282 降至 0.2353）
- 测试集总样本数：10000，正确分类数：7183，**总体准确率 71.83%**
- 各类别准确率：airplane 76.10%、automobile 86.00%、bird 61.30%、cat 42.50%、deer 67.30%、dog 69.20%、frog 83.40%、horse 69.10%、ship 83.00%、truck 79.60%
- 样本输出示例（前 8 条）：true=cat, predicted=dog 等，完整见 `python evaluate.py` 输出

## 权重

- 权重文件：`cifar10_cnn.pth`（约 2.2MB）
- 下载链接：https://github.com/Gu-Yingying/my-git-test/releases/tag/v1.0-recognition
- 使用方法：将权重放到 `src/` 目录，执行 `python evaluate.py`；已有权重时无需重新训练

## 未完成项

无。