# Latent Graph Diffusion (LGD) 论文复现指南

> **论文**: Unifying Generation and Prediction on Graphs with Latent Graph Diffusion
> **目标**: 复现 Table 4 — Node-level classification (Photo, Physics, OGBN-Arxiv)

---

## 一、硬件和系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10/11 或 Linux | Windows 10/11 |
| GPU | NVIDIA GPU (≥8GB VRAM) | RTX 5070 Ti (16GB) |
| CUDA | ≥11.8 | CUDA 12.8 |
| 内存 | ≥16GB | 32GB |
| 磁盘空间 | ≥10GB (环境+数据+模型) | 20GB |
| Python | 3.10 ~ 3.12 | 3.12.6 |

> **注意**: 如果使用较新的 GPU 架构（如 RTX 50 系列 sm_120），本项目已内置兼容性补丁（见第四节），无需额外操作。

---

## 二、环境安装

### 2.1 解压项目

将收到的压缩包解压到任意目录，例如:

```
D:\LatentGraphDiffusion\
```

### 2.2 创建虚拟环境

打开命令行，进入项目目录：

```bash
cd D:\LatentGraphDiffusion

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 激活虚拟环境 (Linux)
# source venv/bin/activate
```

### 2.3 安装 PyTorch

根据你的 CUDA 版本选择对应安装命令。访问 https://pytorch.org/get-started/locally/ 查看最新命令。

```bash
# CUDA 12.8 (RTX 50系列)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.4 安装 PyTorch Geometric 及依赖

```bash
pip install torch_geometric
```

然后安装扩展包（将下方 URL 中的 PyTorch 和 CUDA 版本替换为你实际安装的版本）：

```bash
pip install pyg_lib torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

### 2.5 安装 torch_scatter 和 torch_sparse

```bash
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

> **如果安装失败也不影响运行**，项目已内置 `scatter_compat.py` 兼容层自动替代。

### 2.6 安装其他依赖

```bash
pip install pytorch_lightning matplotlib seaborn rdkit ogb yacs tensorboardX
```

### 2.7 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

输出应类似：
```
PyTorch: 2.10.0+cu128
CUDA可用: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

---

## 三、项目文件说明

```
LatentGraphDiffusion/
├── pretrain.py               # Step 1: Encoder 预训练入口
├── train_diffusion.py        # Step 2: Diffusion 训练入口
├── run_photo.py              # 一键运行脚本
├── find_best_ckpt.py         # 自动查找最佳 checkpoint
├── scatter_compat.py         # torch_scatter/torch_sparse 兼容层
├── utils.py                  # 工具函数
├── cfg/                      # 配置文件
│   ├── photo-encoder.yaml    # Photo Encoder 配置
│   └── photo-diffusion.yaml  # Photo Diffusion 配置
├── lgd/                      # 核心模型代码
│   ├── __init__.py           # 子模块自动注册
│   ├── config/               # GraphGym 配置扩展
│   ├── model/                # 模型定义
│   ├── train/                # 训练逻辑
│   └── ...
└── docs/                     # 本文档所在目录
```

---

## 四、兼容性补丁说明

### 为什么需要补丁？

原项目依赖 `torch_scatter` 和 `torch_sparse`（C++ 扩展），在以下情况可能安装失败：
- GPU 架构太新（如 RTX 50 系列）
- PyTorch 版本太新
- 缺少 C++ 编译环境

### 本项目已做的适配

| 文件 | 说明 |
|------|------|
| `scatter_compat.py` | 用纯 PyTorch 算子重写了 scatter 系列函数，自动替代 `torch_scatter`/`torch_sparse` |
| `lgd/__init__.py` | 原项目缺少此文件，补充后确保模型配置能正确注册 |
| `utils.py` | 删除了 Python 3.12 不兼容的 `import imp` |
| `pretrain.py` / `train_diffusion.py` | 在开头加入 `import scatter_compat` 确保兼容层优先加载 |

> 如果你的环境能正常安装 `torch_scatter` 和 `torch_sparse`，兼容层不会产生任何负面影响。

---

## 五、运行流程

### 方式一：一键运行（推荐）

激活虚拟环境后执行：

```bash
python run_photo.py
```

脚本将自动完成以下全部步骤：
1. Encoder 预训练（10次不同随机种子，约 1.7 小时）
2. 自动查找最佳 Encoder checkpoint
3. Diffusion 训练（10次不同随机种子，约 25 小时）
4. 输出最终测试结果

### 方式二：手动分步运行

#### Step 1: Encoder 预训练

```bash
python pretrain.py --cfg cfg/photo-encoder.yaml --repeat 10 wandb.use False
```

- **耗时**: 约 1.7 小时（每次 ~10分钟 × 10次）
- **输出目录**: `results/photo-encoder/`

#### Step 2: 查找最佳 Checkpoint

```bash
python find_best_ckpt.py --encoder_dir results/photo-encoder --diffusion_cfg cfg/photo-diffusion.yaml
```

此脚本自动从 10 次 Encoder 训练中找到验证集最佳的 checkpoint，并更新 Diffusion 配置文件。

#### Step 3: Diffusion 训练

```bash
python train_diffusion.py --cfg cfg/photo-diffusion.yaml --repeat 10 wandb.use False
```

- **耗时**: 约 25 小时（每次 ~2.5小时 × 10次）
- **输出目录**: `results/photo-diffusion/`

#### Step 4: 查看结果

```bash
python -c "import json; d=json.load(open('results/photo-diffusion/agg/test/best.json')); print(f'测试准确率: {d[\"accuracy\"]:.4f} ± {d[\"accuracy_std\"]:.4f}')"
```

---

## 六、预期结果

### 论文 Table 4 目标

| Dataset | Photo | Physics | OGBN-Arxiv |
|---------|-------|---------|------------|
| **LGD** | **96.94 ± 0.14** | **98.55 ± 0.12** | **73.17 ± 0.22** |

- 结果为 10 次独立运行（不同随机种子）的 mean ± std
- 使用兼容层时可能有微小数值差异（约 ±0.5%）

---

## 七、常见问题

**Q: `torch_scatter` 安装失败怎么办？**
不影响运行。项目已内置 `scatter_compat.py` 兼容层自动替代。

**Q: 报错 `ModuleNotFoundError: No module named 'imp'`**
已修复。如果仍出现，检查 `utils.py` 第 8 行是否有 `import imp`，有则删除。

**Q: 报错 `KeyError: 'pretrain_encoder_inductive'`**
检查 `lgd/__init__.py` 是否存在。此文件负责自动注册训练模式。

**Q: CUDA out of memory**
Photo 数据集约占 2-3GB 显存，确保 GPU 至少有 8GB 可用空间。关闭其他占用显存的程序。

**Q: 训练中途断了怎么办？**
可以重新运行命令，训练会从头开始。已完成的 repeat 结果仍保留在 `results/` 目录中。

---

## 八、参考

- 论文: https://arxiv.org/abs/2402.02518
- 原始代码: https://github.com/zhouc20/LatentGraphDiffusion
