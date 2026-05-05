---
title: EEG Denoising Methods & Journals Survey
layout: default
---

# EEG Denoising Methods & Journals Survey
[:eye: EEG去噪方法与期刊调研]

整理EEG去噪领域的传统方法、深度学习方法以及可投稿期刊调研结果，便于快速了解该领域进展。

---

## 目录

1. [传统去噪方法 (Traditional Methods)](#1-传统去噪方法-traditional-methods)
2. [深度学习去噪方法 (Deep Learning Methods)](#2-深度学习去噪方法-deep-learning-methods)
3. [可投稿期刊 (Target Journals)](#3-可投稿期刊-target-journals)

---

## 1. 传统去噪方法 (Traditional Methods)

> 共收录 **11** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | DT-SFDF | 2024 | 二阶盲识别（SOBI）和典型相关分析（CCA）相融合提高源分离质量，采用自适应阈值法对伪源局部定位，严格固定阈值法去除强伪源，SWT分解弱伪源并调整小波系数 | Removal of Ocular and Muscular Artifacts From Multi-Channel EEG Using Improved Spatial-Frequency Filtering | IEEE JBHI | - | Fuzhou University | 福州大学 李玉榕 教授 |
| 2 | MASR | 2025 | 多模态伪迹子空间重建，结合多模态特征提取、动态自适应阈值和通道显著性度量，实现复杂伪迹自动检测与去除 | Adaptive Thresholding in EEG Artifact Removal Through Multimodal Fusion: A Multimodal Artifact Subspace Reconstruction Approach | IEEE TETCI | - | Xi'an Jiaotong-Liverpool University | - |
| 3 | ARMBR | 2025 | 多元后向回归+简化时间锁定参考信号，最少训练数据下自动检测和去除瞬目伪影 | Artifact-reference multivariate backward regression (ARMBR): A novel method for EEG blink artifact removal with minimal data requirements | Journal of Neural Engineering | - | Weill Cornell Medicine | - |
| 4 | STMS | 2025 | 频谱模板匹配与抑制，K均值聚类构建频谱模板集，欧氏距离自适应选择最相关模板，频域抑制污染信号 | Adaptive EOG Artifact Removal Algorithm Using A Spectrum Magnitude Template Matching for Single-Channel EEG | IEEE Sensors Journal | - | Fudan University | 复旦大学 刘骁 |
| 5 | ARICB | 2025 | Chirp Atom分解+分数小波变换FrWT+自适应阈值设计 | Denoising method for EEG-based expert system in fractional wavelet domain using adaptive residual-incorporating chirp-based model | ESWA | - | Beijing Jiaotong University | 北京交通大学 赵文山 |
| 6 | NSRM | 2025 | 非凸广义极小极大凹（GMC）惩罚项替代L1正则化，保持凸性前提下精准分离稀疏眼电伪迹 | Nonconvex Sparse Regularization Method for Eyeblink Artifact Suppression from Single Channel EEG Signals | IEEE TIM | - | Beijing Institute of Technology | 北京理工大学 机车学院 |
| 7 | GED | 2025 | 广义特征分解（GED）在极端低信噪比条件下（如行走、慢跑）去除高振幅运动伪影，应用于脑微状态分析 | Contrast-based artifact removal enables microstate analysis in ambulatory EEG | IEEE TBME | - | The University of British Columbia | 加拿大 不列颠哥伦比亚大学 |
| 8 | GVICA | 2026 | GWO优化VMD + 熵筛选ICA的无参考、多层级自适应去噪框架 | GVICA: A multi-channel EEG hierarchical noise reduction framework based on GWO dynamically optimized VMD-ICA fusion | IEEE TBME | - | Guangdong University of Technology | 引用了ASTI-Net |
| 9 | GED | 2026 | 广义特征分解，对比静息态 EEG（干净参考）和运动态 EEG（含伪影）协方差矩阵，识别去除运动伪影子空间 | Suppressing Non-Stationary Motion Artefacts in Mobile EEG Using Generalized Eigenvalue Decomposition | Sensors | - | University "G. d'Annunzio" of Chieti-Pescara | 用了打乒乓球的数据 |
| 10 | MCMF | 2025 | 多通道形态分量分析，结合形态分量分解和稀疏表示，实现多通道EEG信号的协同去噪 | Multi-channel Morphological Component Analysis for EEG Denoising | IEEE TBME | - | Zhejiang University | 浙江大学 |
| 11 | SSA | 2024 | 奇异谱分析，通过奇异值分解将EEG信号分解为趋势、振荡和噪声子空间，实现自适应去噪 | Singular Spectrum Analysis for EEG Artifact Removal | Journal of Neural Engineering | - | University of Cambridge | 剑桥大学 |

---

## 2. 深度学习去噪方法 (Deep Learning Methods)

> 共收录 **67** 条记录（截至2026年）

### 2.1 时间线总览

| 年份 | 方法数量 |
|:---:|:---:|
| 2020 | 1 |
| 2021 | 6 |
| 2022 | 5 |
| 2023 | 5 |
| 2024 | 13 |
| 2025 | 32 |
| 2026 | 5 |

### 2.2 方法详情

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | EEGNet | 2018 | 紧凑型CNN，深度可分离卷积减少参数量，轻量化设计适用于BCI系统 | EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | University of Texas at Austin | 经典轻量化网络 |
| 2 | DeepCNN | 2019 | 深层卷积神经网络，通过逐层卷积自动学习EEG特征 | Deep convolutional neural networks for motor imagery brain-computer interfaces | IEEE Transactions on Neural Systems and Rehabilitation Engineering | - | University of Houston | - |
| 3 | TSception | 2020 | 时序感知卷积，融合时序和空间特征 | TSception: A Deep Learning Framework for Emotion Recognition using EEG | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | University of Texas at Austin | - |
| 4 | DGCNN | 2019 | 图卷积神经网络，将EEG电极建模为图结构 | Dynamic Graph Convolutional Neural Networks for EEG Emotion Recognition | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Zhejiang University | 需要torch_geometric |
| 5 | ACRNN | 2020 | 注意力机制卷积循环神经网络 | ACRNN: Attention-based Convolutional Recurrent Neural Network for EEG Analysis | IEEE Transactions on Neural Systems and Rehabilitation Engineering | - | Shanghai Jiao Tong University | - |
| 6 | RGNN | 2021 | 关系图神经网络建模电极间关系 | Relation-Aware Graph Neural Network for EEG Analysis | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Xidian University | - |
| 7 | ASTI-Net | 2024 | 非对称空间-时间交互网络，高效捕捉EEG时空特征 | ASTI-Net: Asymmetric Spatial-Temporal Interaction Network for EEG Denoising | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Southeast University | 东南大学 |
| 8 | SCA-Net | 2024 | 通道自注意力网络，自适应权重调整通道重要性 | SCA-Net: Self-Channel Attention Network for EEG Signal Processing | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Tsinghua University | 清华大学 |
| 9 | MTFNet | 2024 | 多时间尺度特征融合网络 | Multi-Time-Scale Feature Fusion Network for EEG Denoising | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Zhejiang University | 浙江大学 |
| 10 | SSA-Net | 2025 | 频谱空间注意力网络 | SSA-Net: Spectral-Spatial Attention Network for EEG Artifact Removal | IEEE Transactions on Neural Systems and Rehabilitation Engineering | Yes | Fudan University | 复旦大学 |

---

## 3. 可投稿期刊 (Target Journals)

> 共收录 **25** 条期刊

### 3.1 神经工程与脑机接口

| 序号 | 期刊名称 | 期刊简称 | 大类学科 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | IEEE Transactions on Neural Systems and Rehabilitation Engineering | IEEE TNSRE | 工程技术 | 2区 | 2区 | 5.2 | 约200 | IEEE | 神经工程顶刊 |
| 2 | Journal of Neural Engineering | JNE | 工程技术 | 2区 | 2区 | 4.5 | 约150 | IOP Publishing | 神经工程重要期刊 |
| 3 | IEEE Transactions on Neural Systems and Rehabilitation Engineering | IEEE TNSRE | 工程技术 | 1区 | 1区 | 5.6 | 约180 | IEEE | - |

### 3.2 生物医学工程

| 序号 | 期刊名称 | 期刊简称 | 大类学科 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | IEEE Transactions on Biomedical Engineering | IEEE TBME | 工程技术 | 1区 | 1区 | 5.8 | 约300 | IEEE | 生物医学工程顶刊 |
| 2 | IEEE Transactions on Medical Imaging | IEEE TMI | 工程技术 | 1区 | 1区 | 8.2 | 约200 | IEEE | 医学影像顶刊 |
| 3 | IEEE Journal of Biomedical and Health Informatics | IEEE JBHI | 工程技术 | 1区 | 1区 | 6.5 | 约250 | IEEE | 生物医学信息学 |

### 3.3 传感器与信号处理

| 序号 | 期刊名称 | 期刊简称 | 大类学科 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | IEEE Sensors Journal | IEEE SJ | 工程技术 | 2区 | 2区 | 3.5 | 约1000 | IEEE | 传感器领域重要期刊 |
| 2 | IEEE Transactions on Instrumentation and Measurement | IEEE TIM | 工程技术 | 2区 | 2区 | 4.5 | 约400 | IEEE | 仪器测量领域 |
| 3 | IEEE Transactions on Emerging Topics in Computational Intelligence | IEEE TETCI | 工程技术 | 3区 | 2区 | 4.3 | 约150 | IEEE | 新兴计算智能 |

---

## 贡献

欢迎提交Issue或Pull Request来补充完善数据。

## 许可

MIT License