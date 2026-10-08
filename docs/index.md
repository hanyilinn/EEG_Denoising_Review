---
layout: default
title: EEG Research Review
---

## 按主题浏览

<nav class="resource-nav" aria-label="研究资源目录" markdown="1">

1. [研究论文整理](#一研究论文整理-research-papers)
   - [深度学习EEG去噪方法](#11-深度学习eeg去噪方法-deep-learning-eeg-denoising)
   - [传统EEG去噪方法](#12-传统eeg去噪方法-traditional-eeg-denoising)
   - [EEG去噪相关综述](#13-eeg去噪相关综述-eeg-denoising-reviews)
   - [EEG去噪结合下游任务的研究](#14-eeg去噪结合下游任务的研究-eeg-denoising-with-downstream-tasks)
   - [EEG基础模型相关研究](#15-eeg基础模型相关研究-eeg-foundation-model-studies)
   - [EEG Agent相关研究](#16-eeg-agent相关研究-eeg-agent-related-studies)
   - [EEG Benchmark相关研究](#17-eeg-benchmark相关研究-eeg-benchmark-related-studies)
2. [可投稿期刊整理](#二可投稿期刊整理-publication-venues)
   - [EEG去噪可投稿期刊（已有EEG去噪论文发表）](#21-eeg去噪可投稿期刊已有eeg去噪论文发表)
   - [EEG领域下其他可供选择的期刊](#22-eeg领域下其他可供选择的期刊-other-journals-in-eeg-field)
3. [数据集整理](#三数据集整理-datasets)
   - [EEG去噪数据集](#31-eeg去噪数据集-eeg-denoising-datasets)
   - [运动想象数据集](#32-运动想象数据集-motor-imagery-datasets)
   - [情绪识别数据集](#33-情绪识别数据集-emotion-recognition-datasets)

</nav>

## 一、研究论文整理 (Research Papers)

该部分按研究主题组织已读论文与待持续跟踪方向。当前重点包括EEG去噪方法、综述、任务导向去噪、EEG基础模型、EEG智能体和EEG benchmark。

### 1.1 深度学习EEG去噪方法 (Deep Learning EEG Denoising)


#### 1.1.1 时间线总览

> 共收录 **76** 条记录（截至2026年）

| 年份 | 方法数量 |
|:---:|:---:|
| 2020 | 1 |
| 2021 | 6 |
| 2022 | 6 |
| 2023 | 6 |
| 2024 | 14 |
| 2025 | 26 |
| 2026 | 17 |

#### 1.1.2 方法详情

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | 1D-ResCNN | 2020 | 一维残差卷积神经网络端到端去噪... | [A novel end-to-end 1D-ResCNN model to remove artifact from EEG signals](https://www.sciencedirect.com/science/article/abs/pii/S0925231220305944) | Neurocomputing | - | Shaanxi Normal University | - |
| 2 | FCNN | 2021 | EEG去噪基准数据集+基准网络... | [EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8) | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 3 | SimpleCNN | 2021 | EEG去噪基准数据集+基准网络... | [EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8) | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 4 | ComplexCNN | 2021 | EEG去噪基准数据集+基准网络... | [EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8) | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 5 | RNN | 2021 | EEG去噪基准数据集+基准网络... | [EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ac2bf8) | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 6 | Novel CNN | 2021 | 特征维数逐渐上升和时序降采样抑制过拟合... | [A Novel Convolutional Neural Network Model to Remove Muscle Artifacts from EEG](https://ieeexplore.ieee.org/document/9414228) | ICASSP2021 | 是 | Southern University of Science and Technology | - |
| 7 | EEGANet | 2021 | 首次使用GAN进行降噪... | [EEGANet: Removal of ocular artifacts from the EEG signal using generative adversarial networks](https://ieeexplore.ieee.org/document/9627782) | IEEE JBHI | 是 | Vidyasirimedhi Institute of Science and Technology | - |
| 8 | 1-D CNN auto-encoder | 2022 | 利用切比切夫矩形式的正交特征的分数阶压缩一维CNN自编码器... | [Orthogonal features based EEG signals denoising using fractional and compressed onedimensional CNN AutoEncoder](https://ieeexplore.ieee.org/document/9865981) | IEEE TNSRE | - | DA-IICT | - |
| 9 | IC-U-Net | 2022 | 结合ICA和U-Net... | [IC-U-Net: a U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal](https://www.sciencedirect.com/science/article/pii/S1053811922007017) | NeuroImage | 是 | National Tsing Hua University | - |
| 10 | GRU-MARSC | 2022 | 鉴别-去除两阶段、基于GRU... | [Two-Stage Intelligent Multi-Type Artifact Removal for Single-Channel EEG Settings: A GRU Autoencoder Based Approach](https://ieeexplore.ieee.org/document/9741365) | IEEE TBME | - | University of Science and Technology of China | - |
| 11 | Freq_Enhan | 2022 | 频域信息增强网络去噪能力... | [Frequency Information Enhanced Deep EEG Denoising Network for Ocular Artifact Removal](https://ieeexplore.ieee.org/document/9908307) | IEEE Sensors Journal | - | University of Science and Technology of China | - |
| 12 | EEGDnet | 2022 | 2-D Transformer结构... | [EEGDnet: Fusing non-local and local self-similarity for EEG signal denoising with transformer](https://www.sciencedirect.com/science/article/pii/S0010482522009568) | Computers in Biology and Medicine | 是 | University of Electronic Science and Technology of China | - |
| 13 | DeepSeparator | 2022 | 编码器-解码器架构... | [Embedding decomposition for artifacts removal in EEG signals](https://iopscience.iop.org/article/10.1088/1741-2552/ac63eb) | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 14 | DuoCL | 2023 | 双尺度CNN-LSTM网络... | [EEG Reconstruction With a Dual-Scale CNN-LSTM Model for Deep Artifact Removal](https://ieeexplore.ieee.org/document/9973303) | IEEE JBHI | 是 | Wuhan University | - |
| 15 | SDNet | 2023 | 先分割再去噪... | [A Segmentation-Denoising Network for Artifact Removal From Single-Channel EEG](https://ieeexplore.ieee.org/document/10130669) | IEEE Sensors Journal | 是 | University of Science and Technology of China | - |
| 16 | GCTNet | 2023 | GAN引导的CNN和Transformer并行网络... | A GAN Guided Parallel CNN and Transformer Network for EEG Denoising | IEEE JBHI | 是 | University of Science and Technology of China | - |
| 17 | AR-WGAN | 2023 | Wasserstein GAN... | An Approach for EEG Denoising Based on Wasserstein Generative Adversarial Network | IEEE TNSRE | - | Shenzhen Institute of Advanced Technology | - |
| 18 | UDNet | 2023 | 多尺度集中注意力的不确定性感知去噪网络... | Uncertainty-aware Denoising Network for artifact removal in EEG signals | IEEE TNSRE | - | Beijing Jiaotong University | - |
| 19 | DRNet | 2023 | 状态分割+动态权值跳过连接... | Enhancing artifact removal from scalp eeg using state-wise deep convolutional network | IEEE BIBM | 是 | Tsinghua University | - |
| 20 | Denosieformer | 2024 | 基于Transformer的脑电信号去噪架构... | Denosieformer: A Transformer-Based Approach for Single-Channel EEG Artifact Removal | IEEE TIM | - | Nanjing University of Aeronautics and Astronautics | - |
| 21 | EEGIFNet | 2024 | 双分支分别用于恢复干净EEG和伪影... | A Dual-Branch Interactive Fusion Network to Remove Artifacts From Single-Channel EEG | IEEE TIM | 是 | University of Science and Technology of China | - |
| 22 | DT-SFDF | 2024 | 二阶盲识别（SOBI）和典型相关分析（CCA）相融合的方法提高源分离质量，采用自适应阈值法对伪源进行... | Removal of Ocular and Muscular Artifacts from Multi-channel EEG Using Improved Spatial-Frequency Filtering | IEEE JBHI | - | Fuzhou University | 福州大学 李玉榕 教授 |
| 23 | Stacked Multi-Head Attention Transformer | 2024 | 通过在Transformer模型中引入多个多头注意层... | EEG Artifact Removal using Stacked Multi-Head Attention Transformer Architecture | EMBC2024 | - | Indian Institute of Technology Kharagpur | - |
| 24 | DSATCN | 2024 | 利用不同频带的多水平和多尺度时间依赖关系... | Dual-Stream Attention-TCN for EMG Removal From a Single-Channel EEG | IEEE IoTJ | 是 | Guangdong University of Technology | - |
| 25 | DTP-Net | 2024 | 密集连接时间金字塔网络，时频域多尺度特征重用... | DTP-Net: Learning to Reconstruct EEG Signals in Time-Frequency Domain by Multi-Scale Feature Reuse | IEEE JBHI | [是](https://github.com/WilliamRo/EEG-Denoise) | Zhejiang University | Feng Yu |
| 26 | STFNet | 2024 | 自注意机制建立了通道间关系的模型... | Integrating spatial and temporal features for enhanced artifact removal in multi-channel EEG recordings | Journal of Neural Engineering | 是 | University of Science and Technology of China | - |
| 27 | LK-DARTS | 2024 | 神经网络架构搜索... | Enhancing EEG artifact removal through neural architecture search with large kernels | Advanced Engineering Informatics | - | University of Science and Technology of China | - |
| 28 | EEGDiR | 2024 | 保留网络架构和信号嵌入处理脑电... | EEGDiR: Electroencephalogram denoising network for temporal information storage and global modeling through Retentive Network | Computers in Biology and Medicine | 是 | Chengdu University of Technology | - |
| 29 | AGO-CycleGAN | 2024 | 自生成运算神经元和注意力引导的特征金字塔网络（配以改进瓶颈结构）作为生成器，结合PatchGAN判别... | Restoration of motion-corrupted EEG signals using attention-guided operational CycleGAN | EAAI | - | Qatar University | - |
| 30 | LTDNet-EEG | 2024 | 轻量级网络DSC+卡尔曼滤波，在树莓派上实现... | LTDNet-EEG: A Lightweight Network of Portable/Wearable Devices for Real-Time EEG Signal Denoising | IEEE TCE | - | Fuzhou University | - |
| 31 | DAE | 2024 | AutoEncoder结构，基于Tensorflow Lite在智能手机上实现... | Deep autoencoder for real-time single-channel EEG cleaning and its smartphone implementation using TensorFlow Lite with hardware/software acceleration | IEEE TBME | - | The University of Manchester | - |
| 32 | EEGDfus | 2024 | 扩散模型实现精细脑电重构... | EEGDfus: A Conditional Diffusion Model for Fine-Grained EEG Denoising | IEEE JBHI | 是 | University of Science and Technology of China | - |
| 33 | EKFNet | 2024 | 轻量级模型：多尺度特征融合+自适应增益估计+动态预测卡尔曼增益值... | EKFNet: edge-based Kalman filter network for real-time EEG signal denoising | Journal of Neural Engineering | 是 | Minjiang University | - |
| 34 | DHCT-GAN | 2025 | 双分支网络恢复噪声+EEG，CNN+Transformer... | DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch Hybrid CNN–Transformer Network | Sensors | - | National Supercomputing Center in Shenzhen | - |
| 35 | TFG-Net | 2025 | 双流时频门控特征融合... | TFG-Net: A Dual-Stream Time-Frequency Gated Feature Fusion Network for EEG Signal Denoising | arxiv/submitted to Elsevier | - | HoHai University | 感觉文章质量不是很行，报告在EEGdenoiseNet上EMG去噪效果优于EOG |
| 36 | FLANet | 2025 | 多尺度时间卷积和空间频谱注意网络的脑电信号伪影去除... | FLANet: A multiscale temporal convolution and spatial-spectral attention network for EEG artifact removal with adversarial training | Journal of Neural Engineering | - | Fudan University | - |
| 37 | CT-DCENet | 2025 | CNN+Transformer双阶段集成学习，研究混合噪声的去除... | CT-DCENet: Deep EEG Denoising via CNN-Transformer-based Dual-stage Collaborative Ensemble Learning | IEEE JBHI | 是 | Fuzhou University | 一作作者汤云波，原先在DuoCL通讯作者陈丹组里读博，22年毕业入职福州大学 |
| 38 | ART | 2025 | 针对多通道信号，主要使用Transformer，有许多针对下游任务的研究和分析... | Augmenting brain-computer interfaces with ART: An artifact removal transformer for reconstructing multichannel EEG signals | NeuroImage | 是 | National Tsing Hua University | 作者疑似为Tzyy-Ping Jung的徒子徒孙 |
| 39 | MSCGRU | 2025 | MSCNN+GRU... | A hybrid network based on multi-scale convolutional neural network and bidirectional gated recurrent unit for EEG denoising | Neuroscience | - | Southwest University of Science and Technology | - |
| 40 | Essentia | 2025 | 去噪扩散模型加语义引导损失... | Essentia: Boosting Artifact Removal from EEG through Semantic Guidance Utilizing Diffusion Model | ICASSP2025 | 是 | Nankai University | - |
| 41 | ASTI-Net | 2025 | 用于任意通道数目设置的EEG去噪网络，单通道和多通道输入双分支结构，采用可变形卷积、通道注意力和逐通... | A Flexible Spatio-Temporal Architecture Design for Artifact Removal in EEG with Arbitrary Channel-Settings | IEEE JBHI | 是 | University of Science and Technology of China | - |
| 42 | DenoiseMamba | 2025 | 引入了 ConvSSD 模块，该模块将卷积神经网络 (CNN) 与结构化状态空间对偶 (SSD) 机... | DenoiseMamba: An Innovative Approach for EEG Artifact Removal Leveraging Mamba and CNN | IEEE JBHI | - | Fuzhou University | 福州大学 李玉榕 教授 |
| 43 | WNOTNet | 2025 | 小波神经算子（WNO）+Transformer。WNO支持多尺度分析并捕获复杂的时频特征，而Tran... | WNOTNet: A Hybrid Wavelet Neural Operator and Transformer Framework for Enhanced EEG Denoising | IEEE TIM | - | Tianjing University | 天津大学电气自动化与信息工程学院     高忠科 教授 国家优青 |
| 44 | DAE(PbP-QLP) | 2025 | 基于量化的压缩去噪自编码器（DAE）模型，该模型使用PbP-QLP，一种低秩近似（LRA）技术，用于... | Optimized EEG Multi-Noise Removal and Compression: Deploying a PbP-QLP Enhanced Autoencoder on STM32 Microcontroller | IEEE TCE | - | Indian Institute of Technology Patna | 印度理工学院帕特纳分校 Udit Satija |
| 45 | BiGRU-TFA | 2025 | BiGRU-TFA是一种用于EEG信号重建的注意力增强模型，它整合了时间和频率特征。该模型旨在解决传... | BiGRU-TFA: An Attention-Enhanced Model for EEG Signal Reconstruction Using Temporal and Frequency Features | IEEE Sensors Journal | - | Birla Institute of Technology | 印度比拉理工学院，梅斯拉校区，兰契 |
| 46 | ASNet | 2025 | 基于卷积注意力的自适应分离网络，基于U-Net架构，设计了一种新的卷积注意力模块，提高粗粒度和细粒度... | Convolutional attention-based adaptive separation network for EEG artefact removal | BSPC | 是 | Chengdu University of Information Technology | https://github.com/ qwertwjq/ASNet/tree/main |
| 47 | LaBraM Fine-tune | 2025 | 提出了一种基于精细调优的大脑模型（LaBraM）的相关注意力映射方法，该方法利用IMU数据中的空间通... | IMU-Enhanced EEG Motion Artifact Removal with Fine-Tuned Large Brain Models | Arxiv | - | University of California, San Diego (UCSD) | Tzyy-Ping Jung |
| 48 | D4PM | 2025 | 双分支扩散模型，一个学习干净EEG分布，一个学习伪影分布，通过Dual-FiLM机制嵌入噪声水平和伪... | D4PM: A Dual-branch Driven Denoising Diffusion Probabilistic Model with Joint Posterior Diffusion Sampling for EEG Artifacts Removal | Arxiv | 是 | Taiyuan University of Technology | 太原理工大学 杨卫华 |
| 49 | MoE | 2025 | MoE模型驱动的算法，多种EMG类型和多个信噪比对应不同的模型... | A Statistical Mixture-of-Experts Framework for EMG Artifact Removal in EEG: Empirical Insights and a Proof-of-Concept Application | Arxiv | - | Harvard John A. Paulson School of Engineering and Applied Sciences | 哈佛大学和约翰霍普金斯大学共同署名的论文 |
| 50 | EORNet | 2025 | 去除眼电噪声，使用状态空间模型在线性复杂度上建模时间关系... | Efficient Ocular Artifacts Removal from EEG Recordings using State Space Model | IEEE Sensors Journal | - | University of Science and Technology of China | - |
| 51 | MSTP-Net | 2025 | 提出“感受野假说”并构建多尺度时序传播网络，实验验证了扩大神经元感受野对提升降噪效果的有效性... | EEG noise reduction based on the Multi-Scale Temporal Propagation Network | BSPC | - | Zhejiang University | Feng Yu |
| 52 | Microwavenet | 2025 | 轻量级深度学习模型，多分支Morlet小波变换、CBAM 注意力机制及融合soft-DTW、MSE、... | Microwavenet Lightweight CBAM-Augmented Wavelet-Attentive Networks for Robust EEG Denoising | MLSP | 是 | Indian Institute of Technology Mandi | 印度人做的 |
| 53 | LRR-Unet | 2025 | 深度展开网络，其核心是结合传统低秩恢复（LRR）的可解释性与深度学习的强学习能力... | LRR‐UNet: A Deep Unfolding Network With Low‐Rank Recovery for EEG Signal Denoising | CNS Neuroscience & Therapeutics | - | Tianjing University | 天津大学医学院 吕良福、臧蕴亮 |
| 54 | Nested GAN | 2025 | 嵌套生成对抗网络（嵌套 GAN），通过时频域内 GAN 与时域外 GAN 的联合优化，结合复数值 R... | End-to-End EEG Artifact Removal Method via Nested Generative Adversarial Network | Biomedical Physics & Engineering Express | - | Soochow University | 通讯作者Dongyang Xu ustc.edu邮箱 |
| 55 | ReHA-Net | 2025 | 基于U-Net编码器-解码器架构，融合混合注意力模块、多尺度可分离卷积块（MSC Blocks）和可... | ReHA-Net: a ReVIN–hybrid attention network with multiscale convolution for robust EEG artifact removal in brain–computer interfaces | Scientific Reports | - | SRM Institute of Science and Technology | - |
| 56 | TF-Denoiser | 2025 | 时域和频域优化的联合去噪深度学习模型，通过傅里叶变换分别处理实部与虚部，并引入多注意力模块同时提取局... | TF-Denoiser: A Time-Frequency Domain Joint Method for EEG Artifact Removal | Electronics | - | Zhengzhou University of Light Industry | 郑州轻工业大学 |
| 57 | AI Hardware | 2025 | 边缘硬件上实现了部署用于EEG伪迹去除的深度自编码器，硬件包括Arduino Nano 33 BLE... | EEG Artifact Removal At the Edge Using AI Hardware | IEEE Sensors Letter | - | University of Manchester | 曼彻斯特大学 |
| 58 | OMP-LSTM | 2025 | 将正交匹配追踪（OMP）和长短时记忆网络（LSTM）相结合的新方法，通过利用OMP系数在眼电伪影区域... | OMP-LSTM: An Automated Method for Separation of Electrooculogram Artifacts From EEG Signal | IEEE TETCI | - | Bharti School of Telecommunication Technology and Management | 印度理工学院德里分校 |
| 59 | EDGeNet | 2025 | 平衡了高性能的去噪能力与极致的轻量化设计... | EDGeNet: Electroencephalography Denoising Efficient Network for Fast Artifact Removal | EMBC 2025 | 是 | Indian Institute of Technology | 印度理工学院 |
| 60 | STAND-Net | 2026 | 结合脉冲卷积编码器-解码器与漏电积分-放电神经元来模拟脑电图的时空动态，采用增强膨胀残差骨干网络捕捉... | STAND-Net: A Spiking Temporal Attention autoeNcoDer Network for Efficient EEG Artifact Removal | IEEE JBHI | - | Beihang University | 北京航空航天大学 高硕副教授 |
| 61 | SSPT | 2026 | 设计了基于深度学习的脑电图去噪的自监督预训练策略，在多种单通道降噪网络上进行了验证... | Self-Supervised Pre-Training for EEG denoising | Advanced Engineering Informatics | - | University of Science and Technology of China | - |
| 62 | Importance of Balance | 2026 | 利用平衡带符号图的谱理论，通过将带符号图的频率线性映射到正图，成功设计了可学习的低通图滤波器，并以极... | Importance of Balance: Lightweight Transformer via Signed Graph Algorithm Unrolling for EEG Signal Denoising | ICASSP2026 | - | Peking University+New York University | - |
| 63 | FemtoEOGClean | 2026 | 通过参数缩放实验发现眼电去噪在仅需约2000参数时性能即达平台期，并据此设计了至少比同类网络小100... | Femtomodels for EEG Artifact Removal: A Parameter Lower-Bound for Generalisable EOG Denoising | ICASSP2026 | - | - | - |
| 64 | BandRouteNet | 2026 | 频带感知+全频带上下文的双路径设计，在每个频带内部，引入一个路由机制，能够自适应地决定在哪里（时间位... | BandRouteNet: An Adaptive Band Routing Neural Network for EEG Artifact Removal | Arxiv | - | Ho Chi Minh City University of Technology | 唯一越南作者Phat Lam |
| 65 | BG-MSCA | 2026 | BiGRU提取时序特征，多尺度交叉注意力（MSCA）模块通过多尺度门控卷积捕获层次化局部特征、多头注意力建模长程全局依赖，经交叉注意力融合跨尺度信息... | [BG-MSCA: Bidirectional GRU with Multi-Scale Cross Attention for EEG signal denoising](https://www.sciencedirect.com/science/article/pii/S1746809426008815) | BSPC | - | - | - |
| 66 | DAR | 2026 | 监督式一维卷积自编码器，使用成对的伪迹污染与MR校正EEG片段进行训练，同时抑制同步EEG-fMRI中的梯度伪迹和心冲击伪迹（BCG），并通过留一受试者验证跨被试泛化... | [Supervised autoencoder for gradient and BCG artifact removal in EEG during simultaneous EEG-fMRI](https://www.sciencedirect.com/science/article/pii/S0730725X26001232) | Magnetic Resonance Imaging | - | Bangladesh University of Engineering and Technology + University of Illinois Chicago + Qatar University | 使用Carbon Wire Loop公开数据集；保留枕区8–12 Hz alpha活动，并结合显著性分析解释模型关注模式 |
| 67 | FARU-Net | 2026 | 频率感知残差U-Net，在潜在空间通过频率感知瓶颈模块（FBM）进行频谱重校准，并在跳跃连接中加入注意力门控，同时以时域和频域损失约束波形与频谱一致性... | [Dual-Domain Symmetry: A Frequency-Aware Residual U-Net for High-Fidelity EEG Artifact Removal](https://www.mdpi.com/2073-8994/18/6/988) | Symmetry | - | Chengdu University of Technology | 在EEGdenoiseNet上处理单通道EOG/EMG混合伪迹，并在PhysioNet运动想象数据上评估跨数据集频谱保持与下游分类 |
| 68 | Ultra-Compact DSConv U-Net | 2026 | 固定网络结构、损失、数据划分和训练流程，仅扫描深度可分离卷积U-Net的通道宽度，以隔离模型容量影响；发现重构性能在约3K–6.5K参数后趋于饱和，并检验重构指标与下游BCI效用之间的偏差... | [How Much Capacity Does EEG Denoising Need? Ultra-Compact Networks reveal Benchmark Saturation and Metric-Utility Gap](https://arxiv.org/abs/2606.08594) | Arxiv | - | Indian Institute of Technology Mandi | 容量控制与评测研究，并非单纯追求SOTA；CSP+LDA下去噪反而降低运动想象分类性能，代码声明将在正式发表后公开 |
| 69 | ET-informed DL | 2026 | 利用同步眼动追踪（ET）信息训练记录内深度学习模型，预测EEG中可由眼动解释的成分；同时结合真实头模型模拟眼动伪迹，以区分眼动相关神经活动、非神经伪迹和其他脑活动... | [Isolating Eye-Movement Artifacts from EEG Signals](https://www.worldscientific.com/doi/abs/10.1142/S0129065726500437) | International Journal of Neural Systems | - | University of South Carolina + University of Southern California | 提供眼动信息辅助的去混杂与评测框架，可用于评估ICA等伪迹隔离方法的敏感度和特异度 |
| 70 | Multi-head Noise Regression | 2026 | 两头单通道噪声回归器，从2 s EEG片段同时估计EOG与EMG噪声信号比（NSR, dB）；最佳模型为膨胀TCN，用连续伪迹强度作为控制信号，触发选择性小波去眨眼伪迹而非对所有片段统一去噪... | [Multi-head noise regression for single-channel EEG: estimating ocular and muscle contamination to guide artifact removal](https://iopscience.iop.org/article/10.1088/1741-2552/ae541d) | Journal of Neural Engineering | [是](https://github.com/usmanqamarshaikh/EEG-Multi-Head-Noise-Regression) | Auckland University of Technology + New Zealand College of Chiropractic + Aalborg University | 在EEGdenoiseNet合成EOG/EMG污染上训练，并在独立眨眼数据、P3 ERP数据和55名受试者RSVP P300 speller数据上验证；选择性去噪可减少过度清洗并改善AUC |
| 71 | ZUNA1.1 | 2026 | 380M参数扩散自编码EEG基础模型，用于灵活的EEG信号重建；支持最长30 s变长序列、任意通道数量与头皮位置、任意时间片段或整通道重建，并面向去噪和超分辨率任务... | [ZUNA1.1: A more flexible EEG foundation model for Denoising and Super-resolution](https://arxiv.org/abs/2607.27308) | Arxiv | [是](https://huggingface.co/Zyphra/ZUNA1.1) | Zyphra | 开源模型声明为Apache-2.0；相较ZUNA1更强调通道/时间布局灵活性，并报告优于MNE中常用的球面样条插值等标准重建方法 |
| 72 | Gated Artifact Management | 2026 | 面向低密度干电极眼镜式EEG的门控伪迹管理管线；先用轻量级gate classifier识别含伪迹窗口，仅在需要时触发后续处理，再结合时域特征检测、卷积循环网络三分类伪迹识别和U-Net去噪自编码器进行校正... | [A Gated Artifact Management Pipeline for Low-Density Eyewear EEG](https://iopscience.iop.org/article/10.1088/1361-6579/ae99ab) | Physiological Measurement | - | Politecnico di Milano + EssilorLuxottica SA + Università di Pavia | 针对可穿戴眼镜式低密度EEG；29名受试者留一被试评估，gate balanced accuracy 0.89，三类伪迹分类balanced accuracy 0.84、median macro-F1 0.83，端到端median normalized RMSE 0.19、spectral cosine similarity 0.89；强调避免对干净神经活动过度处理并兼容资源受限设备 |
| 73 | MTGNet | 2026 | 任务导向、频谱引导的EEG去噪框架；用并行Mamba-Transformer主干建模时频依赖，Guidance Network估计噪声强度，TACR-LoRA在无干净参考的下游任务上做轻量适配... | [MTGNet: A task-oriented and spectrally guided framework for EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ae9ef1) | Journal of Neural Engineering | - | Tianjin University + Tiangong University | 在EEGDenoiseNet上处理EMG/EOG/混合伪迹，并在MDD半仿真分类和真实疲劳EEG分类中验证下游效用；报告EMG/EOG/混合伪迹S-RRMSE相对最强基线分别降低18.9%、31.5%、14.0%，真实疲劳分类准确率较未处理输入提升6.20-6.69个百分点 |
| 74 | EEG-CPDD | 2026 | 协方差保持的条件扩散EEG去噪框架，面向EEG源定位前处理；针对MSE驱动去噪容易过平滑、破坏空间协方差的问题，通过概率式信号重建和噪声信息引导来保持源定位所需的协方差结构... | [EEG-CPDD: A Covariance-Preserving Diffusion Denoiser for Reliable EEG Source Localization](https://ieeexplore.ieee.org/document/11683990) | IEEE JBHI | - | Guangdong University of Technology + Chongqing University of Posts and Telecommunications | IEEE Fellow 李远清老师组论文；噪声预测网络 RLinDenoiseNet 结合 RLinFormer 低秩注意力、FFiLM 频域噪声水平注入和 SEFuse 双流通道注意力融合；摘要报告在多个benchmark上提升去噪效果、保持空间协方差，并在多种源定位算法中降低偶极定位误差、提升能量重建和参数鲁棒性；DOI: 10.1109/JBHI.2026.3732549 |
| 75 | Diff-ADN | 2026 | 两阶段单通道去噪框架：先按伪迹严重度进行初步重建，再用扩散训练得到的残差编码器执行一次确定性修正，推理时无需迭代反向扩散；覆盖EOG、EMG、ECG及混合伪迹... | [Diff-ADN: A diffusion-guided artifact denoising network with deterministic residual refinement for EEG](https://pubmed.ncbi.nlm.nih.gov/42772331/) | Journal of Neural Engineering | - | Northwestern Polytechnical University | 在4个公开运动想象数据集上检验去噪后的解码恢复；ECG去噪通常恢复4.88–5.59个百分点，BCI IV-2a在-6 dB PTB-ECG污染下最高恢复21.90个百分点；同时指出波形重建指标改善不一定对应BCI准确率的等比例恢复；DOI: 10.1088/1741-2552/aeab36 |
| 76 | EEG-WNet | 2026 | 轻量级小波引导双分支网络：联合时域建模与小波先验学习，并通过跨注意力融合、残差引导细化和多尺度通道—空间注意力提升眼电伪迹重建质量与推理效率 | [EEG-WNet: A wavelet-guided dual-branch network for ocular artifact suppression in electroencephalogram](https://www.sciencedirect.com/science/article/pii/S1746809426021117) | Biomedical Signal Processing and Control | - | Tongji University | 以半合成EOG污染为主，并补充EMG和合成噪声实验；在EEGdenoiseNet上报告RRMSE_t=0.212、SNR=15.42 dB、CC=0.967，另在DEAP真实记录上做定性迁移；8.48M参数、CPU推理1.34 ms；在线发表于2026年，编入2027年卷；DOI: 10.1016/j.bspc.2026.111553 |

---

### 1.2 传统EEG去噪方法 (Traditional EEG Denoising)

> 主要整理传统/信号处理类方法，共收录 **13** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | Single-channel techniques | 2016 | 比较单通道与多通道肌电伪迹去除思路，主张对多通道EEG逐通道独立去噪也可在低信噪比下优于部分多通道盲源分离方法... | [Removing Muscle Artifacts From EEG Data: Multichannel or Single-Channel Techniques?](https://ieeexplore.ieee.org/document/7349109) | IEEE Sensors Journal | - | Hefei University of Technology + University of British Columbia | 通过仿真和真实肌电污染EEG比较不同策略，强调单通道技术在强肌电污染和低SNR场景中的优势 |
| 2 | EEMD-CCA | 2019 | 结合集合经验模态分解（EEMD）与典型相关分析（CCA）去除肌电伪迹，先分解非平稳EEG再利用通道间相关信息识别并抑制肌电成分... | [A Novel EEMD-CCA Approach to Removing Muscle Artifacts for Pervasive EEG](https://ieeexplore.ieee.org/document/8476603) | IEEE Sensors Journal | - | Hefei University of Technology + University of British Columbia | 面向身体传感网络和可穿戴/长期EEG监测；在仿真、半仿真和真实数据上验证，并报告少通道设置也具备竞争力 |
| 3 | DT-SFDF | 2024 | 二阶盲识别（SOBI）和典型相关分析（CCA）相融合的方法提高源分离质量，采用自适应阈值法对伪源进行... | Removal of Ocular and Muscular Artifacts From Multi-Channel EEG Using Improved Spatial-Frequency Filtering | IEEE JBHI | - | Fuzhou University | 福州大学 李玉榕 教授 |
| 4 | MASR | 2025 | 多模态伪迹子空间重建,结合多模态特征提取技术、动态自适应阈值和通道显著性度量，实现了对脑电信号中复杂... | Adaptive Thresholding in EEG Artifact Removal Through Multimodal Fusion: A Multimodal Artifact Subspace Reconstruction Approach | IEEE TETCI | - | Xi'an Jiaotong-Liverpool University | - |
| 5 | ARMBR | 2025 | 通过多元后向回归和简化的时间锁定参考信号，在需要最少训练数据的情况下，实现了对瞬目伪影的自动检测和有... | Artifact-reference multivariate backward regression (ARMBR): A novel method for EEG blink artifact removal with minimal data requirements | Journal of Neural Engineering | - | Weill Cornell Medicine | - |
| 6 | STMS | 2025 | 频谱模板匹配与抑制算法，通过K均值聚类构建频谱模板集，并根据欧氏距离自适应选择最相关模板，然后在频域... | Adaptive EOG Artifact Removal Algorithm Using A Spectrum Magnitude Template Matching for Single-Channel EEG | IEEE Sensors Journal | - | Fudan University | 复旦大学 刘骁 |
| 7 | ARICB | 2025 | Chirp Atom分解+分数小波变换FrWT+自适应阈值设计... | Denoising method for EEG-based expert system in fractional wavelet domain using adaptive residual-incorporating chirp-based model | ESWA | - | Beijing Jiaotong University | 北京交通大学 赵文山 |
| 8 | NSRM | 2025 | 通过一种非凸的广义极小极大凹（GMC）惩罚项替代传统的L1正则化，在保持优化问题凸性的前提下，更精准... | Nonconvex Sparse Regularization Method for Eyeblink Artifact Suppression from Single Channel EEG Signals | IEEE TIM | - | Beijing Institute of Technology | 北京理工大学 机车学院 |
| 9 | GED | 2025 | 广义特征分解（GED）在极端低信噪比条件下（如行走、慢跑时）去除高振幅运动伪影的有效性，并成功将清洗... | Contrast-based artifact removal enables microstate analysis in ambulatory EEG | IEEE TBME | - | The University of British Columbia | 加拿大 不列颠哥伦比亚大学 生物医学工程学院 |
| 10 | GVICA | 2026 | “GWO优化VMD + 熵筛选ICA”的无参考、多层级自适应去噪框架... | GVICA: A multi-channel EEG hierarchical noise reduction framework based on GWO dynamically optimized VMD-ICA fusion | IEEE TBME | - | Guangdong University of Technology | 引用了ASTI-Net |
| 11 | GED | 2026 | 利用广义特征分解，通过对比静息态 EEG（干净参考）和运动态 EEG（含伪影）的协方差矩阵，识别并去... | Suppressing Non-Stationary Motion Artefacts in Mobile EEG Using Generalized Eigenvalue Decomposition | Sensors | - | University "G. d'Annunzio" of Chieti-Pescara, Chieti, Italy | 用了打乒乓球的数据 |
| 12 | ICA-S3M | 2026 | ICA与switching state-space model结合；先将多通道EEG分解为独立成分，再对每个保留IC建模神经振荡和宽带伪迹状态，输出逐时间点artifact probability，无需训练数据... | [ICA-S3M: switching state-space model guided automatic EEG artifact removal from independent components](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1865301/full) | Frontiers in Neuroscience | - | Stanford Medicine | 面向EMG伪迹与“EMG smearing”问题；将神经振荡建模为基于记录自身频谱拟合的阻尼振荡器，将宽带伪迹建模为AR(2)过程，并与EEGdenoiseNet CNN在仿真、半合成真实EEG和自然听音乐记录中比较 |
| 13 | SPAR-EEG | 2026 | 面向无辅助通道、无伪迹标签和无人工选择干净基线的可穿戴单通道EEG；依次以VMD选择性衰减高频EMG爆发，并以SSA分别处理眨眼样EOG瞬态和缓慢运动漂移，只在伪迹主导区域按自适应强度衰减，而非全局删除信号成分 | [SPAR-EEG: Selective Pass-Wise Artifact Reduction for Wearable Single-Channel EEG Denoising](https://doi.org/10.1109/TNSRE.2026.3734253) | IEEE TNSRE | - | Auckland University of Technology + New Zealand College of Chiropractic | 在EEGdenoiseNet的26个输入SNR水平上，EMG、EOG和混合伪迹区域的平均SNR提升分别为9.05、8.28和8.00 dB；还通过PhysioBank、ERP保持、干电极运动EEG及仅使用FP1/FP2的RSVP/P300拼写任务验证，P300重复曲线AUC提高0.048，Letter@15准确率提高7.8个百分点；PMID: 42747931 |

---

### 1.3 EEG去噪相关综述 (EEG Denoising Reviews)

> 共收录 **8** 条综述

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | 综述 | 2015 | EEG伪迹去除技术综述与指南 | [EEG artifact removal—state-of-the-art and guidelines](https://iopscience.iop.org/article/10.1088/1741-2560/12/3/031001/meta) | Journal of Neural Engineering | - | University of Deusto | - |
| 2 | 综述 | 2016 | 脑电信号伪迹检测与去除方法综述 | [Methods for artifact detection and removal from scalp EEG: A review](https://www.sciencedirect.com/science/article/pii/S098770531630199X) | Neurophysiologie Clinique/Clinical Neurophysiology | - | National University of Singapore | - |
| 3 | 综述 | 2018 | EEG伪迹与噪声去除文献综述 | [Artifacts and noise removal for electroencephalogram (EEG): A literature review](https://ieeexplore.ieee.org/abstract/document/8405493/) | IEEE Symposium on Computer Applications & Industrial Electronics (ISCAIE) | - | Universiti Sains Malaysia | - |
| 4 | 综述 | 2019 | 肌电伪迹去除方法综述与建议 | [Removal of Muscle Artifacts From the EEG: A Review and Recommendations](https://ieeexplore.ieee.org/abstract/document/8672623) | IEEE Sensors Journal | - | University of Science and Technology of China | - |
| 5 | 综述 | 2019 | EEG伪迹去除综述 | [Removal of artifacts from EEG signals: a review](https://www.mdpi.com/1424-8220/19/5/987) | Sensors | - | Chinese Academy of Sciences | - |
| 6 | 综述 | 2020 | EEG信号伪迹去除方法综述 | [Methods for removal of artifacts from EEG signal: A review](https://iopscience.iop.org/article/10.1088/1742-6596/1706/1/012093/meta) | Journal of Physics: Conference Series | - | Koneru Lakshmaiah Education Foundation | - |
| 7 | 综述 | 2026 | - | [Physiological artifacts removal in EEG signals: a comprehensive overview of conventional to deep learning methods to support brain health monitoring](https://www.mdpi.com/2227-7080/13/12/578) | Cogent Engineering | - | Manipal Academy of Higher Education | 综述文章 |
| 8 | 综述 | 2026 | 深度学习EEG去噪综述；以端到端去噪管线为框架，梳理数据/目标构造、输入表示、网络架构、目标函数和评估策略，并讨论选择性去噪、多任务去噪、下游验证与部署问题 | [Deep-learning based electroencephalogram denoising: A literature review](https://iopscience.iop.org/article/10.1088/1741-2552/ae89e6) | Journal of Neural Engineering | - | University of Science and Technology of China + The University of British Columbia | PMID: 42443114；DOI: 10.1088/1741-2552/ae89e6 |

---

### 1.4 EEG去噪结合下游任务的研究 (EEG Denoising with Downstream Tasks)

> 共收录 **13** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | - | 2021 | 少通道SSVEP，自采10例数据，比较了无EMG参考的BSS方法和有辅助的EMG方法 | Muscle Artifact Removal Toward Mobile SSVEP-Based BCI: A Comparative Study | IEEE TIM | - | University of Science and Technology of China | - |
| 2 | - | 2023 | 评估伪迹去除对于基于CNN的异常和正常脑电图数据分类算法性能的影响 | The effect of artifact rejection on the performance of a convolutional neural network based algorithm for binary EEG data classification | BSPC | - | Amsterdam UMC location University of Amsterdam | - |
| 3 | Diff-SleepNet | 2024 | 基于DDPM的信号去噪模块，在去噪数据集上预训练，在迁移到睡眠数据集上微调，基于短时傅里叶变换和Transformer组合的特征提取模块，多模态融合模块 | Sleep Stage Classification with Multi-Modal Fusion and Denoising Diffusion Model | IEEE JBHI | [是](https://github.com/Odysseus0816/Diff-SleepNet) | Dalian University of Technology | 空链接 |
| 4 | TOED-GAN | 2024 | 任务导向，面向BCI实际场景，有自采数据集，模拟移动场景、故意加了更多噪声，基于GAN，生成器U-Net | Task-oriented EEG denoising generative adversarial network for enhancing SSVEP-BCI performance | JNE | 否 | National University of Defense Technology | 通讯作者胡德文院士 |
| 5 | FDC-Net | 2025 | EEG去噪任务和多维情绪解码 | FDC-Net: Rethinking the association between EEG artifact removal and multi-dimensional affective computing | Arxiv | 是 | Beijing University of Technology | - |
| 6 | TOL | 2026 | 通过盲源分离分解EEG成分，学习型选择器预测保留概率并加权重建；借助下游任务损失协同优化，仅需任务标签、无需干净参考。 | Task-Oriented Learning for Automatic EEG Denoising | IEEE TIM | 否 | 中国科学院自动化研究所 | - |
| 7 | Multi-head Noise Regression | 2026 | 先估计单通道EOG/EMG污染强度，再只对高污染片段触发小波眨眼去噪；用于验证“选择性去噪”是否比统一去噪更能保护ERP和BCI解码性能 | [Multi-head noise regression for single-channel EEG: estimating ocular and muscle contamination to guide artifact removal](https://iopscience.iop.org/article/10.1088/1741-2552/ae541d) | JNE | [是](https://github.com/usmanqamarshaikh/EEG-Multi-Head-Noise-Regression) | Auckland University of Technology + New Zealand College of Chiropractic + Aalborg University | 在P3 ERP和RSVP P300 speller任务中，统一去噪会削弱显著ERP通道或降低解码，TCN门控的选择性去噪在较少处理片段的同时改善AUC |
| 8 | FAAR | 2026 | 提出Fast Automatic Artifact Rejection，基于频带谱幅值、RMS、最大梯度、过零率和峰度构建epoch级信号质量指数，并从记录内自校准阈值；在13个MOABB公开MI数据集上比较无拒绝、AutoReject、Isolation Forest和FAAR对下游MI解码的影响 | [From EEG Cleaning to Decoding: The Role of Artifact Rejection in MI-based BCIs](https://arxiv.org/abs/2605.12408) | Arxiv | - | Yneuro + SCCN/UC San Diego | 清洗不是总有益：低baseline/低SNR被试更可能受益，BA<0.6时FAAR/AutoReject约57%被试提升；FAAR拒绝epoch比例≤2.3%，并降低跨被试性能方差，适合作为任务导向artifact rejection基线 |
| 9 | Preprocessing Sensitivity / NA-PGI | 2026 | 将EEG预处理选择形式化为反事实干预空间，系统评估同一原始试次在128种预处理管线下的预测翻转；提出Preprocessing Uncertainty度量逐试次预处理不确定性，并用Normalized Adaptive PGI降低预测对预处理变化的敏感性 | [Same Brain, Different Prediction: How Preprocessing Choices Undermine EEG Decoding Reliability](https://arxiv.org/abs/2605.07212) | Arxiv | [是](https://github.com/dengzhe-hou/EEG-Preprocessing-Sensitivity) | Tohoku University + University of Georgia + Texas A&M University + Worcester Polytechnic Institute | 覆盖MI、睡眠、P300、情绪六个公开数据集；BCI-IV-2a中仅改变预处理即可使42.4%试次预测翻转，强调EEG去噪/预处理benchmark应报告下游准确率之外的预测稳定性和预处理不确定性 |
| 10 | EEG-AI | 2026 | 构建human-in-the-loop的EEG预处理智能体系统，由LLM驱动决策agent调用EEG分析工具，结合标准预处理、多个ICA/伪迹分类器概率输出和迭代推理循环，决定ICA成分保留/剔除及是否重跑分析；每轮通过closed-loop policy复评以提高可复现性、可审计性和专家监督效率 | [EEG-AI: An agentic system for AI-assisted semi-automated EEG preprocessing and artifact removal](https://www.sciencedirect.com/science/article/pii/S0165027026000890) | Journal of Neuroscience Methods | - | Unity Health Toronto + University of Toronto + Western University + Toronto Metropolitan University | EEG agent方向的代表性系统论文；在合成EEG和专家标注真实数据上评估artifact detection、ICA分类和重建质量，相对专家标注基线报告Pearson r=0.666±0.188、RMSE=5×10^-6±1×10^-6 |
| 11 | AutocleanEEG-ICVision | 2026 | 在EEG Autoclean平台中加入基于视觉语言模型的ICA伪迹分类agent，将ICA dashboard的地形图、频谱、时间序列等可视化作为输入，自动输出脑源/眼动/肌肉/心电/通道噪声等类别、置信度和可解释理由，用于半自动artifact rejection和人工复核 | [AutocleanEEG - ICVision: Automated ICA Artifact Classification Using Vision-Language AI](https://ieeexplore.ieee.org/document/11539774) | ICMI 2026 | 是 | Cincinnati Children’s Hospital / University of Cincinnati | 面向AI-assisted EEG preprocessing与XAI；关键词包括AI-agent、Vision-language AI、ICA component、artifact rejection和BCI；适合作为智能体/多模态AI辅助EEG去噪工具方向补充 |
| 12 | MTGNet | 2026 | 先用EEGDenoiseNet学习通用去噪先验，再通过TACR-LoRA在MDD和疲劳分类任务中只依赖任务标签适配，目标是“去噪后下游分类仍有用”而不是单纯压低重构误差 | [MTGNet: A task-oriented and spectrally guided framework for EEG denoising](https://iopscience.iop.org/article/10.1088/1741-2552/ae9ef1) | JNE | - | Tianjin University + Tiangong University | 使用ShallowNet、TCNet、EEGNet做跨分类器验证，并做Dataset II/III跨数据集迁移；结果表明MTGNet不是只服务某一个分类器，但作者也承认真实任务只验证了MDD和疲劳，泛化到MI、睡眠、情绪、癫痫等还需进一步验证 |
| 13 | SQI-GEDAI Benchmark | 2026 | 在统一无伪迹处理基线下系统比较9种自动EEG伪迹处理方法，并提出SQI引导的GEDAI，仅修正信号质量较差的通道—时间段，以减少对原始EEG的不必要改动... | [Automated Artifact Removal in EEG Age Prediction: A systematic comparison](https://arxiv.org/abs/2609.31195) | Arxiv | - | Yneuro + collaborating institutions | 在TUEG训练脑龄模型，并零目标域拟合地测试ds005385、LEMON和TDBRAIN；只有GEDAI与SQI-GEDAI在3种深度网络上稳定改善MAE和R²，其余方法平均无益或有害；SQI门控将中位修改比例由74.78%降至42.80%，说明伪迹处理必须用目标下游任务验证 |

---

### 1.5 EEG基础模型相关研究 (EEG Foundation Model Studies)

#### 1.5.1 核心基础模型论文

> 共收录 **13** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | BENDR | 2021 | 将Transformer和对比式自监督学习引入大规模EEG预训练，先在TUEG等大规模EEG上学习通用表征，再迁移到BCI小样本分类任务... | [BENDR: using transformers and a contrastive self-supervised learning task to learn from massive amounts of EEG data](https://arxiv.org/abs/2101.12037) | Frontiers in Human Neuroscience / Arxiv | [是](https://github.com/SPOClab-ca/BENDR) | University of Toronto | 早期EEG foundation/self-supervised代表，常被后续EEG基础模型作为基线 |
| 2 | BIOT | 2023 | Biosignal Transformer，将不同长度、通道数和采样格式的生理信号tokenize为统一“句子”，支持跨EEG/ECG/传感器数据的联合预训练与迁移... | [BIOT: Cross-data Biosignal Learning in the Wild](https://arxiv.org/abs/2305.10351) | NeurIPS 2023 | [是](https://github.com/ycq091044/BIOT) | University of Illinois Urbana-Champaign + Harvard Medical School | 不是纯EEG模型，但在EEG foundation model benchmark中常作为跨生理信号基础模型基线 |
| 3 | Neuro-GPT | 2023 | 由EEG encoder和GPT模块组成，用masked EEG segment reconstruction进行自监督预训练，再在低数据量运动想象任务上微调验证泛化能力... | [Neuro-GPT: Towards A Foundation Model for EEG](https://arxiv.org/abs/2311.03764) | ISBI 2024 | [是](https://github.com/wenhui0206/NeuroGPT) | University of Southern California + Université de Montréal | 明确提出EEG foundation model框架，适合作为LaBraM前后的谱系节点 |
| 4 | BrainWave | 2024 | 面向临床应用的脑信号基础模型，覆盖侵入式与非侵入式神经记录，通过大规模预训练学习可迁移的疾病/异常相关神经表征... | [BrainWave: A Brain Signal Foundation Model for Clinical Applications](https://arxiv.org/abs/2402.10251) | Arxiv | - | - | 范围比EEG更广，适合作为临床脑信号foundation model代表 |
| 5 | LaBraM | 2024 | Large Brain Model，利用大规模EEG数据和神经tokenizer学习跨数据集、跨任务的通用脑电表征，面向BCI和临床任务迁移... | [Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://arxiv.org/abs/2405.18765) | ICLR 2024 Spotlight | [是](https://github.com/935963004/LaBraM) | Shanghai Jiao Tong University | 当前最有代表性的EEG大模型之一，后续大量研究以其为基线或适配对象 |
| 6 | EEGPT / BrainGPT | 2024 | 采用自回归预训练释放EEG generalist foundation model能力，强调统一处理多样EEG格式并提升跨任务迁移表现... | [BrainGPT: Unleashing the Potential of EEG Generalist Foundation Model by Autoregressive Pre-training](https://arxiv.org/abs/2410.19779) | NeurIPS 2024 | [是](https://github.com/BINE022/EEGPT) | - | 代码仓库题名为EEGPT，论文题名为BrainGPT；适合作为通用EEG自回归预训练路线 |
| 7 | CBraMod | 2024 | Criss-Cross Brain Foundation Model，通过频段内/频段间的criss-cross建模机制学习EEG时空依赖，用于跨任务EEG decoding... | [CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding](https://arxiv.org/abs/2412.07236) | ICLR 2025 | [是](https://github.com/wjq-learning/CBraMod) | Zhejiang University | 与LaBraM、EEGMamba并列的主流EEG foundation model基线 |
| 8 | EEGMamba | 2024 | 首个面向通用EEG多任务分类的Mamba/状态空间模型路线，结合双向SSM和Mixture-of-Experts以适配不同任务、长度和通道设置... | [EEGMamba: Bidirectional State Space Model with Mixture of Experts for EEG Multi-task Classification](https://arxiv.org/abs/2407.20254) | Neural Networks 2025 | [是](https://github.com/wjq-learning/EEGMamba) | Zhejiang University | 代表EEG基础模型中的线性复杂度SSM路线，常与LaBraM/CBraMod共同比较 |
| 9 | NeuroLM | 2024 | 将EEG视作“外语”，通过text-aligned neural tokenizer把EEG编码为离散neural tokens，并借助LLM能力实现多任务学习与推理... | [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/abs/2409.00101) | Arxiv | - | Shanghai Jiao Tong University | EEG+语言模型/tokenizer路线，和后续EEG智能体、脑电到文本方向关系紧密 |
| 10 | CEReBrO | 2025 | Compact Encoder for Representations of Brain Oscillations，使用per-channel patch tokenization和alternating attention同时建模通道内时间动态与跨通道空间关系... | [CEReBrO: Compact Encoder for Representations of Brain Oscillations Using Efficient Alternating Attention](https://arxiv.org/abs/2501.10885) | Arxiv | [是](https://github.com/pulp-bio/BioFoundation) | ETH Zurich + University of Bologna | 小型、可复现、偏高效EEG基础模型；后续S-CEReBrO扩展到连续EEG监测 |
| 11 | ZUNA / ZUNA1.1 | 2026 | 380M参数扩散自编码EEG/BCI基础模型，面向任意通道布局的EEG重建、缺失通道补全、去噪和超分辨率；ZUNA1.1进一步支持更灵活的变长序列与局部片段重建... | [ZUNA](https://arxiv.org/abs/2602.18478) / [ZUNA1.1](https://arxiv.org/abs/2607.27308) | Arxiv | [是](https://huggingface.co/Zyphra/ZUNA1.1) | Zyphra | 与EEG去噪网页高度相关，已在深度学习去噪表中单独列为ZUNA1.1 |
| 12 | MSBraM | 2026 | Multi-Scale self-supervised Brain foundation Model，先用多尺度neural tokenizer将原始EEG离散为不同时间分辨率semantic codes，再用curriculum multi-scale masking学习层级动态... | [MSBraM: A Multi-scale Self-supervised Brain Foundation Model for Hierarchical EEG Dynamics Learning](https://arxiv.org/abs/2607.21402) | Arxiv | - | - | 强调EEG内在多尺度时间结构，适合作为后LaBraM时代的多尺度自监督路线 |
| 13 | EEGDM | 2026 | 基于生成式扩散的EEG自监督表征学习框架；使用结构化状态空间模型进行扩散预训练以捕获长程时间依赖，再通过Latent Fusion Module融合多层扩散表征完成下游分类... | [EEGDM: Label-Efficient EEG Representation Learning with Generative Diffusion Model](https://doi.org/10.1109/JBHI.2026.3737616) | IEEE JBHI | [是](https://github.com/jhpuah/EEGDM) | Xiamen University Malaysia + University Malaya Medical Center + collaborators | 在TUEV事件分类、CHB-MIT癫痫检测和IIIC癫痫分类上验证；强调以更少预训练和适配样本获得有竞争力的结果，并开源训练代码与预训练/微调检查点；由2025年预印本更新为2026年正式期刊论文 |

#### 1.5.2 基础模型评测与适配研究

> 共收录 **5** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | Stress Test | 2026 | 对LaBraM、EEGMamba、CBraMod、REVE、BENDR和BIOT在5个临床任务/4个数据集上做冻结线性探针评测，并加入随机初始化、标签置乱、投影敏感性等负对照... | [Stress-Testing EEG Foundation Models for Clinical Decoding: Dataset Identity and Targeted Negative Controls](https://arxiv.org/abs/2607.24519) | Arxiv | - | - | 指出dataset identity可从冻结embedding中被高精度解码，提醒EEG基础模型benchmark必须做泄漏与负对照检查 |
| 2 | Speech Transfer Benchmark | 2026 | 系统评测LaBraM和EEGMamba能否迁移到overt/covert/imagined speech decoding，并与EEGNet、ShallowFBCSPNet、EEGConformer等强基线比较... | [Does EEG Foundation Models Transfer to Speech? A Benchmark on Overt and Imagined Speech Decoding](https://arxiv.org/abs/2607.27268) | Arxiv | - | - | 结论提示当前通用EEG预训练在语音解码上不一定优于小CNN，对“通用脑电解码模型”边界很有参考价值 |
| 3 | Generalization Framework | 2026 | 提出多维度评估框架，在低标签、少通道、参数高效适配等现实约束下评估LaBraM、CSBrain、CBraMod等EEG基础模型的泛化能力... | [A Multi-dimensional Framework for Evaluating Generalization in EEG Foundation Models](https://arxiv.org/abs/2605.28563) | Arxiv | - | University of Southern California | 非常适合作为后续task-oriented benchmark设计参考 |
| 4 | Identity Trap / FMScope | 2026 | 提出FMScope冻结表征诊断协议，从方差分解、subject-axis erasure、aperiodic 1/f消融、层级label probing等角度检查EEG基础模型是否学到受试者身份捷径... | [The Identity Trap in EEG Foundation Models: A Diagnostic Audit](https://arxiv.org/abs/2606.06647) | Arxiv | - | UC San Diego | 对跨被试、临床分类和预处理benchmark尤其重要，避免把身份信息误认为疾病/任务信息 |
| 5 | Representation Audit | 2026 | 用layer-wise probing、LEACE式子空间擦除和透明分类器分析CSBrain、CBraMod、LaBraM学到了什么、用了什么，以及可被传统EEG特征解释多少... | [What Do EEG Foundation Models Capture from Human Brain Signals?](https://arxiv.org/abs/2605.11410) | Arxiv | - | - | 解释性和可信性导向，适合与基础模型性能表配套阅读 |

---

### 1.6 EEG Agent相关研究 (EEG Agent Related Studies)

> 共收录 **6** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | EEGAgent | 2025 | 基于LLM的通用EEG自动分析框架，调度预处理、特征提取、事件检测等工具，支持基础信息感知、时空探索、事件检测、人机交互和报告生成... | [EEGAgent: A Unified Framework for Automated EEG Analysis Using Large Language Models](https://arxiv.org/abs/2511.09947) | Arxiv | [是](https://github.com/rebootingLine/EEGAgent) | Zhejiang University | 偏通用EEG分析智能体，可作为“EEG分析工具链 + LLM调度”的早期代表 |
| 2 | BrainAgent | 2026 | LLM驱动的多智能体脑信号理解框架，由中心supervisor协调多个专门子智能体，将自然语言意图转化为可执行的端到端脑信号处理流程... | [BrainAgent: A Large Language Model-Driven Multi-Agent Framework for Autonomous Brain Signal Understanding](https://arxiv.org/abs/2606.25400) | Arxiv | - | Zhejiang University | 进一步强调层级式多智能体、长流程自动化和脑信号分析benchmark，覆盖范围比单纯EEG更广 |
| 3 | EasyBCI Agent | 2026 | 面向BCI/神经数据预处理的两阶段LLM智能体；Plan Agent生成不暴露原始数据的Data Fingerprint并选择文献依据的算子序列，Execution Agent生成、运行并自纠错代码，结合质量门控经验库复用策略... | [EasyBCI Agent: Towards Universal Neural Data Preprocessing for Brain-Computer Interfaces](https://arxiv.org/abs/2607.29007) | Arxiv | - | - | 与EEG去噪/预处理最相关；在EEG固定线性分类器评估中，报告比人工pipeline更能保留下游任务相关可分性 |
| 4 | CogEEGAgent | 2026 | 面向认知EEG分析的可审计智能体，基于MNE-Python，将LLM意图理解与确定性科学执行分离，并通过typed contracts、confirmation access控制和selection-aware verification降低自适应搜索带来的假阳性风险... | [CogEEGAgent: Toward Autonomous Cognitive EEG Analysis with Grounded Execution and Selection-Aware Verification](https://arxiv.org/abs/2607.25045) | Arxiv | [是](https://github.com/dengzhe-hou/CogEEGAgent) | Tohoku University | 更偏认知EEG统计分析自动化；对未来构建可审计EEG benchmark agent有参考价值 |
| 5 | AutoBCI | 2026 | 面向跨任务EEG解码网络设计的双智能体框架：Designer Agent基于多轮实验反馈生成和改进网络结构，Forecaster Agent结合模型代码、训练协议和早期学习曲线预测完整训练性能并筛选候选模型... | [AutoBCI: Forecast-Guided Agentic Neural Architecture Discovery for EEG-Based Brain-Computer Interfaces](https://arxiv.org/abs/2609.35456) | Arxiv | - | Nanyang Technological University + The University of Hong Kong + Southeast University | 在运动想象、情绪识别和睡眠分期共14个数据集上评估6种LLM；最佳自动发现架构平均balanced accuracy为64.16%，略高于REVE的63.87%；PEEK将早期性能预测MAE由2.20降至1.36个百分点，并估计减少44.9%的训练epoch |
| 6 | autoSCORE | 2026 | 将卷积神经网络SCORE-AI集成进商用EEG阅读器，把长时EEG自动切分为2小时片段，判断正常/异常并识别4类异常模式，同时给出按概率排序的时间标记供临床专家快速复核和修正 | [Clinical validation of autoSCORE: A human-in-the-loop AI tool for interpretation of long-term EEG recordings](https://www.sciencedirect.com/science/article/pii/S1388245726009302) | Clinical Neurophysiology | - | International multicenter clinical EEG consortium | 使用40段住院和40段动态长时EEG（年龄0.9–82.3岁），以3名专家独立判读形成参考标准；AI与专家总体准确率相当，局灶性非癫痫异常检测更优；每类只复核概率最高的10个标记即可用完整片段1.5%–5.5%的内容纠正所有异常片段。它不是LLM agent，但属于高价值的人机协同EEG智能工具；DOI: 10.1016/j.clinph.2026.2112430 |

---

### 1.7 EEG Benchmark相关研究 (EEG Benchmark Related Studies)

> 共收录 **4** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | Preprocessing Benchmark | 2020 | 系统比较EEG结果对预处理流程的敏感性，围绕ICA-based preprocessing、ASR等自动化流程，评估不同预处理选择对信号统计、ERP/ERSP和眨眼伪迹残差等结果的影响... | [How Sensitive Are EEG Results to Preprocessing Methods: A Benchmarking Study](https://pubmed.ncbi.nlm.nih.gov/32217478/) | IEEE TNSRE | - | The University of Texas at San Antonio + Army Research Laboratory + SCCN/UC San Diego | 适合作为“预处理选择会改变EEG结论”的早期benchmark参考；使用17个EEG研究比较LARG、MARA和两种ASR变体，强调应详细报告处理细节并用自动化处理管线联盟量化流程选择影响 |
| 2 | LibEER | 2024 | 面向EEG情绪识别的综合benchmark和算法库，统一数据处理、模型训练和评估流程，覆盖多种公开情绪EEG数据集与传统/深度学习方法，便于公平复现实验比较... | [LibEER: A Comprehensive Benchmark and Algorithm Library for EEG-Based Emotion Recognition](https://arxiv.org/abs/2410.09767) | Arxiv / Benchmark Library | [是](https://github.com/yi-ding-cs/LibEER) | Tsinghua University + Shanghai Jiao Tong University + Monash University | 适合放入通用EEG benchmark模块；重点不在去噪，而在情绪识别任务的标准化评估、算法复现和跨数据集比较 |
| 3 | BrainBench | 2026 | 构建面向LLM的综合EEG理解benchmark，评估大语言模型对EEG知识、信号分析、任务理解、临床/BCI场景推理等问题的回答能力... | [BrainBench: Benchmarking Large Language Models for Comprehensive EEG Understanding](https://arxiv.org/abs/2608.04156) | Arxiv | - | - | 与EEG Agent和EEG基础模型方向关系紧密；适合作为评估LLM/agent是否具备EEG领域知识和分析能力的benchmark |
| 4 | GCN Harmonization | 2026 | 针对不同MI-BCI数据集电极布局不一致导致的跨数据集迁移困难，提出基于GCN的信号级空间harmonization框架，将异构EEG映射到统一物理电极montage，并用EEGNet、FBCNet、ADFCNN等下游分类器验证... | [Graph convolutional network-based harmonization of EEG for cross-dataset transfer in motor imagery in BCI](https://iopscience.iop.org/article/10.1088/1741-2552/ae9344) | Journal of Neural Engineering | - | Singapore Institute of Technology | 适合作为跨数据集MI-BCI benchmark/数据标准化方向参考；强调在信号层解决电极布局不兼容，并报告harmonized EEG可改善跨数据集迁移和目标域微调表现 |

---

## 二、可投稿期刊整理 (Publication Venues)

该部分整理EEG去噪及相关EEG研究可考虑的投稿期刊，区分已有EEG去噪论文发表的期刊与更广泛的EEG领域候选期刊。

**指标更新：2026-10-08。** 影响因子（JIF）与CiteScore采用2025统计年度、2026年发布的最新年值；不是2026年实时预测值。点击数值可查看本次采用的来源。优先使用出版商官网，官网不可访问或未标明年度时使用高校研究门户及公开指标汇总（BioxBio、Journal Indexes、LetPub等）；二级来源数据仅供选刊参考，正式评价以[JCR](https://jcr.clarivate.com/)和[Scopus](https://www.scopus.com/sources)为准。“未查到”不等于0。

> **分区口径更正：** [中科院文献情报中心已宣布自2026年起停止更新与发布期刊分区表](https://cssar.cas.cn/library/dtxx/202604/t20260409_8183275.html)。本页只展示独立第三方的**2026年新锐分区**，不称为“2026年中科院分区”，也不与JCR/Scopus的Q1–Q4混用。分区逐条对照[承德医学院公开的2026年新锐分区名单](https://kjc.cdmc.edu.cn/art/2026/4/1/art_299_101218.html)；该名单不含TOP字段，因此不保留未经核验的TOP标记。发布机构与口径见[新锐分区介绍](https://www.xr-scholar.com/Docs/RankingIntro)。

年文章数及备注中的自引率沿用原记录，统计年度未核验；本次只更新分区、JIF与CiteScore。

### 2.1 EEG去噪可投稿期刊（已有EEG去噪论文发表）

> 共收录 **29** 条期刊（按学科分类）

#### 2.1.1 医学

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Journal of Biomedical and Health Informatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221020) | IEEE JBHI | 一区 | [7.7](https://www.bioxbio.com/journal/IEEE-J-BIOMED-HEALTH) | [14.3](https://journalindexes.com/ieee-journal-of-biomedical-and-health-informatics-21682194/) | 659 | IEEE | - |
| 2 | [NeuroImage](https://www.sciencedirect.com/journal/neuroimage) | NeuroImage | 二区 | [5.3](https://www.bioxbio.com/journal/NEUROIMAGE) | [9.6](https://journalindexes.com/neuroimage-10538119/) | 447 | Elsevier | - |
| 3 | [Biomedical Signal Processing and Control](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) | BSPC | 二区 | [5.7](https://www.bioxbio.com/journal/BIOMED-SIGNAL-PROCES) | [13.0](https://fis.tu-dresden.de/portal/en/journals/biomedical-signal-processing-and-control%28a559a689-bbdc-42a7-8a1e-f8d7ae65936f%29.html) | 1370 | Elsevier | - |
| 4 | [IEEE Transactions on Biomedical Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10) | IEEE TBME | 二区 | [4.4](https://www.bioxbio.com/journal/IEEE-T-BIO-MED-ENG) | [10.0](https://fis.tu-dresden.de/portal/en/journals/ieee-transactions-on-biomedical-engineering%28c5ee94b3-9b63-4df7-8949-3276ef827af7%29.html) | 331 | IEEE | - |
| 5 | [IEEE Transactions on Neural Systems and Rehabilitation Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7333) | IEEE TNSRE | 二区 | [6.6](https://www.bioxbio.com/journal/IEEE-T-NEUR-SYS-REH) | [11.0](https://journalindexes.com/ieee-transactions-on-neural-systems-and-rehabilitation-engineering-15344320/) | 407 | IEEE | - |
| 6 | [Computers in Biology and Medicine](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine) | CiBM | 未列入 | 未查到 | [13.1](https://fis.tu-dresden.de/portal/en/journals/computers-in-biology-and-medicine%2800a4f59b-c47e-4bed-b6a3-ca5ca1e0a866%29.html) | 274 | Elsevier | 未查到2025 JIF；投稿前核实WoS收录状态 |
| 7 | [CNS Neuroscience & Therapeutics](https://onlinelibrary.wiley.com/journal/17555949) | CNSNT | 二区 | [6.6](https://www.bioxbio.com/journal/CNS-NEUROSCI-THER) | [8.5](https://onlinelibrary.wiley.com/journal/17555949) | 490 | Wiley-Blackwell | - |
| 8 | [Journal of Neural Engineering](https://iopscience.iop.org/journal/1741-2552) | JNE | 二区 | [4.0](https://www.bioxbio.com/journal/J-NEURAL-ENG) | [6.9](https://journalindexes.com/journal-of-neural-engineering-17412552/) | 320 | IOP Publishing | - |
| 9 | [IEEE Journal of Translational Engineering in Health and Medicine](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221039) | IEEE JTEHM | 二区 | [3.9](https://sci.justscience.cn/details.html?id=14211&sci=1) | [10.2](https://journalindexes.com/ieee-journal-of-translational-engineering-in-health-and-medicine-21682372/) | 69 | IEEE | - |
| 10 | [Frontiers in Human Neuroscience](https://www.frontiersin.org/journals/human-neuroscience) | FHN | 三区 | [3.6](https://www.bioxbio.com/journal/FRONT-HUM-NEUROSCI) | [5.9](https://journalindexes.com/frontiers-in-human-neuroscience-16625161/) | 462 | Frontiers | - |
| 11 | [Computer Methods in Biomechanics and Biomedical Engineering](https://www.tandfonline.com/journals/gcmb20) | CMBBE | 四区 | [1.9](https://www.bioxbio.com/journal/COMPUT-METHOD-BIOMEC) | [4.6](https://journalindexes.com/computer-methods-in-biomechanics-and-biomedical-engineering-10255842/) | 229 | Taylor & Francis | - |
| 12 | [Neuroscience](https://www.sciencedirect.com/journal/neuroscience) | Neuroscience | 三区 | [3.3](https://www.bioxbio.com/journal/NEUROSCIENCE) | [4.6](https://journalindexes.com/neuroscience-03064522/) | 478 | Elsevier | - |
| 13 | [Biomedical Physics & Engineering Express](https://iopscience.iop.org/journal/2057-1976) | BPEE | 四区 | [2.0](https://www.ablesci.com/journal/detail?id=524Qjr) | [2.8](https://journalindexes.com/biomedical-physics-and-engineering-express-20571976/) | 279 | IOP Publishing | - |

#### 2.1.2 计算机科学

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | TNNLS | 二区 | [9.7](https://www.bioxbio.com/journal/IEEE-T-NEUR-NET-LEAR) | [23.7](https://journalindexes.com/ieee-transactions-on-neural-networks-and-learning-systems-2162237X/) | 869 | IEEE | - |
| 2 | [Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence) | EAAI | 一区 | [9.0](https://www.bioxbio.com/journal/ENG-APPL-ARTIF-INTEL) | [11.7](https://fis.tu-dresden.de/portal/en/journals/engineering-applications-of-artificial-intelligence--the-international-journal-of-realtime-automation%2870918efa-cd41-4c2a-8d21-01688d31e6df%29.html) | 1909 | Elsevier | - |
| 3 | [Expert Systems With Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications) | ESWA | 一区 | [9.4](https://www.bioxbio.com/journal/EXPERT-SYST-APPL) | [17.0](https://fis.tu-dresden.de/portal/en/journals/expert-systems-with-applications--an-international-journal%28635663c1-5308-4ee9-bd93-1779b0080c40%29.html) | 2925 | Elsevier | - |
| 4 | [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems) | KBS | 一区 | [8.0](https://www.bioxbio.com/journal/KNOWL-BASED-SYST) | [13.7](https://fis.tu-dresden.de/portal/en/journals/knowledgebased-systems%28cb8c8519-2e8c-49ca-81d2-89900b6eccfb%29.html) | 1354 | Elsevier | - |
| 5 | [IEEE Internet of Things Journal](https://ieee-iotj.org/) | IEEE IoTJ | 一区 | [8.7](https://xplorestaging.ieee.org/xpl/topics-issue?punumber=6488907) | [14.7](https://journalindexes.com/ieee-internet-of-things-journal-23274662/) | 2984 | IEEE | - |
| 6 | [IEEE Transactions on Consumer Electronics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=30) | IEEE TCE | 一区 | [9.9](https://www.bioxbio.com/journal/IEEE-T-CONSUM-ELECTR) | [8.1](https://journalindexes.com/ieee-transactions-on-consumer-electronics-00983063/) | 699 | IEEE | 自引率52.3% |
| 7 | [IEEE Transactions on Emerging Topics in Computational Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7433297) | IEEE TETCI | 二区 | [6.0](https://xplorestaging.ieee.org/xpl/tocresult.jsp?isnumber=11538055&punumber=7433297) | [11.5](https://journalindexes.com/ieee-transactions-on-emerging-topics-in-computational-intelligence-2471285X/) | 419 | IEEE | - |
| 8 | [IEEE Transactions on Cognitive and Developmental Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7274989) | IEEE TCDS | 二区 | [4.7](https://goingpub.com/journal_detail/2915.html) | [11.1](https://journalindexes.com/ieee-transactions-on-cognitive-and-developmental-systems-23798920/) | 171 | IEEE | - |
| 9 | [Electronics](https://www.mdpi.com/journal/electronics) | - | 三区 | [2.9](https://www.mdpi.com/news/17132) | [7.0](https://www.mdpi.com/journal/electronics/history) | 5022 | MDPI | - |
| 10 | [Signal Image and Video Processing](https://www.springer.com/journal/11760) | SIVP | 四区 | [2.7](https://www.bioxbio.com/journal/SIGNAL-IMAGE-VIDEO-P) | [3.9](https://www.letpub.com.cn/index.php?journalid=8221&page=journalapp&view=detail) | 634 | Springer | - |

#### 2.1.3 工程技术

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Advanced Engineering Informatics](https://www.sciencedirect.com/journal/advanced-engineering-informatics) | AdvEI | 一区 | [11.5](https://www.bioxbio.com/journal/ADV-ENG-INFORM) | [15.8](https://fis.tu-dresden.de/portal/en/journals/advanced-engineering-informatics%289ab8f114-666e-4cc4-bf79-0ba9552fae7c%29.html) | 658 | Elsevier | - |
| 2 | [IEEE Transactions on Instrumentation and Measurement](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=19) | IEEE TIM | 二区 | [7.0](https://www.bioxbio.com/journal/IEEE-T-INSTRUM-MEAS) | [9.8](https://fis.tu-dresden.de/portal/de/journals/ieee-transactions-on-instrumentation-and-measurement%289068506e-490f-4808-8819-c257fd814fde%29.html) | 2532 | IEEE | - |
| 3 | [Cogent Engineering](https://www.tandfonline.com/journals/oaen20) | \ | 三区 | [2.7](https://www.tandfonline.com/action/journalInformation?journalCode=oaen20) | [6.9](https://journalindexes.com/cogent-engineering-23311916/) | 356 | Taylor & Francis | - |
| 4 | [Technologies](https://www.mdpi.com/journal/technologies) | \ | 三区 | [5.2](https://www.mdpi.com/journal/technologies/history) | [6.7](https://journalindexes.com/technologies-22277080/) | 262 | MDPI | - |

#### 2.1.4 综合性期刊

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Sensors Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7361) | IEEE SJ | 二区 | [4.5](https://www.bioxbio.com/journal/IEEE-SENS-J) | [7.8](https://fis.tu-dresden.de/portal/en/journals/ieee-sensors-journal%28c28b1ba9-dc24-4e05-9546-737744c349a4%29.html) | 3965 | IEEE | - |
| 2 | [Sensors](https://www.mdpi.com/journal/sensors) | Sen. | 三区 | [4.0](https://www.bioxbio.com/journal/SENSORS-BASEL) | [9.4](https://www.mdpi.com/journal/sensors/history) | 8165 | MDPI | - |


---

### 2.2 EEG领域下其他可供选择的期刊 (Other Journals in EEG Field)

> 共收录 **34** 条期刊（按学科分类）

#### 2.2.1 计算机科学

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | IEEE TPAMI | 一区 | [20.4](https://www.bioxbio.com/journal/IEEE-T-PATTERN-ANAL) | [41.1](https://innovate.ieee.org/ieee-journals-continue-to-excel-in-citation-rankings/) | 732 | IEEE | - |
| 2 | [Cyborg and Bionic Systems](https://spj.science.org/journal/cbsystems) | CBS | 一区 | [20.9](https://spj.science.org/journal/cbsystems) | [21.8](https://fis.tu-dresden.de/portal/en/journals/cyborg-and-bionic-systems%28512509a8-62c8-4a2e-8862-c7bdec7a23de%29.html) | 65 | AAAS | 中国科学院自动化研究所与美国科学促进会合作出版 |
| 3 | [Information Fusion](https://www.sciencedirect.com/journal/information-fusion) | Inf. Fus. | 一区 | [17.4](https://www.bioxbio.com/journal/INFORM-FUSION) | [26.7](https://journalindexes.com/information-fusion-15662535/) | 554 | Elsevier | - |
| 4 | [IEEE Transactions on Knowledge and Data Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69) | IEEE TKDE | 二区 | [11.6](https://www.bioxbio.com/journal/IEEE-T-KNOWL-DATA-EN) | [22.7](https://fis.tu-dresden.de/portal/en/journals/ieee-transactions-on-knowledge-and-data-engineering--tkde%28f056011d-90d7-4da8-a310-12bdd6c3d3ee%29.html) | 666 | IEEE | - |
| 5 | [IEEE Transactions on Cybernetics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221036) | IEEE TCYB | 一区 | [11.3](https://www.bioxbio.com/journal/IEEE-T-CYBERNETICS) | [28.2](https://journalindexes.com/ieee-transactions-on-cybernetics-21682267/) | 441 | IEEE | - |
| 6 | [IEEE Transactions on Industrial Informatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9424) | IEEE TII | 一区 | [9.8](https://www.bioxbio.com/journal/IEEE-T-IND-INFORM) | [22.4](https://journalindexes.com/ieee-transactions-on-industrial-informatics-15513203/) | 1132 | IEEE | - |
| 7 | [IEEE Transactions on Multimedia](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046) | IEEE TMM | 一区 | [9.9](https://www.bioxbio.com/journal/IEEE-T-MULTIMEDIA) | [15.7](https://fis.tu-dresden.de/portal/en/journals/ieee-transactions-on-multimedia%286a8cdc88-a5e4-4f33-bae6-1315498038fd%29.html) | 872 | IEEE | - |
| 8 | [IEEE Transactions on Systems Man Cybernetics-Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221037) | IEEE TSMC | 一区 | [8.4](https://www.bioxbio.com/journal/IEEE-T-SYST-MAN-CY-S) | [17.8](https://journalindexes.com/ieee-transactions-on-systems-man-and-cybernetics-systems-21682216/) | 667 | IEEE | - |
| 9 | [Pattern Recognition](https://www.sciencedirect.com/journal/pattern-recognition) | PR | 一区 | [9.1](https://www.bioxbio.com/journal/PATTERN-RECOGN) | [17.3](https://fis.tu-dresden.de/portal/en/journals/pattern-recognition--the-journal-of-the-pattern-recognition-society%2814e53020-b9b4-4769-b128-49c20eec2c4a%29.html) | 966 | Elsevier | - |
| 10 | [Neural Networks](https://www.sciencedirect.com/journal/neural-networks) | NN | 二区 | [7.2](https://www.bioxbio.com/journal/NEURAL-NETWORKS) | [10.9](https://journalindexes.com/neural-networks-08936080/) | 856 | Elsevier | - |
| 11 | [Applied Soft Computing](https://www.sciencedirect.com/journal/applied-soft-computing) | ASC | 二区 | [7.8](https://www.bioxbio.com/journal/APPL-SOFT-COMPUT) | [13.1](https://journalindexes.com/applied-soft-computing-15684946/) | 1269 | Elsevier | - |
| 12 | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | Neurocomputing | 二区 | [6.7](https://www.bioxbio.com/journal/NEUROCOMPUTING) | [10.8](https://fis.tu-dresden.de/portal/en/journals/neurocomputing%28942b261e-960b-4538-9be4-3184d9070778%29.html) | 1629 | Elsevier | - |
| 13 | [IEEE Transactions on Automation Science and Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8856) | IEEE TASE | 二区 | [7.9](https://www.bioxbio.com/journal/IEEE-T-AUTOM-SCI-ENG) | [10.3](https://journalindexes.com/ieee-transactions-on-automation-science-and-engineering-15455955/) | 840 | IEEE | - |
| 14 | [IEEE Transactions on Human-Machine Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221038) | IEEE THMS | 二区 | [4.4](https://www.bioxbio.com/journal/IEEE-T-HUM-MACH-SYST) | [10.6](https://journalindexes.com/ieee-transactions-on-human-machine-systems-21682291/) | 78 | IEEE | - |
| 15 | [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97) | IEEE SPL | 三区 | [3.6](https://www.bioxbio.com/journal/IEEE-SIGNAL-PROC-LET) | [6.3](https://journalindexes.com/ieee-signal-processing-letters-10709908/) | 635 | IEEE | - |
| 16 | [Applied Intelligence](https://www.springer.com/journal/10489) | AppI | 三区 | [3.5](https://www.bioxbio.com/journal/APPL-INTELL) | [9.1](https://letpub.com.cn/index.php?journalid=737&page=journalapp&view=detail) | 635 | Springer | - |

#### 2.2.2 医学

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Journal of NeuroEngineering and Rehabilitation](https://jneuroengrehab.biomedcentral.com/) | JNER | 二区 | [6.0](https://www.bioxbio.com/journal/J-NEUROENG-REHABIL) | [8.6](https://journalindexes.com/journal-of-neuroengineering-and-rehabilitation-17430003/) | 225 | BioMed Central | - |
| 2 | [Artificial Intelligence in Medicine](https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine) | AIM | 二区 | [7.8](https://www.bioxbio.com/journal/ARTIF-INTELL-MED) | [15.0](https://letpub.com.cn/index.php?journalid=893&page=journalapp&view=detail&xuanxiangk_id=2) | 207 | Elsevier | - |
| 3 | [Human Brain Mapping](https://onlinelibrary.wiley.com/journal/10970193) | HBM | 二区 | [4.0](https://www.bioxbio.com/journal/HUM-BRAIN-MAPP) | [8.2](https://fis.tu-dresden.de/portal/en/journals/human-brain-mapping%284cb511ae-ea84-40c0-893a-8e427434d69c%29.html) | 339 | Wiley-Liss | - |
| 4 | [Journal of Biomedical Informatics](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics) | JBI | 二区 | [5.9](https://www.bioxbio.com/journal/J-BIOMED-INFORM) | [10.8](https://journalindexes.com/journal-of-biomedical-informatics-15320464/) | 146 | Elsevier | - |
| 5 | [IEEE Transactions on Biomedical Circuits and Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4156126) | IEEE TBCS | 二区 | [5.1](https://www.bioxbio.com/journal/IEEE-T-BIOMED-CIRC-S) | [10.6](https://fis.tu-dresden.de/portal/en/journals/ieee-transactions-on-biomedical-circuits-and-systems%286249a3f9-328a-4d53-9d63-53434e63f4c3%29.html) | 102 | IEEE | - |
| 6 | [Computer Methods and Programs in Biomedicine](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine) | CMPB | 二区 | [6.4](https://www.bioxbio.com/journal/COMPUT-METH-PROG-BIO) | [11.9](https://fis.tu-dresden.de/portal/en/journals/computer-methods-and-programs-in-biomedicine--an-international-journal-devoted-to-the-development-implementation-and-exchange-of-computing-methodology-and-software-systems-in-biomedical-research-and-medical-practice%283200bb6c-bdfa-4996-9407-741e21183131%29.html) | 440 | Elsevier | - |
| 7 | [IEEE Transactions on Medical Robotics and Bionics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8263503) | IEEE TMRB | 二区 | [4.3](https://www.ablesci.com/journal/detail?id=p3q7ND) | [6.7](https://journalindexes.com/ieee-transactions-on-medical-robotics-and-bionics-25763202/) | 156 | IEEE | - |
| 8 | [Clinical Neurophysiology](https://www.sciencedirect.com/journal/clinical-neurophysiology) | - | 二区 | [3.4](https://www.bioxbio.com/journal/CLIN-NEUROPHYSIOL) | [6.3](https://journalindexes.com/clinical-neurophysiology-13882457/) | 234 | Elsevier | - |
| 9 | [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience) | Fron. Neur. | 三区 | [4.0](https://www.frontiersin.org/journals/neuroscience) | [7.4](https://www.frontiersin.org/journals/neuroscience) | 1118 | Frontiers | - |
| 10 | [Cognitive Neurodynamics](https://www.springer.com/journal/11571) | CN | 二区 | [4.5](https://www.bioxbio.com/journal/COGN-NEURODYNAMICS) | [7.1](https://fis.tu-dresden.de/portal/en/journals/cognitive-neurodynamics%289fe701ba-4071-41c8-8f9b-6f25bb61ee6a%29.html) | 126 | Springer | - |

#### 2.2.3 工程技术

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Signal Processing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=78) | IEEE TSP | 二区 | [5.5](https://www.bioxbio.com/journal/IEEE-T-SIGNAL-PROCES) | [11.7](https://journalindexes.com/ieee-transactions-on-signal-processing-1053587X/) | 375 | IEEE | - |
| 2 | [Computers & Industrial Engineering](https://www.sciencedirect.com/journal/computers-and-industrial-engineering) | CIE | 一区 | [7.3](https://www.bioxbio.com/journal/COMPUT-IND-ENG) | [13.5](https://fis.tu-dresden.de/portal/en/journals/computers-and-industrial-engineering%2812685f83-32a4-4500-946b-d48854cf5d64%29.html) | 812 | Elsevier | - |

#### 2.2.4 综合性期刊

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Advanced Science](https://onlinelibrary.wiley.com/journal/21983844) | Adv. Sci. | 一区 | [14.1](https://advanced.onlinelibrary.wiley.com/journal/21983844) | [18.1](https://advanced.onlinelibrary.wiley.com/journal/21983844) | 3290 | Wiley-VCH | - |
| 2 | [Science Advances](https://www.science.org/journal/sciadv) | Sci. Adv. | 一区 | [13.9](https://www.wolterskluwer.com/en/solutions/ovid/science-advances-15526) | [19.0](https://fis.tu-dresden.de/portal/en/journals/science-advances%28d3aea9a7-4044-46b3-a8ec-1eac64fafaff%29.html) | 2263 | AAAS | - |
| 3 | [Scientific Data](https://www.nature.com/sdata/) | Sci. Dat. | 二区 | [7.2](https://www.nature.com/sdata/journal-impact) | [9.6](https://journalindexes.com/scientific-data-20524463/) | 1405 | Springer Nature | Open Access |
| 4 | [Scientific Reports](https://www.nature.com/srep/) | Sci. Rep. | 三区 | [4.9](https://www.bioxbio.com/journal/SCI-REP-UK) | [6.4](https://www.mathnet.ru/php/journal.phtml?jrnid=sr&option_lang=eng) | 31052 | Springer Nature | Open Access |
| 5 | [PLoS One](https://journals.plos.org/plosone/) | PLoS One | 三区 | [2.8](https://www.bioxbio.com/journal/PLOS-ONE) | [4.8](https://journalindexes.com/plos-one-19326203/) | 16608 | Public Library of Science | Open Access |

#### 2.2.5 生物学

| 序号 | 期刊名称 | 期刊简称 | 2026年新锐分区 | 影响因子（2025） | CiteScore（2025） | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE-ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8857) | IEEE TCBB | 三区 | [4.1](https://www.bioxbio.com/journal/IEEE-ACM-T-COMPUT-BI) | [9.7](https://journalindexes.com/ieee-transactions-on-computational-biology-and-bioinformatics-29984165/) | 228 | IEEE | 现刊名为IEEE Transactions on Computational Biology and Bioinformatics；CiteScore按现刊名查询 |


---

## 三、数据集整理 (Datasets)

该部分整理EEG去噪及相关任务中常用的数据集，后续可继续扩展到EEG基础模型预训练数据集、下游解码benchmark数据集和智能体评测数据集。

### 3.1 EEG去噪数据集 (EEG Denoising Datasets)

> 共收录 **3** 条数据集

#### 3.1.1 EEGdenoiseNet

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising |
| 发表期刊 | Journal of Neural Engineering, Vol. 18, No. 5, p. 056057, 2021 |
| 包含噪声类型 | EOG（眼电）、EMG（肌电） |
| 单通道/多通道 | 单通道 |
| 数据量 | 干净EEG片段：4514个；EMG片段：5598个；EOG片段：3400个 |
| 预处理情况 | 每个片段长度为512个采样点，对应2秒（采样率256 Hz） |

#### 3.1.2 MIT-BIH心律失常数据集

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | The impact of the MIT-BIH arrhythmia database / PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals |
| 发表期刊 | IEEE Engineering in Medicine and Biology Magazine, 2001; Circulation, 2000 |
| 包含噪声类型 | ECG（心电） |
| 单通道/多通道 | 单通道 |
| 数据量 | 3600个ECG片段 |
| 预处理情况 | ECG信号经45 Hz低通滤波，重采样至256 Hz，并切割为2秒片段 |

#### 3.1.3 运动伪迹数据集（Kevin T. Sweeney 提供）

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | A methodology for validating artifact removal techniques for physiological signals |
| 发表期刊 | IEEE Transactions on Information Technology in Biomedicine, Vol. 16, No. 5, pp. 918-926, 2012 |
| 包含噪声类型 | 运动伪迹 |
| 单通道/多通道 | 单通道 |
| 数据量 | 2760个运动片段 |
| 预处理情况 | 重采样至256 Hz，切割为2秒片段 |

### 3.2 运动想象数据集 (Motor Imagery Datasets)

> 共收录 **2** 条数据集

#### 3.2.1 BCI Competition IV 2a

| 属性 | 内容 |
|:---:|:---|
| 数据集链接 | [BCI Competition IV - Data set 2a](https://www.bbci.de/competition/iv/#dataset2a) |
| 任务类型 | 四分类运动想象：左手、右手、双脚、舌头 |
| 被试与通道 | 9名被试；22个EEG通道，另含3个EOG通道 |
| 数据组织 | 每名被试包含训练与测试两个session，每个session 288个试次；采样率250 Hz |
| 适用方向 | 运动想象BCI分类、跨被试泛化、预处理敏感性分析、任务导向EEG去噪/伪迹拒绝benchmark |
| 备注 | 经典MI-BCI公开数据集，常用于验证去噪或预处理是否真正改善下游解码性能 |

#### 3.2.2 OpenBMI

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | [A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces](https://academic.oup.com/gigascience/article/8/5/giz002/5304369) |
| 数据集链接 | [GigaDB OpenBMI Dataset](http://gigadb.org/dataset/100542) |
| 任务类型 | 二分类运动想象：左手、右手 |
| 被试与通道 | 54名健康被试；62通道EEG |
| 数据组织 | 包含多session运动想象记录，规模大于BCI Competition IV 2a，适合跨被试和跨session评估 |
| 适用方向 | 大样本MI-BCI解码、跨数据集迁移、domain adaptation、任务导向去噪benchmark |
| 备注 | 适合与BCIC IV 2a共同构建运动想象benchmark，用于检查去噪模型是否提升真实下游任务表现 |

### 3.3 情绪识别数据集 (Emotion Recognition Datasets)

> 共收录 **2** 条数据集

#### 3.3.1 SEED

| 属性 | 内容 |
|:---:|:---|
| 数据集链接 | [SJTU BCMI SEED Dataset](https://bcmi.sjtu.edu.cn/home/seed/) |
| 任务类型 | EEG情绪识别，通常包含正性、中性、负性三类情绪 |
| 被试与通道 | 15名被试；62通道EEG |
| 数据组织 | 每名被试多次实验session，使用电影片段诱发情绪状态 |
| 适用方向 | 情绪识别、跨session/跨被试迁移、EEG表征学习、任务导向预处理与去噪评估 |
| 备注 | 国内EEG情绪识别研究中使用频率很高，适合纳入通用EEG解码benchmark |

#### 3.3.2 DEAP

| 属性 | 内容 |
|:---:|:---|
| 数据集链接 | [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) |
| 文献名称 | [DEAP: A Database for Emotion Analysis using Physiological Signals](https://ieeexplore.ieee.org/document/5871728) |
| 任务类型 | 基于音乐视频诱发的情绪识别/情感计算；常用维度包括valence、arousal、dominance、liking |
| 被试与通道 | 32名被试；32通道EEG，同时包含外周生理信号 |
| 数据组织 | 每名被试观看40段1分钟音乐视频并给出情绪评分 |
| 适用方向 | 情绪识别、跨被试泛化、多模态生理信号建模、通用EEG解码benchmark |
| 备注 | 国际情绪EEG研究的经典数据集，可与SEED共同覆盖情绪识别下游任务 |

---
