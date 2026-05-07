# EEG Denoising Methods & Journals Survey
👁️ EEG去噪方法与期刊调研

整理EEG去噪领域的方法、综述、期刊和数据集调研结果，便于快速了解该领域进展。

---

## 目录

1. [深度学习EEG去噪方法](#1-深度学习eeg去噪方法-deep-learning-eeg-denoising)
2. [传统EEG去噪方法](#2-传统eeg去噪方法-traditional-eeg-denoising)
3. [EEG去噪相关综述](#3-eeg去噪相关综述-eeg-denoising-reviews)
4. [EEG去噪结合下游任务的研究](#4-eeg去噪结合下游任务的研究-eeg-denoising-with-downstream-tasks)
5. [EEG去噪可投稿期刊（已有EEG去噪论文发表）](#5-eeg去噪可投稿期刊-期刊已有eeg去噪论文发表)
6. [EEG领域下其他可供选择的期刊](#6-eeg领域下其他可供选择的期刊-other-journals-in-eeg-field)
7. [EEG去噪研究适合使用的数据集](#7-eeg去噪研究适合使用的数据集-eeg-denoising-datasets)

---

## 1. 深度学习EEG去噪方法 (Deep Learning EEG Denoising)


### 1.1 时间线总览

> 共收录 **61** 条记录（截至2026年）

| 年份 | 方法数量 |
|:---:|:---:|
| 2020 | 1 |
| 2021 | 3 |
| 2022 | 6 |
| 2023 | 6 |
| 2024 | 14 |
| 2025 | 28 |
| 2026 | 6 |

### 1.2 方法详情

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | 1D-ResCNN | 2020 | 一维残差卷积神经网络端到端去噪... | A novel end-to-end 1D-ResCNN model to remove artifact from EEG signals | Neurocomputing | - | Shaanxi Normal University | - |
| 2 | FCNN | 2021 | EEG去噪基准数据集+基准网络... | EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 3 | SimpleCNN | 2021 | EEG去噪基准数据集+基准网络... | EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 4 | ComplexCNN | 2021 | EEG去噪基准数据集+基准网络... | EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 5 | RNN | 2021 | EEG去噪基准数据集+基准网络... | EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG denoising | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 6 | Novel CNN | 2021 | 特征维数逐渐上升和时序降采样抑制过拟合... | A Novel Convolutional Neural Network Model to Remove Muscle Artifacts from EEG | ICASSP2021 | 是 | Southern University of Science and Technology | - |
| 7 | EEGANet | 2021 | 首次使用GAN进行降噪... | EEGANet: Removal of ocular artifacts from the EEG signal using generative adversarial networks | IEEE JBHI | 是 | Vidyasirimedhi Institute of Science and Technology | - |
| 8 | 1-D CNN auto-encoder | 2022 | 利用切比切夫矩形式的正交特征的分数阶压缩一维CNN自编码器... | Orthogonal features based EEG signals denoising using fractional and compressed onedimensional CNN AutoEncoder | IEEE TNSRE | - | DA-IICT | - |
| 9 | IC-U-Net | 2022 | 结合ICA和U-Net... | IC-U-Net: a U-Net-based denoising autoencoder using mixtures of independent components for automatic EEG artifact removal | NeuroImage | 是 | National Tsing Hua University | - |
| 10 | GRU-MARSC | 2022 | 鉴别-去除两阶段、基于GRU... | Two-Stage Intelligent Multi-Type Artifact Removal for Single-Channel EEG Settings: A GRU Autoencoder Based Approach | IEEE TBME | - | University of Science and Technology of China | - |
| 11 | Freq_Enhan | 2022 | 频域信息增强网络去噪能力... | Frequency Information Enhanced Deep EEG Denoising Network for Ocular Artifact Removal | IEEE Sensors Journal | - | University of Science and Technology of China | - |
| 12 | EEGDnet | 2022 | 2-D Transformer结构... | EEGDnet: Fusing non-local and local self-similarity for EEG signal denoising with transformer | Computers in Biology and Medicine | 是 | University of Electronic Science and Technology of China | - |
| 13 | DeepSeparator | 2022 | 编码器-解码器架构... | Embedding decomposition for artifacts removal in EEG signals | Journal of Neural Engineering | 是 | Southern University of Science and Technology | - |
| 14 | DuoCL | 2023 | 双尺度CNN-LSTM网络... | EEG Reconstruction With a Dual-Scale CNN-LSTM Model for Deep Artifact Removal | IEEE JBHI | 是 | Wuhan University | - |
| 15 | SDNet | 2023 | 先分割再去噪... | A Segmentation-Denoising Network for Artifact Removal From Single-Channel EEG | IEEE Sensors Journal | 是 | University of Science and Technology of China | - |
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

---

## 2. 传统EEG去噪方法 (Traditional EEG Denoising)

> 主要为2025年及之后的研究，共收录 **9** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | DT-SFDF | 2024 | 二阶盲识别（SOBI）和典型相关分析（CCA）相融合的方法提高源分离质量，采用自适应阈值法对伪源进行... | Removal of Ocular and Muscular Artifacts From Multi-Channel EEG Using Improved Spatial-Frequency Filtering | IEEE JBHI | - | Fuzhou University | 福州大学 李玉榕 教授 |
| 2 | MASR | 2025 | 多模态伪迹子空间重建,结合多模态特征提取技术、动态自适应阈值和通道显著性度量，实现了对脑电信号中复杂... | Adaptive Thresholding in EEG Artifact Removal Through Multimodal Fusion: A Multimodal Artifact Subspace Reconstruction Approach | IEEE TETCI | - | Xi'an Jiaotong-Liverpool University | - |
| 3 | ARMBR | 2025 | 通过多元后向回归和简化的时间锁定参考信号，在需要最少训练数据的情况下，实现了对瞬目伪影的自动检测和有... | Artifact-reference multivariate backward regression (ARMBR): A novel method for EEG blink artifact removal with minimal data requirements | Journal of Neural Engineering | - | Weill Cornell Medicine | - |
| 4 | STMS | 2025 | 频谱模板匹配与抑制算法，通过K均值聚类构建频谱模板集，并根据欧氏距离自适应选择最相关模板，然后在频域... | Adaptive EOG Artifact Removal Algorithm Using A Spectrum Magnitude Template Matching for Single-Channel EEG | IEEE Sensors Journal | - | Fudan University | 复旦大学 刘骁 |
| 5 | ARICB | 2025 | Chirp Atom分解+分数小波变换FrWT+自适应阈值设计... | Denoising method for EEG-based expert system in fractional wavelet domain using adaptive residual-incorporating chirp-based model | ESWA | - | Beijing Jiaotong University | 北京交通大学 赵文山 |
| 6 | NSRM | 2025 | 通过一种非凸的广义极小极大凹（GMC）惩罚项替代传统的L1正则化，在保持优化问题凸性的前提下，更精准... | Nonconvex Sparse Regularization Method for Eyeblink Artifact Suppression from Single Channel EEG Signals | IEEE TIM | - | Beijing Institute of Technology | 北京理工大学 机车学院 |
| 7 | GED | 2025 | 广义特征分解（GED）在极端低信噪比条件下（如行走、慢跑时）去除高振幅运动伪影的有效性，并成功将清洗... | Contrast-based artifact removal enables microstate analysis in ambulatory EEG | IEEE TBME | - | The University of British Columbia | 加拿大 不列颠哥伦比亚大学 生物医学工程学院 |
| 8 | GVICA | 2026 | “GWO优化VMD + 熵筛选ICA”的无参考、多层级自适应去噪框架... | GVICA: A multi-channel EEG hierarchical noise reduction framework based on GWO dynamically optimized VMD-ICA fusion | IEEE TBME | - | Guangdong University of Technology | 引用了ASTI-Net |
| 9 | GED | 2026 | 利用广义特征分解，通过对比静息态 EEG（干净参考）和运动态 EEG（含伪影）的协方差矩阵，识别并去... | Suppressing Non-Stationary Motion Artefacts in Mobile EEG Using Generalized Eigenvalue Decomposition | Sensors | - | University "G. d'Annunzio" of Chieti-Pescara, Chieti, Italy | 用了打乒乓球的数据 |

---

## 3. EEG去噪相关综述 (EEG Denoising Reviews)

> 共收录 **7** 条综述

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | 综述 | 2015 | EEG伪迹去除技术综述与指南 | [EEG artifact removal—state-of-the-art and guidelines](https://iopscience.iop.org/article/10.1088/1741-2560/12/3/031001/meta) | Journal of Neural Engineering | - | - | - |
| 2 | 综述 | 2016 | 脑电信号伪迹检测与去除方法综述 | [Methods for artifact detection and removal from scalp EEG: A review](https://www.sciencedirect.com/science/article/pii/S098770531630199X) | Neurophysiologie Clinique/Clinical Neurophysiology | - | - | - |
| 3 | 综述 | 2018 | EEG伪迹与噪声去除文献综述 | [Artifacts and noise removal for electroencephalogram (EEG): A literature review](https://ieeexplore.ieee.org/abstract/document/8405493/) | IEEE Symposium on Computer Applications & Industrial Electronics (ISCAIE) | - | - | - |
| 4 | 综述 | 2019 | 肌电伪迹去除方法综述与建议 | [Removal of Muscle Artifacts From the EEG: A Review and Recommendations](https://ieeexplore.ieee.org/abstract/document/8672623) | IEEE Sensors Journal | - | - | - |
| 5 | 综述 | 2019 | EEG伪迹去除综述 | [Removal of artifacts from EEG signals: a review](https://www.mdpi.com/1424-8220/19/5/987) | Sensors | - | - | - |
| 6 | 综述 | 2020 | EEG信号伪迹去除方法综述 | [Methods for removal of artifacts from EEG signal: A review](https://iopscience.iop.org/article/10.1088/1742-6596/1706/1/012093/meta) | Journal of Physics: Conference Series | - | - | - |
| 7 | 综述 | 2026 | - | Physiological artifacts removal in EEG signals: a comprehensive overview of conventional to deep learning methods to support brain health monitoring | Cogent Engineering | - | - | 印度马尼帕尔理工学院，综述文章 |

---

## 4. EEG去噪结合下游任务的研究 (EEG Denoising with Downstream Tasks)

> 共收录 **6** 条记录

| 序号 | 名称 | 发表时间 | 主要思路 | 文章名称 | 发表期刊 | 开源 | 作者单位 | 备注 |
|:---:|:---:|:---:|:---|:---|:---:|:---:|:---|:---|
| 1 | - | 2021 | 少通道SSVEP，自采10例数据，比较了无EMG参考的BSS方法和有辅助的EMG方法 | Muscle Artifact Removal Toward Mobile SSVEP-Based BCI: A Comparative Study | IEEE TIM | - | University of Science and Technology of China | - |
| 2 | - | 2023 | 评估伪迹去除对于基于CNN的异常和正常脑电图数据分类算法性能的影响 | The effect of artifact rejection on the performance of a convolutional neural network based algorithm for binary EEG data classification | BSPC | - | Amsterdam UMC location University of Amsterdam | - |
| 3 | Diff-SleepNet | 2024 | 基于DDPM的信号去噪模块，在去噪数据集上预训练，在迁移到睡眠数据集上微调，基于短时傅里叶变换和Transformer组合的特征提取模块，多模态融合模块 | Sleep Stage Classification with Multi-Modal Fusion and Denoising Diffusion Model | IEEE JBHI | [是](https://github.com/Odysseus0816/Diff-SleepNet) | Dalian University of Technology | 空链接 |
| 4 | TOED-GAN | 2024 | 任务导向，面向BCI实际场景，有自采数据集，模拟移动场景、故意加了更多噪声，基于GAN，生成器U-Net | Task-oriented EEG denoising generative adversarial network for enhancing SSVEP-BCI performance | JNE | 否 | National University of Defense Technology | 通讯作者胡德文院士 |
| 5 | FDC-Net | 2025 | EEG去噪任务和多维情绪解码 | FDC-Net: Rethinking the association between EEG artifact removal and multi-dimensional affective computing | Arxiv | 是 | Beijing University of Technology | - |
| 6 | TOL | 2026 | 通过盲源分离分解EEG成分，学习型选择器预测保留概率并加权重建；借助下游任务损失协同优化，仅需任务标签、无需干净参考。 | Task-Oriented Learning for Automatic EEG Denoising | IEEE TIM | 否 | 中国科学院自动化研究所 | - |

---

## 5. EEG去噪可投稿期刊（已有EEG去噪论文发表）

> 共收录 **29** 条期刊（按学科分类）

### 5.1 医学

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Journal of Biomedical and Health Informatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221020) | IEEE JBHI | 二区TOP | 一区TOP | 6.8 | 659 | IEEE | - |
| 2 | [NeuroImage](https://www.sciencedirect.com/journal/neuroimage) | NeuroImage | 二区TOP | 二区TOP | 4.5 | 447 | Elsevier | - |
| 3 | [Biomedical Signal Processing and Control](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) | BSPC | 二区 | 二区TOP | 4.9 | 1370 | Elsevier | - |
| 4 | [IEEE Transactions on Biomedical Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10) | IEEE TBME | 二区 | 二区TOP | 4.5 | 331 | IEEE | - |
| 5 | [IEEE Transactions on Neural Systems and Rehabilitation Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7333) | IEEE TNSRE | 二区 | 二区TOP | 5.2 | 407 | IEEE | - |
| 6 | [Computers in Biology and Medicine](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine) | CiBM | 二区 | - | 6.3 | 274 | Elsevier | On Hold，2025年11月踢出SCI |
| 7 | [CNS Neuroscience & Therapeutics](https://onlinelibrary.wiley.com/journal/17555949) | CNSNT | 二区 | 二区TOP | 5 | 490 | Wiley-Blackwell | - |
| 8 | [Journal of Neural Engineering](https://iopscience.iop.org/journal/1741-2552) | JNE | 三区 | 二区TOP | 3.8 | 320 | IOP Publishing | - |
| 9 | [IEEE Journal of Translation Engineering in Health and Medicine](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221036) | IEEE JTEHM | 三区 | 二区 | 4.4 | 69 | IEEE | - |
| 10 | [Frontiers in Human Neuroscience](https://www.frontiersin.org/journals/human-neuroscience) | FHN | 四区 | 三区 | 2.7 | 462 | Frontiers | - |
| 11 | [Computer Methods in Biomechanics and Biomedical Engineering](https://www.tandfonline.com/journals/gcmb20) | CMBBE | 四区 | 四区 | 1.6 | 229 | Taylor & Francis | - |
| 12 | [Neuroscience](https://www.sciencedirect.com/journal/neuroscience) | Neuroscience | 四区 | 四区 | 2.8 | 478 | Elsevier | - |
| 13 | [Biomedical Physics & Engineering Express](https://iopscience.iop.org/journal/2057-1976) | BPEE | 四区 | - | 1.6 | 279 | IOP Publishing | - |

### 5.2 计算机科学

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385) | TNNLS | 一区TOP | 二区TOP | 8.9 | 869 | IEEE | - |
| 2 | [Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence) | EAAI | 一区TOP | 一区TOP | 8 | 1909 | Elsevier | - |
| 3 | [Expert Systems With Applications](https://www.sciencedirect.com/journal/expert-systems-with-applications) | ESWA | 一区TOP | 一区TOP | 7.5  | 2925 | Elsevier | - |
| 4 | [Knowledge-Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems) | KBS | 一区TOP | 一区TOP | 7.6 | 1354 | Elsevier | - |
| 5 | [IEEE Internet of Things Journal](https://ieee-iotj.org/) | IEEE IoTJ | 二区TOP | 一区TOP | 8.9 | 2984 | IEEE | - |
| 6 | [IEEE Transactions on Consumer Electronics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=30) | IEEE TCE | 二区 | 一区TOP | 10.9 | 699 | IEEE | 自引率52.3% |
| 7 | [IEEE Transactions on Emerging Topics in Computational Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7755) | IEEE TETCI | 二区 | 二区 | 6.5 | 419 | IEEE | - |
| 8 | [IEEE Transactions on Cognitive and Development systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7274989) | IEEE TCDS | 三区 | 二区 | 4.9 | 171 | IEEE | - |
| 9 | [Electronics](https://www.mdpi.com/journal/electronics) | - | 四区 | - | 2.6 | 5022 | MDPI | - |
| 10 | [Signal Image and Video Processing](https://www.springer.com/journal/11760) | SIVP | 四区 | - | 2.1 | 634 | Springer | - |

### 5.3 工程技术

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Advanced Engineering Informatics](https://www.sciencedirect.com/journal/advanced-engineering-informatics) | AdvEI | 一区TOP | 一区TOP | 9.9 | 658 | Elsevier | - |
| 2 | [IEEE Transactions on Instrumentation and Measurement](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=19) | IEEE TIM | 二区 | 二区TOP | 5.9 | 2532 | IEEE | - |
| 3 | [Cogent Engineering](https://www.tandfonline.com/journals/oaen20) | \ | 四区 | 三区 | 2.5 | 356 | Taylor & Francis | - |
| 4 | [Technologies](https://www.mdpi.com/journal/technologies) | \ | 三区 | 三区 | 3.6 | 262 | MDPI | - |

### 5.4 综合性期刊

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Sensors Journal](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7361) | IEEE SJ | 三区 | 二区 | 4.5 | 3965 | IEEE | - |
| 2 | [Sensors](https://www.mdpi.com/journal/sensors) | Sen. | 三区 | 三区 | 3.5 | 8165 | MDPI | - |


---

## 6. EEG领域下其他可供选择的期刊 (Other Journals in EEG Field)

> 共收录 **35** 条期刊（按学科分类）

### 6.1 计算机科学

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Pattern Analysis and Machine Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) | IEEE TPAMI | 一区TOP | 一区TOP | 18.6 | 732 | IEEE | - |
| 2 | [Cyborg and Bionic Systems](https://spj.science.org/journal/cbs) | CBS | 一区TOP | 一区TOP | 18.1 | 65 |  AAAS | 中国科学院自动化研究所与美国科学促进会合作出版 |
| 3 | [Information Fusion](https://www.sciencedirect.com/journal/information-fusion) | Inf. Fus. | 一区TOP | 一区TOP | 15.5  | 554 | Elsevier | - |
| 4 | [IEEE Transactions on Knowledge and Data Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=69) | IEEE TKDE | 一区TOP | 二区TOP | 10.4 | 666 | IEEE | - |
| 5 | [IEEE Transactions on Cybernetics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221036) | IEEE TCYB | 一区TOP | 一区TOP | 10.5 | 441 | IEEE | - |
| 6 | [IEEE Transactions on Industrial Informatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9424) | IEEE TII | 一区TOP | 一区TOP | 9.9 | 1132 | IEEE | - |
| 7 | [IEEE Transactions on Multimedia](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046) | IEEE TMM | 一区TOP | 一区TOP | 9.7 | 872 | IEEE | - |
| 8 | [IEEE Transactions on Systems Man Cybernetics-Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221037) | IEEE TSMC | 一区TOP | 一区TOP | 8.7 | 667 | IEEE | - |
| 9 | [Pattern Recognition](https://www.sciencedirect.com/journal/pattern-recognition) | PR | 一区TOP | 一区TOP | 7.6 | 966 | Elsevier | - |
| 10 | [Neural Networks](https://www.sciencedirect.com/journal/neural-networks) | NN | 二区TOP | 二区TOP | 6.3 | 856 | Elsevier | - |
| 11 | [Applied Soft Computing](https://www.sciencedirect.com/journal/applied-soft-computing) | ASC | 二区TOP | 二区TOP | 6.6 | 1269 | Elsevier | - |
| 12 | [Neurocomputing](https://www.sciencedirect.com/journal/neurocomputing) | Neurocomputing | 二区 | 二区TOP | 6.5 | 1629 | Elsevier | - |
| 13 | [IEEE Transactions on Automation Science and Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8856) | IEEE TASE | 二区 | 二区TOP | 6.4 | 840 | IEEE | - |
| 14 | [IEEE Transactions on Human-Machine Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221038) | IEEE THMS | 二区 | 二区 | 4.4 | 78 | IEEE | - |
| 15 | [IEEE Signal Processing Letters](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=97) | IEEE SPL | 三区 | 三区 | 3.9 | 635 | IEEE | - |
| 16 | [Applied Intelligence](https://www.springer.com/journal/10489) | AppI | 三区 | - | 3.5 | 635 | Springer | - |

### 6.2 医学

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Journal of NeuroEngineering and Rehabilitation](https://jneuroengrehab.biomedcentral.com/) | JNER | 一区TOP | 二区TOP | 5.2 | 225 | BioMed Central | - |
| 2 | [Artificial Intelligence in Medicine](https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine) | AIM | 二区TOP | 二区TOP | 6.2 | 207 | Elsevier | - |
| 3 | [Human Brain Mapping](https://onlinelibrary.wiley.com/journal/10970193) | HBM | 二区 | 二区TOP | 3.3 | 339 | Wiley-Liss | - |
| 4 | [Journal of Biomedical Informatics](https://www.sciencedirect.com/journal/journal-of-biomedical-informatics) | JBI | 二区 | 二区 | 4.5 | 146 | Elsevier | - |
| 5 | [IEEE Transactions on Biomedical Circuits and Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4156126) | IEEE TBCS | 二区 | 二区 | 4.9 | 102 | IEEE | - |
| 6 | [Computer Methods and Programs in Biomedicine](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine) | CMPB | 二区 | 二区TOP | 4.8 | 440 | Elsevier | - |
| 7 | [IEEE Transactions on Medical Robotics and Bionics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8263503) | IEEE TMRB | 三区 | 二区 | 3.8 | 156 | IEEE | - |
| 8 | [Clinical Neurophysiology](https://www.sciencedirect.com/journal/clinical-neurophysiology) | - | 三区 | - | 3.6 | 234 | Elsevier | - |
| 9 | [Frontiers in Neuroscience](https://www.frontiersin.org/journals/neuroscience) | Fron. Neur. | 三区 | - | 3.2 | 1118 | Frontiers | - |
| 10 | [Cognitive Neurodynamics](https://www.springer.com/journal/11571) | CN | 四区 | 二区 | 3.9 | 126 | Springer | - |

### 6.3 工程技术

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE Transactions on Signal Processing](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=78) | IEEE TSP | 二区TOP | 二区TOP | 5.8 | 375 | IEEE | - |
| 2 | [Computers & Industrial Engineering](https://www.sciencedirect.com/journal/computers-and-industrial-engineering) | CIE | 二区TOP | 一区TOP | 6.5 | 812 | Elsevier | - |

### 6.4 综合性期刊

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [Advanced Science](https://onlinelibrary.wiley.com/journal/21983844) | Adv. Sci. | 一区TOP | 一区TOP | 14.1 | 3290 | Wiley-VCH | - |
| 2 | [Science Advances](https://www.science.org/journal/sciadv) | Sci. Adv. | 一区TOP | 一区TOP | 12.5 | 2263 |  AAAS | - |
| 3 | [Scientific Data](https://www.nature.com/sdata/) | Sci. Dat. | 二区 | - | 6.9 | 1405 | Springer Nature | Open Access |
| 4 | [Scientific Reports](https://www.nature.com/srep/) | Sci. Rep. | 三区 | - | 3.9 | 31052 | Springer Nature | Open Access |
| 5 | [PLoS One](https://journals.plos.org/plosone/) | PLoS One | 三区 | - | 2.6 | 16608 | Public Library of Science | Open Access |

### 6.5 生物学

| 序号 | 期刊名称 | 期刊简称 | 2025年中科院分区 | 2026年中科院分区 | 影响因子 | 年文章数 | 出版机构 | 备注 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| 1 | [IEEE-ACM Transactions on Computational Biology and Bioinformatics](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=8857) | IEEE TCBB | 三区 | - | 3.4 | 228 | IEEE | - |


---

## 7. EEG去噪研究适合使用的数据集 (EEG Denoising Datasets)

> 共收录 **3** 条数据集

### 7.1 EEGdenoiseNet

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | EEGdenoiseNet: A benchmark dataset for deep learning solutions of EEG denoising |
| 发表期刊 | Journal of Neural Engineering, Vol. 18, No. 5, p. 056057, 2021 |
| 包含噪声类型 | EOG（眼电）、EMG（肌电） |
| 单通道/多通道 | 单通道 |
| 数据量 | 干净EEG片段：4514个；EMG片段：5598个；EOG片段：3400个 |
| 预处理情况 | 每个片段长度为512个采样点，对应2秒（采样率256 Hz） |

### 7.2 MIT-BIH心律失常数据集

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | The impact of the MIT-BIH arrhythmia database / PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals |
| 发表期刊 | IEEE Engineering in Medicine and Biology Magazine, 2001; Circulation, 2000 |
| 包含噪声类型 | ECG（心电） |
| 单通道/多通道 | 单通道 |
| 数据量 | 3600个ECG片段 |
| 预处理情况 | ECG信号经45 Hz低通滤波，重采样至256 Hz，并切割为2秒片段 |

### 7.3 运动伪迹数据集（Kevin T. Sweeney 提供）

| 属性 | 内容 |
|:---:|:---|
| 文献名称 | A methodology for validating artifact removal techniques for physiological signals |
| 发表期刊 | IEEE Transactions on Information Technology in Biomedicine, Vol. 16, No. 5, pp. 918-926, 2012 |
| 包含噪声类型 | 运动伪迹 |
| 单通道/多通道 | 单通道 |
| 数据量 | 2760个运动片段 |
| 预处理情况 | 重采样至256 Hz，切割为2秒片段 |

---

## 贡献

欢迎提交Issue或Pull Request来补充完善数据。

## 许可

MIT License
