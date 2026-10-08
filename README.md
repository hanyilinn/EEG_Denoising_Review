# EEG Denoising Methods & Journals Survey

整理EEG去噪领域的传统方法、深度学习方法以及可投稿期刊调研结果。

网页浏览版：[https://hanyilinn.github.io/EEG_Denoising_Review/](https://hanyilinn.github.io/EEG_Denoising_Review/)

## 项目结构

```
EEG_Denoising_Review/
├── README.md
├── data/
│   ├── EEG_denoising_traditional.csv   # 传统去噪方法
│   ├── EEG_denoising_deeplearning.csv  # 深度学习去噪方法
│   └── journals.csv                     # 可投稿期刊列表
```

## 数据说明

### 1. 传统去噪方法 (EEG_denoising_traditional.csv)

| 字段 | 说明 |
|------|------|
| 序号 | 编号 |
| 名称 | 方法简称 |
| 发表时间 | 论文发表年份 |
| 主要思路 | 方法核心思想 |
| 文章名称 | 论文标题 |
| 发表期刊 | 期刊名称 |
| 是否开源 | 是否开源代码 |
| 作者单位 | 研究机构 |
| 备注 | 其他备注 |

### 2. 深度学习去噪方法 (EEG_denoising_deeplearning.csv)

| 字段 | 说明 |
|------|------|
| 序号 | 编号 |
| 名称 | 网络模型名称 |
| 发表时间 | 论文发表年份 |
| 主要思路 | 方法核心思想 |
| 文章名称 | 论文标题 |
| 发表期刊 | 期刊名称/会议名称 |
| 是否开源 | 是否开源代码 |
| 作者单位 | 研究机构 |
| 备注 | 其他备注 |

### 3. 期刊调研 (journals.csv)

| 字段 | 说明 |
|------|------|
| 序号 | 编号 |
| 期刊名称 | 期刊全称 |
| 期刊简称 | 期刊缩写 |
| 大类学科 | 学科分类 |
| 2026年新锐分区 | 独立第三方新锐分区；不是中科院分区 |
| 影响因子（2025） | 2026年发布的2025年度JIF；数值附来源链接 |
| CiteScore（2025） | 2026年发布的2025年度CiteScore；数值附来源链接 |
| 年文章数 | 年发文量 |
| 出版机构 | 出版社 |
| 备注 | 其他备注 |

## 贡献

欢迎提交Issue或Pull Request来补充完善数据。

## 许可

MIT License
