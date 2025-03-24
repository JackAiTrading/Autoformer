# Autoformer (NeurIPS 2021)

Autoformer：具有自相关性的分解Transformer用于长期序列预测

时间序列预测是实际应用中的关键需求。受经典时间序列分析和随机过程理论的启发，我们提出了Autoformer作为一个通用的序列预测模型[[论文](https://arxiv.org/abs/2106.13008)]。**Autoformer超越了Transformer家族，首次实现了序列级连接。**

在长期预测中，Autoformer达到了SOTA（最先进水平），在六个基准测试中相对提升了**38%**，涵盖了五个实际应用场景：**能源、交通、经济、天气和疾病**。

:triangular_flag_on_post:**新闻**（2023年8月）Autoformer已被包含在[Hugging Face](https://huggingface.co/models?search=autoformer)中。参见[博客](https://huggingface.co/blog/autoformer)。

:triangular_flag_on_post:**新闻**（2023年6月）Autoformer的扩展版本（[统一深度模型对全球气象站的可解释天气预报](https://www.nature.com/articles/s42256-023-00667-9)）已在《自然机器智能》上作为[封面文章](https://www.nature.com/natmachintell/volumes/5/issues/6)发表。

:triangular_flag_on_post:**新闻**（2023年2月）Autoformer已被包含在我们的[[时间序列库]](https://github.com/thuml/Time-Series-Library)中，该库涵盖长短期预测、插补、异常检测和分类。

:triangular_flag_on_post:**新闻**（2022年2月-2022年3月）Autoformer已部署在[2022年冬奥会](https://en.wikipedia.org/wiki/2022_Winter_Olympics)中，为比赛场馆提供天气预报，包括风速和温度。

## Autoformer vs. Transformers

**1. 深度分解架构**

我们将Transformer改造为深度分解架构，可以在预测过程中逐步分解趋势和季节性成分。

<p align="center">
<img src=".\pic\Autoformer.png" height = "250" alt="" align=center />
<br><br>
<b>图1.</b> Autoformer的整体架构。
</p>

**2. 序列级自相关机制**

受随机过程理论的启发，我们设计了自相关机制，可以发现基于周期的依赖关系并在序列级别聚合信息。这使模型具有固有的对数线性复杂度。这种序列级连接与之前的自注意力家族明显不同。

<p align="center">
<img src=".\pic\Auto-Correlation.png" height = "250" alt="" align=center />
<br><br>
<b>图2.</b> 自相关机制。
</p>

## 开始使用

1. 安装Python 3.6，PyTorch 1.9.0。
2. 下载数据。你可以从[Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing)获取所有六个基准数据集。**所有数据集都经过良好预处理**，可以直接使用。
3. 训练模型。我们在`./scripts`文件夹下提供了所有基准测试的实验脚本。你可以通过以下命令重现实验结果：

```bash
bash ./scripts/ETT_script/Autoformer_ETTm1.sh
bash ./scripts/ECL_script/Autoformer.sh
bash ./scripts/Exchange_script/Autoformer.sh
bash ./scripts/Traffic_script/Autoformer.sh
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/ILI_script/Autoformer.sh
```


4. 特别设计的实现

- **加速自相关：** 我们将自相关机制构建为类似批归一化的块，使其更易于内存访问。详见[论文](https://arxiv.org/abs/2106.13008)。

- **无需位置嵌入：** 由于序列级连接会固有地保持顺序信息，Autoformer不需要位置嵌入，这与Transformers不同。

### 使用Docker重现结果

为了轻松使用Docker、conda和Make重现结果，你可以按照以下步骤操作：
1. 使用`make init`初始化docker镜像。
2. 使用`make get_dataset`下载数据集。
3. 使用`make run_module module="bash scripts/ETT_script/Autoformer_ETTm1.sh"`运行`scripts/`中的每个脚本。
4. 或者一次性运行所有脚本：
```
for file in `ls scripts`; do make run_module module="bash scripts/$script"; done
```

### 简单示例
参见`predict.ipynb`了解工作流程（中文）。

## 主要结果

我们在六个基准测试上进行了实验，涵盖了五大主流应用。我们将我们的模型与十个基线模型进行了比较，包括Informer、N-BEATS等。总体而言，在长期预测设置中，Autoformer达到了SOTA，相对之前基线模型提升了**38%**。

<p align="center">
<img src=".\pic\results.png" height = "550" alt="" align=center />
</p>

## 基线模型

我们将继续添加序列预测模型以扩展此仓库：

- [x] Autoformer
- [x] Informer
- [x] Transformer
- [x] Reformer
- [ ] LogTrans
- [ ] N-BEATS

## 引用

如果你觉得这个仓库有用，请引用我们的论文。

```
@inproceedings{wu2021autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```


## 联系方式

如果你有任何问题或想使用代码，请联系wuhx23@mails.tsinghua.edu.cn。

## 致谢

我们非常感谢以下GitHub仓库提供的宝贵代码库或数据集：

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

