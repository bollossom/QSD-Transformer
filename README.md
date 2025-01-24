
# Quantized Spike-driven Transformer ([ICLR25](https://arxiv.org/abs/2501.13492))

[Xuerui Qiu](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Jieyuan Zhang](https://www.ericzhuestc.site/), [Wenjie Wei](), [Honglin Cao](), [Junsheng Guo](), [Rui-Jie Zhu](https://scholar.google.com/citations?user=08ITzJsAAAAJ&hl=zh-CN),[Yimeng Shan](),[Yang Yang](), [Malu Zhang](), [Haizhou Li](https://www.colips.org/~eleliha/)

University of Electronic Science and Technology of China

Institute of Automation, Chinese Academy of Sciences


### Instructions for running the code:
> **Quantization Model ImageNet From Scratch**: See [Train_Base.md](classification/readme.md).\
> **Object Detection**: See [Detection.md](det/readme.md).\
> **Semantic Segmentation**: See [Segementation.md](seg/readme.md)


:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.

## Abstract
The ambition of brain-inspired Spiking Neural Networks (SNNs) is to become a low-power alternative to traditional Artificial Neural Networks (ANNs). This work addresses two major challenges in realizing this vision: the performance gap between SNNs and ANNs, and the high training costs of SNNs. We identify intrinsic flaws in spiking neurons caused by binary firing mechanisms and propose a Spike Firing Approximation (SFA) method using integer training and spike-driven inference. This optimizes the spike firing pattern of spiking neurons, enhancing efficient training, reducing power consumption, improving performance, enabling easier scaling, and better utilizing neuromorphic chips. We also develop an efficient spike-driven Transformer architecture and a spike-masked autoencoder to prevent performance degradation during SNN scaling. On ImageNet-1k, we achieve state-of-the-art top-1 accuracy of 78.5\%, 79.8\%, 84.0\%, and 86.2\% with models containing 10M, 19M, 83M, and 173M parameters, respectively. For instance, the 10M model outperforms the best existing SNN by 7.2\% on ImageNet, with training time acceleration and inference energy efficiency improved by 4.5x and 3.9x, respectively. We validate the effectiveness and efficiency of the proposed method across various tasks, including object detection, semantic segmentation, and neuromorphic vision tasks. This work enables SNNs to match ANN performance while maintaining the low-power advantage, marking a significant step towards SNNs as a general visual backbone.

![avatar](./images/main.png)

## Results
We address the performance and training consumption gap between SNNs and ANNs. A key contribution is identifying the mechanistic flaw of binary spike firing in spiking neurons. To overcome these limitations, we propose a Spike Firing Approximation (SFA) method. This method is based on integer training and spike-driven inference, aiming to optimize the spike firing pattern of spiking neurons. Our results demonstrate that optimization the spike firing pattern leads to comprehensive improvements in SNNs, including enhanced training efficiency, reduced power consumption, improved performance, easier scalability, and better utilization of neuromorphic chips. Additionally, we develop an efficient spike-driven Transformer architecture and a spike masked autoencoder to prevent performance degradation during SNN scaling. By addressing the training and performance challenges of large-scale SNNs, we pave the way for a new era in neuromorphic computing.

![avatar](./images/fig.png)


## Contact Information

```
@inproceedings{qiu2025quantized,
  title={Quantized Spike-driven Transformer},
  author={Qiu Xuerui, Zhang Jieyuan, Wei Wenjie, Cao Honglin, Guo Junsheng, Zhu Rui-Jie, Shan Yimeng, Yang Yang, Zhang Malu, Li Haizhou},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5J9B7Sb8rO}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact  `qiuxuerui2024@ia.ac.cn`.

## Acknowledgement
The object detection and semantic segmentation parts are based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) respectively. Thanks for their wonderful work.
