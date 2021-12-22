
# 整体模型框架
用源域数据训练一个BIO分类器，随后在目标域测试，取BIO分类器得到的BIembedding合并进行kmeans聚类

## Data
SNIPS AddToPlaylist作为目标域；
训练集和验证集中不包含目标域数据；
测试集中仅包含目标域数据；

## Config

## Train√
BIO三分类；

## Predict√
```
outputs = (pandas) ["tokens","true_labels","predict_labels","encoder_out"]
```
- tokens(list)
- true_labels，里边的NS如何设计更方便？(list)
- predict_labels，是传统softmax得到的预测标签。用于后续覆写。(list)
- encoder_out,mask处理后的，维度可以对齐了。用于后续GDA拟合。（np）

将得到的embdding用于kmeans聚类


## OOD detection √
1. GDA-Multi_min √
2. Modify the prediction
    - 按sentence形式
    - 避免 OI 
    - ns 改为 B-ns I-ns
3. ROSE：√
- 主要是相邻两句话之间B- I-改写，会有点小影响。
- 多个NS类型邻接的话，也要注意，true_labels。

## Result

