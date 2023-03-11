# EGES

Enhanced Graph Embedding with Side Information

paper: [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)

数据集：使用电商比赛中的行为数据及商品数据  action_head.csv，jdata_product.csv

## 数据流程

1. 分析用户历史行为日志，构建用户点击序列sessions；

2. 根据sessions构建DGL物品关系图；

3. 基于Node2vec/Random Walk游走dgl.sampling.node2vec_random_walk/random_walk，生成正样本；
   使用torch.multinomial()生成负样本

4. 实现EGES模型

5. 利用Link Prediction测试模型，使用dgl.dataloading.negative_sampler.Uniform(num_negative)进行负采样

6. 生成embedding并可视化，进行冷启动测试

## 环境配置

pytorch = 1.13.0

dgl = 1.0.1.cu116

networkx

## 运行示例

见demo.ipynb

## Embedding可视化

![](https://pic.imgdb.cn/item/640c66e7f144a0100746f946.jpg)

![](https://pic.imgdb.cn/item/640c671af144a01007477d26.jpg)

![](https://pic.imgdb.cn/item/640c667df144a0100745e34c.jpg)

## Reference

https://github.com/Wang-Yu-Qing/EGES

https://github.com/wangzhegeek/EGES