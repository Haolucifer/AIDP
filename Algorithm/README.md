# Inception_v1_v2

## **Inception算法简介** 
2014年，GoogLeNet获得当年ImageNet挑战赛(ILSVRC14)的第一名、GoogLeNet虽然深度只有22层，但大小却比AlexNet和VGG小很多，
GoogleNet参数为500万个，AlexNet参数个数是GoogleNet的12倍，VGGNet参数又是AlexNet的3倍，因此在内存或计算资源有限时，GoogleNet是比较好的选择；
从模型结果来看，GoogLeNet的性能却更加优越。 Inception系列就是GoogLeNet团队设计的一种 “基础神经元”结构，搭建了一个稀疏性、高计算性能的网络结构。
本文档着重介绍inception_v1和inception_v2。

## **Inception_v1**
最原始的inception结构如图所示，该结构将CNN中常用的卷积（1x1，3x3，5x5）、池化操作（3x3）堆叠在一起（卷积、池化后的尺寸相同，将通道相加），一方面增加了网络的宽度，另一方面也增加了网络对尺度的适应性。
不同尺寸的卷积层既能够提取输入特征的细节信息又可以增加特征提取的感受野，同时，池化操作用以减少空间大小，降低过度拟合。

<img src='网络结构图/inception_1.png' width="600px"/>
