# 运行步骤

## 1. 先训练AlexNet或ResNet

具体如何训练，请查看上级目录中的AlexNet和ResNet。

把一张tulip.jpg图片放到本目录下。

把训练得到的模型文件resNet34.pth或者AlexNet.pth放在本目录下。

## 2. 运行analyze_feature_map.py和analyze_kernel_weight.py

运行期间有可能出现找不到路径的错误，应该是路径索引的问题，自己按照自己的文件放置情况来修改。

- 运行analyze_feature_map.py可以查看训练过程中，图像经过某层处理后的效果。

- 运行analyze_kernel_weight.py可以查看训练过程中，图像经过某层处理后，各个训练参数的分布情况。

