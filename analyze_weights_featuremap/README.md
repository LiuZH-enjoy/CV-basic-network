# 运行步骤

## 1. 数据集拆分

首先要下载数据集，分为两个版本

### 版本一：Windows下

- （1）先在此目录下创建data_set文件夹，然后在data_set文件夹下创建新文件夹"flower_data"

- （2）点击链接下载花分类数据集: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

- （3）解压数据集到flower_data文件夹下

- （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val

  结构如下：

  ```
  ├── flower_data   
         ├── flower_photos（解压的数据集文件夹，3670个样本）  
         ├── train（生成的训练集，3306个样本）  
         └── val（生成的验证集，364个样本） 
  ```

  

### 版本二：用Colab（Linux下）

- （1）先在此目录下创建data_set文件夹，然后在data_set文件夹下创建新文件夹"flower_data"。（colab里可以用linux命令创建，也可以直接像Windows那样鼠标操作新建文件夹。
- （2）此时使用Linux命令下载数据集

```shell
!wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
```

- （3）然后用命令把数据集解压到指定位置

```shell
# 用法：tar zxvf xxx(要解压的数据集) -C xxx(解压到的目录)
!tar zxvf flower_photos.tgz -C ./data_set/flower_data
```

- （4）执行"split_data.py"脚本自动将数据集划分成训练集train和验证集val。

## 2. 先训练AlexNet或ResNet

把训练得到的模型文件resNet34.pth或者AlexNet.pth放在本目录下。

## 3. 运行analyze_feature_map.py和analyze_kernel_weight.py

- 运行analyze_feature_map.py可以查看训练过程中，图像经过某层处理后的效果

- 运行analyze_kernel_weight.py可以查看训练过程中，图像经过某层处理后，各个训练参数的分布情况

