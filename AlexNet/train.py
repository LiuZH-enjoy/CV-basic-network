import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

def main():
    # 选择device，如果可以gpu则选择gpu，否则就用cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="./AlexNet/runs/flower_experiment")

    # 定义一个图形增强的dict，包含train和val两个变换类型，具体怎么进行数据增强查看transforms的API，这是最好的办法
    # RandomResizedCrop : 选定图片某一随机位置，然后按照size进行裁剪
    # RandomHorizontalFlip : 图形以一定概率进行随机水平翻转
    # ToTensor : Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize : 对每个channel指定means和std，每个channel分别标准化
    data_transform = {"train":transforms.Compose([transforms.RandomResizedCrop(224), 
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,.05),(0.5,0.5,.05))]),
                "val":transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}
    data_root = os.getcwd()  # get data root path
    image_path = os.path.join(data_root, "AlexNet", "data_set", "flower_data")  # flower data set path
    # 判断文件是否存在
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # ImageFolder使用参考：https://www.cnblogs.com/wanghui-garcia/p/10649364.html
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                        transform=data_transform["train"])
    # 统计所读训练集的图片数量
    train_num = len(train_dataset)
    # 成员变量：self.class_to_idx - 类名对应的 索引
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('./AlexNet/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size>1 else 0, 8]) # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 将数据集变成loader，再进行训练
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, 
                            batch_size=4, 
                            shuffle=False, 
                            num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 实例化网络
    net = AlexNet(num_classes=5, init_weights=True)
    #print(net.features[0])
    # 牢记要转到device
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(net, init_img)

    epochs=10
    # 模型保存路径
    save_path = './AlexNet/AlexNet.pth'
    # 记录模型最高准确率
    best_acc = 0.0
    # 这里的steps = 训练(验证)集数据总量 / 训练(验证)集的batch_size。也就是按照batch_size去读数据，需要读多少次
    train_steps = len(train_loader)
    validate_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        training_loss=0.0
        validate_loss=0.0
        # 利用tqdm为数据读取中，提供进度条显示，有可视化与提醒作用
        # 用法：tqdm(data_loader)
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs=net(images.to(device))
            train_step_loss = loss_function(outputs, labels.to(device))
            train_step_loss.backward()
            optimizer.step()

            # training_loss记录的是每个epoch的总loss，train_step_loss记录的是一个epoch里每个bath_size的loss
            training_loss += train_step_loss.item()
            # desc功能，显示进度条的同时输出每个epoch里，每个batch_size的loss
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, train_step_loss)

        # validate
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            validate_step_loss = loss_function(outputs, val_labels.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            validate_loss += validate_step_loss.item()
            val_bar.desc = "validate epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, validate_step_loss)
        val_accuracy = acc / val_num

        # 每个epoch总结一次，输出train_loss和validate_loss的平均loss以及在验证机上的准确率
        print('[epoch %d] train_loss: %.3f validate_loss: %.3f val_accuracy: %.3f' % 
            (epoch+1, training_loss/train_steps, validate_loss/validate_steps, val_accuracy))
        
        # add loss, acc into tensorboard
        tags = ["Loss/train_loss", "Loss/val_loss", "accuracy"]
        tb_writer.add_scalar(tags[0], training_loss/train_steps, epoch)
        tb_writer.add_scalar(tags[1], validate_loss/validate_steps, epoch)
        tb_writer.add_scalar(tags[2], val_accuracy, epoch)
        # add conv1 weights into tensorboard
        tb_writer.add_histogram(tag="features/conv1",
                                values=net.features[0].weight,
                                global_step=epoch)

        # 如果该epoch训练出来的模型在验证集上的准确率比最高的acc要高，怎保存该准确率并且保存该模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
        
    print('Finished Training')


if __name__ == '__main__':
    main()
