import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {"train":transforms.Compose([transforms.RandomResizedCrop(224), 
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,.05),(0.5,0.5,.05))]),
                "val":transforms.Compose([transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])}
    data_root = os.getcwd()  # get data root path
    image_path = os.path.join(data_root, "AlexNet", "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                        transform=data_transform["train"])
    
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('./AlexNet/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size>1 else 0, 8]) # number of workers
    print('Using {} dataloader workers every process'.format(nw))
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

    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs=1
    save_path = './AlexNet/AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    validate_steps = len(validate_loader)
    for epoch in range(epochs):
        # train
        net.train()
        training_loss=0.0
        validate_loss=0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs=net(images.to(device))
            train_step_loss = loss_function(outputs, labels.to(device))
            train_step_loss.backward()
            optimizer.step()

            # print statistics
            training_loss += train_step_loss.item()
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

        print('[epoch %d] train_loss: %.3f validate_loss: %.3f val_accuracy: %.3f' % 
            (epoch+1, training_loss/train_steps, validate_loss/validate_steps, val_accuracy))
        
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
        
    print('Finished Training')


if __name__ == '__main__':
    main()
