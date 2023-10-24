import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from vgg_resnet_model import VGG
from vgg_resnet_model import transformer
import time

#   配置数据集CIFAR10，显示训练数据的大小
dataset_train = CIFAR10("D:\deeplearning\code\CIFAR10-train", train=True, transform=transformer, download=False)
print(f"长度:{len(dataset_train)}")

#   查看数据集的形状
print(dataset_train[0][0].shape)

#   设置tensorboard
writer = SummaryWriter("log of vgg")

#   设置配置为GPU计算
device = torch.device("cuda")

#   创建网络实例
vgg = VGG()
vgg = vgg.to(device)

#   用dataloader处理数据，一次取出64张图片
dataloader_train = DataLoader(dataset_train, batch_size=64, drop_last=True)

#   实例化损失函数，使用GPU配置
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#   确定学习率，设置随机梯度下降优化器
learning_rate = 0.01
optim = torch.optim.SGD(vgg.parameters(), learning_rate)

#   设置训练步骤
train_step = 0
epoch = 30

begin = time.time()
#   开始一轮训练
for i in range(epoch):
    print(f"-------第{i + 1}轮训练开始------")
    total_train_accuracy = 0
    train_accuacy = 0
    for data in dataloader_train:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = vgg(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        accuracy = (outputs.argmax(1) == targets).sum()
        train_accuacy += accuracy
        train_step += 1
        writer.add_scalar("train loss 10", loss.item(), train_step)
        if train_step % 100 == 0:
            print(f"训练次数为：{train_step}")
            writer.add_scalar("train loss 100", loss.item(), train_step)
    total_train_accuracy += train_accuacy / len(dataset_train)
    writer.add_scalar("train accuracy", total_train_accuracy, train_step)
    if i == 19:
        torch.save(vgg.state_dict(), "my vgg param.pth")

end = time.time()
print("训练时间是：", end - begin)
writer.close()
