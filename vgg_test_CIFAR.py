from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from my_vgg_model import transformer
from my_vgg_model import VGG

import torch
import torch.nn as nn

device = torch.device("cuda")

vgg_ok = VGG()
param = torch.load("my vgg param.pth")
vgg_ok.load_state_dict(param)
vgg_ok.to(device)

dataset_test = CIFAR10("dataset of test", train=False, transform=transformer, download=False)
dataloader_test = DataLoader(dataset_test, batch_size=64, drop_last=True)

loss_fn = nn.CrossEntropyLoss()
test_step = 0
writer = SummaryWriter("log of vgg test")
i = 1
for test in dataloader_test:
    print(f"------第{i}次测试------")
    i += 1
    imgs, targets = test
    imgs = imgs.to(device)
    targets = targets.to(device)
    output_test = vgg_ok(imgs)
    loss = loss_fn(output_test, targets)

    test_step += 1
    writer.add_scalar("test loss", loss.item(), test_step)
    accuracy = (output_test.argmax(1) == targets).sum()
    batch_accuracy = accuracy / 64
    writer.add_scalar("test accuracy", batch_accuracy, test_step)

writer.close()