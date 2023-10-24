import torch.nn as nn
import torch
from torchvision import transforms


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.block12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
        )
        self.block1_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.block22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
        )
        self.block2_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.block32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
        )

        self.block3_relu = nn.Sequential(
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
        )
        self.block4_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.block52 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
        )
        self.block5_relu = nn.Sequential(
            nn.ReLU(),
        )
        # 可以使用res
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
        )
        self.block6_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifer = nn.Sequential(
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=10),
        )

    def BLOCK(self, input):
        X1 = self.block1(input)
        X2 = self.block12(X1)
        X2 += X1
        X2 = self.block1_relu(X2)
        X2 = self.block2(X2)
        X3 = self.block22(X2)
        X3 += X2
        X3 = self.block2_relu(X3)
        X3 = self.block3(X3)
        X4 = self.block32(X3)
        X4 += X3
        X4 = self.block3_relu(X4)
        X5 = self.block4(X4)
        X5 += X4
        X5 = self.block4_relu(X5)
        X5 = self.block5(X5)
        X6 = self.block52(X5)
        X6 += X5
        X6 = self.block5_relu(X6)
        # torch.Size([64, 512, 4, 4])
        X7 = self.block6(X6)
        # torch.Size([64, 512, 2, 2])
        # X7 += X6
        X7 = self.block6_relu(X7)
        return X7

    def forward(self, input):
        output1 = self.BLOCK(input)
        output1 = torch.reshape(output1, (64, -1))
        output2 = self.classifer(output1)
        return output2


#   设置transformer
transformer = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
