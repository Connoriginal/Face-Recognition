import torch
import torch.nn as nn

import math

class VGG16(nn.Module):

    def __init__(self, _input_channel, num_class):
        super().__init__()

        # 모델 구현
        self.conv1 = nn.Sequential(
          nn.Conv2d(in_channels=_input_channel, out_channels=64, kernel_size = 3, padding = 1),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        self.conv2 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 1),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.conv3 = nn.Sequential(
          nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
          nn.BatchNorm2d(256),
          nn.ReLU(True),
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1),
          nn.BatchNorm2d(256),
          nn.ReLU(True),
          nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1),
          nn.BatchNorm2d(256),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True) # for MNIST
        )

        self.conv4 = nn.Sequential(
          nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.conv5 = nn.Sequential(
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
          nn.BatchNorm2d(512),
          nn.ReLU(True),
          nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer = nn.Sequential(
          nn.Linear(in_features = 16*512, out_features = 4096),
          nn.BatchNorm1d(4096),
          nn.ReLU(True),
          nn.Linear(in_features = 4096, out_features = 4096),
          nn.BatchNorm1d(4096),
          nn.ReLU(True),
          nn.Linear(in_features = 4096, out_features = num_class),
          # nn.Softmax(dim=-1)
        )



        



    def forward(self, x):
        # forward 구현

        ## convolution
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # 여기가 vgg16 features
        x = self.conv5(x)

        x = x.view(x.size(0),-1)

        x = self.layer(x)
        return x