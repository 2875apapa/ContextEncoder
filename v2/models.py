import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            #Layer 1
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            #Layer 2
            nn.Conv2d(64,64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            #Layer 3
            nn.Conv2d(64,128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            #Layer 4
            nn.Conv2d(128,256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
            #Layer 5
            nn.Conv2d(256,512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(),
            #Layer 6
            nn.Conv2d(512, 4000, kernel_size=1),
            #Layer 7
            nn.ConvTranspose2d(4000, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(),
            #Layer 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),
            #Layer 9
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),
            #Layer 10
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(),
            #Layer 11
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            #Layer 1
            nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            #Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            #Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            #Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),

            #Layer 5
            nn.Conv2d(512, 1, kernel_size=3, stride=1,padding=1)
        )


    def forward(self, img):
        return self.model(img)
