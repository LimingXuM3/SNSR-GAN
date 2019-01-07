import math
from spectral_normalization import SpectralNorm
import torch.nn.functional as F
from torch import nn
leak = 0.2
import torch

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(14, 32, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x, label):
        block1_1 = self.block1_1(x)
        block1_2 = self.block1_2(label)
        block1 = torch.cat([block1_1, block1_2], 1)
        block1 = self.block1(block1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (F.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1_1 = SpectralNorm(nn.Conv2d(3, 32, 3, stride=1, padding=(1,1)))
        self.conv1_2 = SpectralNorm(nn.Conv2d(14, 32, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 3, stride=1, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 3, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 3, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))
        self.conv8 = SpectralNorm(nn.Conv2d(512, 512, 3, stride=2, padding=(1,1)))
        self.fc1 = SpectralNorm(nn.Conv2d(512, 1024, 1))
        self.fc2 = SpectralNorm(nn.Conv2d(1024, 1, 1))
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        self.bn7 = nn.BatchNorm2d(512)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, label):
        batch_size = x.size(0)
        m = x
        m = nn.LeakyReLU(leak)(self.conv1_1(m))
        y = nn.LeakyReLU(leak)(self.conv1_2(label))
        m = torch.cat([m, y], 1)
        m = nn.LeakyReLU(leak)(self.bn1(self.conv2(m)))
        m = nn.LeakyReLU(leak)(self.bn2(self.conv3(m)))
        m = nn.LeakyReLU(leak)(self.bn3(self.conv4(m)))
        m = nn.LeakyReLU(leak)(self.bn4(self.conv5(m)))
        m = nn.LeakyReLU(leak)(self.bn5(self.conv6(m)))
        m = nn.LeakyReLU(leak)(self.bn6(self.conv7(m)))
        m = nn.LeakyReLU(leak)(self.bn7(self.conv8(m)))
        m = self.pool(m)
        m = nn.LeakyReLU(leak)(self.fc1(m))
        m = self.fc2(m)
        
        return F.sigmoid(m.view(batch_size))
        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
