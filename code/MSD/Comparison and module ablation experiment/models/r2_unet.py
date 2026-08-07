import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentConv(nn.Module):
    """循环卷积块 - R2U-Net的核心组件"""

    def __init__(self, in_channels, out_channels, t=2):
        super(RecurrentConv, self).__init__()
        self.t = t  # 循环次数
        self.out_channels = out_channels

        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 循环卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 最终激活函数
        self.final_relu = nn.ReLU(inplace=True)

        # Dropout
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        # 第一次前向传播
        x1 = self.conv1(x)
        x1 = self.dropout(x1)

        # 循环卷积
        x_out = x1
        for _ in range(self.t):
            x_out = self.conv2(x1 + x_out)  # 残差连接
            x_out = self.dropout(x_out)

        return self.final_relu(x_out)


class R2UNet(nn.Module):
    """R2U-Net模型"""

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], t=2):
        super(R2UNet, self).__init__()

        # 编码器路径
        self.enc1 = RecurrentConv(in_channels, features[0], t)
        self.enc2 = RecurrentConv(features[0], features[1], t)
        self.enc3 = RecurrentConv(features[1], features[2], t)
        self.enc4 = RecurrentConv(features[2], features[3], t)

        # 桥接层
        self.bridge = RecurrentConv(features[3], features[3] * 2, t)

        # 解码器路径
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = RecurrentConv(features[3] * 2, features[3], t)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = RecurrentConv(features[2] * 2, features[2], t)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = RecurrentConv(features[1] * 2, features[1], t)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = RecurrentConv(features[0] * 2, features[0], t)

        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 桥接层
        bridge = self.bridge(self.pool(e4))

        # 解码器路径（使用跳跃连接）
        d4 = self.up4(bridge)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # 最终输出
        output = self.final_conv(d1)
        return output