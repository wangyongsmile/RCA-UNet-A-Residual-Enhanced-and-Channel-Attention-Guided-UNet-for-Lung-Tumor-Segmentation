import torch
import torch.nn as nn


class RecurrentConv(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RecurrentConv, self).__init__()
        self.t = t
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.dropout(x1)

        x_out = x1
        for _ in range(self.t):
            x_out = self.conv2(x1 + x_out)
            x_out = self.dropout(x_out)

        return self.final_relu(x_out)


class R2UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], t=2):
        super(R2UNet, self).__init__()

        self.enc1 = RecurrentConv(in_channels, features[0], t)
        self.enc2 = RecurrentConv(features[0], features[1], t)
        self.enc3 = RecurrentConv(features[1], features[2], t)
        self.enc4 = RecurrentConv(features[2], features[3], t)

        self.bridge = RecurrentConv(features[3], features[3] * 2, t)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = RecurrentConv(features[3] * 2, features[3], t)

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = RecurrentConv(features[2] * 2, features[2], t)

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = RecurrentConv(features[1] * 2, features[1], t)

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = RecurrentConv(features[0] * 2, features[0], t)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bridge = self.bridge(self.pool(e4))

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

        output = self.final_conv(d1)
        return output