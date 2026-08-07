import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SCRE_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCRE_Unit, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out, inplace=False)


class SCRD_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SCRD_Unit, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return F.relu(out, inplace=False)


class ASC_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASC_Unit, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, dilation=1)
        self.conv2 = DepthwiseSeparableConv(in_channels, out_channels, dilation=2)
        self.conv4 = DepthwiseSeparableConv(in_channels, out_channels, dilation=4)
        self.conv8 = DepthwiseSeparableConv(in_channels, out_channels, dilation=8)
        self.fusion_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out4 = self.conv4(x)
        out8 = self.conv8(x)
        out = torch.cat([out1, out2, out4, out8], dim=1)
        out = self.fusion_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class DRS_CNN2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512]):
        super(DRS_CNN2, self).__init__()

        self.encoder1 = SCRE_Unit(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = SCRE_Unit(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = SCRE_Unit(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = SCRE_Unit(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ASC_Unit(features[3], features[4])
        self.upconv4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = SCRD_Unit(features[3] * 2, features[3])
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = SCRD_Unit(features[2] * 2, features[2])
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = SCRD_Unit(features[1] * 2, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = SCRD_Unit(features[0] * 2, features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck_input = self.pool4(enc4)
        bottleneck = self.bottleneck(bottleneck_input)

        dec4 = self.upconv4(bottleneck)
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))