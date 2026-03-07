import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvBlock(nn.Module):
    """普通卷积块（无残差连接）"""
    def __init__(self, in_channels, out_channels):
        super(SimpleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.4)
        self.extra_dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.extra_dropout(out)
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelOnlyAttentionModule(nn.Module):
    """仅通道注意力模块"""
    def __init__(self, in_channels):
        super(ChannelOnlyAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        ca_out = self.channel_attention(x) * x
        return ca_out


class NoResidualRCAUNet(nn.Module):
    """U-Net变体：无残差连接，但保留通道注意力"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(NoResidualRCAUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention_modules = nn.ModuleList()

        for feature in features:
            self.downs.append(SimpleConvBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(SimpleConvBlock(feature * 2, feature))
            self.attention_modules.append(ChannelOnlyAttentionModule(feature))

        self.bottleneck = SimpleConvBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            skip_connection = self.attention_modules[idx // 2](skip_connection)

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)