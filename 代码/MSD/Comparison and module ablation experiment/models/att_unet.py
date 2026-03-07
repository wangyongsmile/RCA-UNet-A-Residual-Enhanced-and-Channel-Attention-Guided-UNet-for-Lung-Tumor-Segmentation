import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """标准卷积块（无残差，无dropout）"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    """注意力门模块"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """标准Attention U-Net模型（无残差连接，使用普通卷积块）"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # 解码器组件
        self.attention_gates = nn.ModuleList()
        self.up_convs = nn.ModuleList()      # 上采样（转置卷积）
        self.decoder_blocks = nn.ModuleList() # 拼接后的ConvBlock

        for feature in reversed(features):
            # 上采样层
            self.up_convs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # 注意力门
            self.attention_gates.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))
            # 拼接后的卷积块
            self.decoder_blocks.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转

        # 解码器
        for idx in range(len(self.up_convs)):
            x = self.up_convs[idx](x)
            skip_connection = skip_connections[idx]

            # 应用注意力门
            attn_weights = self.attention_gates[idx](x, skip_connection)

            # 尺寸调整（确保一致）
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            # 拼接并卷积
            concat_skip = torch.cat((attn_weights, x), dim=1)
            x = self.decoder_blocks[idx](concat_skip)

        return self.final_conv(x)