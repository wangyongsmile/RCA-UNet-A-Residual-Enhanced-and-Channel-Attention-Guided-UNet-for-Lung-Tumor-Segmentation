import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    """CNN块: 3x3卷积 + 批归一化 + ReLU激活"""

    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualConnectionUnit(nn.Module):
    """简化的残差连接单元(RCU)"""

    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(ResidualConnectionUnit, self).__init__()
        self.blocks = nn.Sequential()

        for i in range(num_blocks):
            block_in_channels = in_channels if i == 0 else out_channels
            self.blocks.add_module(f'block_{i}', CNNBlock(block_in_channels, out_channels))

        # 1x1卷积用于残差输出
        self.res_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, residual_input=None):
        # 如果有残差输入，先进行下采样并拼接
        if residual_input is not None:
            if residual_input.shape[2:] != x.shape[2:]:
                residual_input = F.interpolate(
                    residual_input, size=x.shape[2:], mode='bilinear', align_corners=False
                )
            x = torch.cat([x, residual_input], dim=1)

        # 如果拼接后通道数变化，使用1x1卷积调整通道数
        if x.shape[1] != self.blocks[0].conv.in_channels:
            adjust_conv = nn.Conv2d(x.shape[1], self.blocks[0].conv.in_channels, 1).to(x.device)
            x = adjust_conv(x)

        x = self.blocks(x)
        residual_output = self.res_conv(x)

        return x, residual_output


class IncrementalMRRN(nn.Module):
    """增量多分辨率残差网络"""

    def __init__(self, in_channels=1, out_channels=1, num_streams=3, base_channels=32):
        super(IncrementalMRRN, self).__init__()
        self.num_streams = num_streams
        self.base_channels = base_channels

        # 计算各残差流的通道数
        self.channels_list = [base_channels * (2 ** i) for i in range(num_streams)]

        # 初始卷积层 - 生成第0个残差流的特征
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels_list[0], 3, padding=1),
            nn.BatchNorm2d(self.channels_list[0]),
            nn.ReLU(inplace=True)
        )

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 各残差流的RCU块（第0个流没有RCU块）
        self.rcu_blocks = nn.ModuleList()
        for i in range(1, num_streams):
            # 每个RCU块包含多个RCU单元
            rcu_list = nn.ModuleList()
            for j in range(i + 1):  # 第i个流有i+1个RCU
                # 输入通道数：当前特征 + 来自第j个流的特征
                in_ch = self.channels_list[i] + self.channels_list[j]
                out_ch = self.channels_list[i]
                rcu_list.append(ResidualConnectionUnit(in_ch, out_ch))
            self.rcu_blocks.append(rcu_list)

        # 编码器中的卷积层（用于各残差流之间的转换）
        self.encoder_convs = nn.ModuleList()
        for i in range(num_streams - 1):
            self.encoder_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels_list[i], self.channels_list[i + 1], 3, padding=1),
                    nn.BatchNorm2d(self.channels_list[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )

        # 解码器部分 - 上采样和特征融合
        self.upsample_layers = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()

        for i in range(num_streams - 2, -1, -1):
            self.upsample_layers.append(
                nn.ConvTranspose2d(
                    self.channels_list[i + 1],
                    self.channels_list[i],
                    2, stride=2
                )
            )
            self.decoder_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels_list[i] * 2, self.channels_list[i], 3, padding=1),
                    nn.BatchNorm2d(self.channels_list[i]),
                    nn.ReLU(inplace=True)
                )
            )

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels_list[0], self.channels_list[0] // 2, 3, padding=1),
            nn.BatchNorm2d(self.channels_list[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_list[0] // 2, out_channels, 1)
        )

        # Dropout正则化
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        # 编码器部分
        residual_streams = []

        # 第0个残差流
        stream0 = self.initial_conv(x)
        residual_streams.append(stream0)

        # 构建其他残差流
        current_stream = stream0
        for i in range(1, self.num_streams):
            # 池化并卷积到下一个分辨率
            current_stream = self.pool(current_stream)
            current_stream = self.encoder_convs[i - 1](current_stream)

            # 通过RCU块处理（如果当前流有RCU块）
            if i - 1 < len(self.rcu_blocks):
                rcu_list = self.rcu_blocks[i - 1]

                # 对每个RCU进行处理
                for j, rcu in enumerate(rcu_list):
                    # 确保j不超过residual_streams的长度
                    if j < len(residual_streams):
                        residual_input = residual_streams[j]
                        current_stream, _ = rcu(current_stream, residual_input)
                    else:
                        # 如果没有对应的残差流，只处理当前特征
                        current_stream, _ = rcu(current_stream, None)

                current_stream = self.dropout(current_stream)

            # 将当前流添加到残差流列表
            residual_streams.append(current_stream)

        # 解码器部分
        decoded_stream = residual_streams[-1]

        for i in range(len(self.upsample_layers)):
            # 上采样
            upsampled = self.upsample_layers[i](decoded_stream)

            # 获取对应编码器层的特征
            encoder_feat = residual_streams[-(i + 2)]

            # 调整尺寸（如果需要）
            if upsampled.shape[2:] != encoder_feat.shape[2:]:
                upsampled = F.interpolate(
                    upsampled, size=encoder_feat.shape[2:],
                    mode='bilinear', align_corners=False
                )

            # 拼接特征
            combined = torch.cat([encoder_feat, upsampled], dim=1)

            # 卷积处理
            decoded_stream = self.decoder_convs[i](combined)

        # 最终输出
        output = self.final_conv(decoded_stream)
        return output