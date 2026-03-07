import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualConnectionUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(ResidualConnectionUnit, self).__init__()
        self.blocks = nn.Sequential()
        self.first_block_in_channels = in_channels

        for i in range(num_blocks):
            block_in_channels = self.first_block_in_channels if i == 0 else out_channels
            self.blocks.add_module(f'block_{i}', CNNBlock(block_in_channels, out_channels))

        self.res_conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, residual_input=None):
        if residual_input is not None:
            if residual_input.shape[2:] != x.shape[2:]:
                residual_input = F.interpolate(
                    residual_input, size=x.shape[2:], mode='bilinear', align_corners=False
                )
            x = torch.cat([x, residual_input], dim=1)

        if x.shape[1] != self.first_block_in_channels:
            adjust_conv = nn.Conv2d(x.shape[1], self.first_block_in_channels, 1).to(x.device)
            x = adjust_conv(x)

        x = self.blocks(x)
        residual_output = self.res_conv(x)

        return x, residual_output


class IncrementalMRRN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_streams=3, base_channels=32):
        super(IncrementalMRRN, self).__init__()
        self.num_streams = num_streams
        self.base_channels = base_channels

        self.channels_list = [base_channels * (2 ** i) for i in range(num_streams)]
        print(f"通道数列表: {self.channels_list}")

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.channels_list[0], 3, padding=1),
            nn.BatchNorm2d(self.channels_list[0]),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.rcu_blocks = nn.ModuleList()
        for i in range(1, num_streams):
            rcu_list = nn.ModuleList()
            for j in range(i + 1):
                in_ch = self.channels_list[i] + self.channels_list[j]
                out_ch = self.channels_list[i]
                rcu_list.append(ResidualConnectionUnit(in_ch, out_ch))
            self.rcu_blocks.append(rcu_list)

        self.encoder_convs = nn.ModuleList()
        for i in range(num_streams - 1):
            self.encoder_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels_list[i], self.channels_list[i + 1], 3, padding=1),
                    nn.BatchNorm2d(self.channels_list[i + 1]),
                    nn.ReLU(inplace=True)
                )
            )

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

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels_list[0], self.channels_list[0] // 2, 3, padding=1),
            nn.BatchNorm2d(self.channels_list[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_list[0] // 2, out_channels, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        residual_streams = []
        stream0 = self.initial_conv(x)
        residual_streams.append(stream0)

        current_stream = stream0
        for i in range(1, self.num_streams):
            current_stream = self.pool(current_stream)
            current_stream = self.encoder_convs[i - 1](current_stream)

            if i - 1 < len(self.rcu_blocks):
                rcu_list = self.rcu_blocks[i - 1]
                for j, rcu in enumerate(rcu_list):
                    if j < len(residual_streams):
                        residual_input = residual_streams[j]
                        current_stream, _ = rcu(current_stream, residual_input)
                    else:
                        current_stream, _ = rcu(current_stream, None)

                current_stream = self.dropout(current_stream)

            residual_streams.append(current_stream)

        decoded_stream = residual_streams[-1]

        for i in range(len(self.upsample_layers)):
            upsampled = self.upsample_layers[i](decoded_stream)
            encoder_feat = residual_streams[-(i + 2)]

            if upsampled.shape[2:] != encoder_feat.shape[2:]:
                upsampled = F.interpolate(
                    upsampled, size=encoder_feat.shape[2:],
                    mode='bilinear', align_corners=False
                )

            combined = torch.cat([encoder_feat, upsampled], dim=1)
            decoded_stream = self.decoder_convs[i](combined)

        output = self.final_conv(decoded_stream)
        return output