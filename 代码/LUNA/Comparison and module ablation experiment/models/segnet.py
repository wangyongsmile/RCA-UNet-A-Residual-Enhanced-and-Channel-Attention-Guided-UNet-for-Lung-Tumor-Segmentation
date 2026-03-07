import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_channels=64):
        super(SegNet, self).__init__()

        self.enc1 = self._encoder_block(in_channels, init_channels)
        self.enc2 = self._encoder_block(init_channels, init_channels * 2)
        self.enc3 = self._encoder_block(init_channels * 2, init_channels * 4)
        self.enc4 = self._encoder_block(init_channels * 4, init_channels * 8)
        self.enc5 = self._encoder_block(init_channels * 8, init_channels * 8)

        self.dec5 = self._decoder_block(init_channels * 8, init_channels * 8)
        self.dec4 = self._decoder_block(init_channels * 8, init_channels * 4)
        self.dec3 = self._decoder_block(init_channels * 4, init_channels * 2)
        self.dec2 = self._decoder_block(init_channels * 2, init_channels)
        self.dec1 = self._decoder_block(init_channels, init_channels)

        self.final_conv = nn.Conv2d(init_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.dropout = nn.Dropout2d(0.3)

    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x1_size = x1.size()
        x1, idx1 = self.pool(x1)
        x1 = self.dropout(x1)

        x2 = self.enc2(x1)
        x2_size = x2.size()
        x2, idx2 = self.pool(x2)
        x2 = self.dropout(x2)

        x3 = self.enc3(x2)
        x3_size = x3.size()
        x3, idx3 = self.pool(x3)
        x3 = self.dropout(x3)

        x4 = self.enc4(x3)
        x4_size = x4.size()
        x4, idx4 = self.pool(x4)
        x4 = self.dropout(x4)

        x5 = self.enc5(x4)
        x5_size = x5.size()
        x5, idx5 = self.pool(x5)
        x5 = self.dropout(x5)

        x = self.unpool(x5, idx5, output_size=x5_size)
        x = self.dec5(x)
        x = self.dropout(x)

        x = self.unpool(x, idx4, output_size=x4_size)
        x = self.dec4(x)
        x = self.dropout(x)

        x = self.unpool(x, idx3, output_size=x3_size)
        x = self.dec3(x)
        x = self.dropout(x)

        x = self.unpool(x, idx2, output_size=x2_size)
        x = self.dec2(x)
        x = self.dropout(x)

        x = self.unpool(x, idx1, output_size=x1_size)
        x = self.dec1(x)

        output = self.final_conv(x)
        return torch.sigmoid(output)