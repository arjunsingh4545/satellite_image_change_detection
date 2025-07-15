import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        p = self.pool(x)
        return x, p  # Return both the feature map and pooled result


class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)  # Account for concatenation
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# class DecoderBlock(torch.nn.Module):
#     def __init__(self, in_channels, skip_channels, out_channels):
#         super(DecoderBlock, self).__init__()
#         self.upconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         # in_channels + skip_channels because we concatenate upsampled output with skip connection
#         self.conv1 = ConvBlock(out_channels + skip_channels, out_channels)
#         self.conv2 = ConvBlock(out_channels, out_channels) # UNCOMMENTED THIS LINE
#
#     def forward(self, x, skip):
#         x = self.upconv(x)
#         # Ensure dimensions match before concatenation (sometimes upconv might have 1 pixel off)
#         diffY = skip.size()[2] - x.size()[2]
#         diffX = skip.size()[3] - x.size()[3]
#         x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
#                                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x, skip], dim=1)
#         x = self.conv1(x)
#         x = self.conv2(x) # UNCOMMENTED THIS LINE
#         return x


class SiameseUNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SiameseUNet, self).__init__()

        # Encoder blocks for image 1
        self.encoder1a = EncoderBlock(in_channels, 64)
        self.encoder2a = EncoderBlock(64, 128)
        self.encoder3a = EncoderBlock(128, 256)
        self.encoder4a = EncoderBlock(256, 512)

        # Encoder blocks for image 2
        self.encoder1b = EncoderBlock(in_channels, 64)
        self.encoder2b = EncoderBlock(64, 128)
        self.encoder3b = EncoderBlock(128, 256)
        self.encoder4b = EncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(1024, 1024)  # 512 + 512 from both encoders

        self.decoder4 = DecoderBlock(1024, 512, 512)
        self.decoder3 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)

        # Final convolution to output the segmentation map
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x1, x2):
        # Encoder part for image 1
        e1a, p1a = self.encoder1a(x1)
        e2a, p2a = self.encoder2a(p1a)
        e3a, p3a = self.encoder3a(p2a)
        e4a, p4a = self.encoder4a(p3a)

        # Encoder part for image 2
        e1b, p1b = self.encoder1b(x2)
        e2b, p2b = self.encoder2b(p1b)
        e3b, p3b = self.encoder3b(p2b)
        e4b, p4b = self.encoder4b(p3b)

        # Concatenate the encoder outputs from both images (for each layer)
        concat_e4 = torch.cat([e4a, e4b], dim=1)  # Concatenate along the channel dimension
        concat_e3 = torch.cat([e3a, e3b], dim=1)
        concat_e2 = torch.cat([e2a, e2b], dim=1)
        concat_e1 = torch.cat([e1a, e1b], dim=1)
        # Bottleneck
        b = self.bottleneck(torch.cat([p4a, p4b], dim=1))

        # Decoder part (shared across both images)
        d4 = self.decoder4(b, concat_e4)
        d3 = self.decoder3(d4, concat_e3)
        d2 = self.decoder2(d3, concat_e2)
        d1 = self.decoder1(d2, concat_e1)

        # Final output
        out = self.final_conv(d1)
        return out

