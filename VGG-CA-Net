import torch
import torch.nn as nn


class CA_Block(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super(CA_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channel // reduction)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b,c,h,w
        _, _, h, w = x.size()
        # (b, c, h, w) --> (b, c, h, 1)  --> (b, c, 1, h)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # (b, c, h, w) --> (b, c, 1, w)
        x_w = torch.mean(x, dim=2, keepdim=True)
        # (b, c, 1, w) cat (b, c, 1, h) --->  (b, c, 1, h+w)
        # (b, c, 1, h+w) ---> (b, c/r, 1, h+w)
        x_cat_conv_relu = self.relu(self.bn(self.conv1(torch.cat((x_h, x_w), 3))))
        # (b, c/r, 1, h+w) ---> (b, c/r, 1, h)  、 (b, c/r, 1, w)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        # (b, c/r, 1, h) ---> (b, c, h, 1)
        s_h = self.sigmoid(self.conv2(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # (b, c/r, 1, w) ---> (b, c, 1, w)
        s_w = self.sigmoid(self.conv2(x_cat_conv_split_w))
        # s_h往宽方向进行扩展， s_w往高方向进行扩展
        out = (s_h.expand_as(x) * s_w.expand_as(x)) * x

        return out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
            self.conv = DoubleConv(in_channels, out_channels)
        self.ca_block = CA_Block(in_channels)  
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.ca_block(x1)  
        x = self.conv(x1)
        return x


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VGG_CA_Net(nn.Module):
    def __init__(self, bilinear=True):
        super(VGG_CA_Net, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(1, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(128, 256),
                                  nn.Conv2d(256, 256, kernel_size=3, stride=1,  padding=1))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(256, 512),
                                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(512, 1024),
                                  nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1))
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = Out(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG_CA_Net().to(DEVICE)
    x = torch.randn(1, 1, 512, 512).to(DEVICE)
    print(model(x).shape)
    print(model)
