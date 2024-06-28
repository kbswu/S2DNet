"""
    Time: 8/31/23
    Name: encoder_decoder.py
    Author: Wu Nan
"""
import torch.nn as nn
from torch.nn import init
import torch

from models_sfd.shape_app_module import SpectralDifferenceLayer, AppearanceModel


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))
        if out_size == 64:
            self.conv = unetConv2(96, out_size, False)
        else:
            self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class S2CNet_CNN_MI(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        decode_filters = [64, 128, 256, 512, 1024]
        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = SpectralDifferenceLayer(in_channels, 16, 32, 3, 'cnn')
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(decode_filters[4], decode_filters[3], self.is_deconv)
        self.up_concat3 = unetUp(decode_filters[3], decode_filters[2], self.is_deconv)
        self.up_concat2 = unetUp(decode_filters[2], decode_filters[1], self.is_deconv)
        self.up_concat1 = unetUp(decode_filters[1], decode_filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(decode_filters[0], n_classes, 1)
        self.app_model = AppearanceModel(in_channels)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool(conv4)
        res_1, res_2, res_3, res_4 = self.app_model(inputs)
        center = torch.cat([self.center(maxpool4), res_4], dim=1)
        up4 = self.up_concat4(center, conv4, res_3)
        up3 = self.up_concat3(up4, conv3, res_2)
        up2 = self.up_concat2(up3, conv2, res_1)
        up1 = self.up_concat1(up2, conv1)
        final = self.final(up1)
        return final, center


if __name__ == '__main__':
    x = torch.randn(2, 8, 256, 256).cuda()
    model = S2CNet_CNN_MI(in_channels=8, n_classes=10).cuda()
    y = model(x)
    print('Output shape:', y[0].shape, y[1].shape)