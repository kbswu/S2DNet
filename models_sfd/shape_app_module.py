"""
    Time: 8/31/23
    Name: shape_app_module.py
    Author: Wu Nan
    Adapted from: Su et. al, Lightweight Pixel Difference Networks for Efficient Visual Representation Learning / Pixel Difference Networks for Efficient Edge Detection, 
    TPAMI23, ICCV22, https://github.com/hellozhuo/pidinet
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDifferenceLayer(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size,
                 mode, padding=1, dilation=1, bias=False):
        """
        make the spectral difference between the cross-bands knowledge
        :param in_channels: shape of band knowledge
        :param mid_channels: shape of group-wise projectors' output
        :param out_channels: shape of integrated difference knowledge cross-bands
        :param kernel_size: -
        :param stride: -
        :param padding: -
        :param dilation: -
        :param bias: -
        """
        super().__init__()

        # init conv
        assert mid_channels % in_channels == 0, "mid_channels must be divisible by in_channels"
        # group-wise projector, to project the single-band into self sub-spaces
        stride = 1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=bias)
        # compute the number of difference features
        self.group_size = mid_channels // in_channels
        self.total_interactions = mid_channels * (mid_channels - 1) // 2
        self.num_inter_action = in_channels * (self.group_size * (self.group_size - 1)) // 2
        self.num_intra_diff = self.total_interactions - self.num_inter_action
        # the attention weights for the difference features to mimic the conditional thresholding operation
        self.attention = nn.Parameter(torch.Tensor(self.num_intra_diff), requires_grad=True)
        nn.init.ones_(self.attention)
        # restore the difference features into the original shape

        self.conv2 = nn.Conv2d(self.num_intra_diff, out_channels, kernel_size, stride, padding, dilation,
                                        bias=bias)
        self.activation = nn.ReLU(inplace=True)
        self.bn_diff = nn.BatchNorm2d(self.num_intra_diff)

    def make_difference_feat(self, x):
        diff_list = []
        for group_start1 in range(0, x.shape[1], self.group_size):
            group_end1 = group_start1 + self.group_size
            for group_start2 in range(group_end1, x.shape[1], self.group_size):
                group_end2 = group_start2 + self.group_size
                for i in range(group_start1, group_end1):
                    for j in range(group_start2, group_end2):
                        diff_list.append(x[:, i, :, :] - x[:, j, :, :])
        return torch.stack(diff_list, dim=1)

    def forward(self, x):
        x = self.conv1(x)
        channel_diffs = self.make_difference_feat(x)
        channel_diffs_res = channel_diffs
        channel_diffs *= self.attention.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        channel_diffs = self.activation(self.bn_diff(channel_diffs_res + channel_diffs))
        x = self.activation(self.conv2(channel_diffs))
        return x


class AppearanceOperator(nn.Module):
    def __init__(self, op_type,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.op_type = op_type
        self.diff_op = self.build_diff_op(op_type)

        # the conv settings
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def build_diff_op(self, op_type):
        assert self.op_type in ['cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
        if self.op_type == 'cd':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
                assert padding == dilation, 'padding for cd_conv set wrong'

                weights_c = weights.sum(dim=[2, 3], keepdim=True)
                yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
                y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y - yc

            return func
        elif self.op_type == 'ad':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
                assert padding == dilation, 'padding for ad_conv set wrong'

                shape = weights.shape
                weights = weights.view(shape[0], shape[1], -1)
                weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape)  # clock-wise
                y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y

            return func
        elif self.op_type == 'rd':
            def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
                assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
                assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
                padding = 2 * dilation

                shape = weights.shape
                if weights.is_cuda:
                    buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
                else:
                    buffer = torch.zeros(shape[0], shape[1], 5 * 5)
                weights = weights.view(shape[0], shape[1], -1)
                buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
                buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
                buffer[:, :, 12] = 0
                buffer = buffer.view(shape[0], shape[1], 5, 5)
                y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
                return y

            return func
        else:
            print('impossible to be here unless you force that')
            return None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.diff_op(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class AppearanceBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__()
        self.conv_cd = AppearanceOperator('cd', in_c, out_c, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_ad = AppearanceOperator('ad', in_c, out_c, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_rd = AppearanceOperator('rd', in_c, out_c, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_fusion = nn.Conv2d(3 * out_c, out_c, 3, 1, 1, 1, 1, bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x_cd = self.conv_cd(x)
        x_ad = self.conv_ad(x)
        x_rd = self.conv_rd(x)
        x = self.conv_fusion(self.activation(torch.cat([self.bn(x_cd), self.bn(x_ad), self.bn(x_rd)], dim=1)))
        return x


class GenericProjection(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding, dilation, groups, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_c)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class AppearanceModel(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        in_channels = [32, 64, 128, 256]
        out_channels = [64, 128, 256, 512]
        stride = 1
        self.stem = GenericProjection(num_channels, in_channels[0], 3, stride, 1, 1, 1, True)
        self.s_app_blocks = nn.ModuleList([
            AppearanceBlock(in_channels[i], out_channels[i], 3, 2, 1, 1, 1, True) for i in range(4)
        ])

    def forward(self, x):
        s_app = self.stem(x)
        s_feats = []
        for block in self.s_app_blocks:
            s_app = block(s_app)
            s_feats.append(s_app)
        return s_feats


if __name__ == "__main__":
    x = torch.randn(10, 8, 224, 224).cuda()
    model = AppearanceModel(8).cuda()
    s_app = model(x)
    print(s_app[0].shape)
