# coding=utf-8
import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _single


class ConvNd(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias):
        super(ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if out_channels % 2 != 0:
            raise ValueError('out_channels must be divisible by 2 in complex networks')
        self.in_channels = in_channels
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight_A = Parameter(torch.Tensor(
                self.in_channels, self.out_channels // self.groups, *self.kernel_size))
            self.weight_B = Parameter(torch.Tensor(
                self.in_channels, self.out_channels // self.groups, *self.kernel_size))
        else:
            self.weight_A = Parameter(torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size))
            self.weight_B = Parameter(torch.Tensor(
                self.out_channels, self.in_channels // self.groups, *self.kernel_size))
        if bias:
            self.bias_x = Parameter(torch.Tensor(self.out_channels))
            self.bias_y = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_B, a=math.sqrt(5))
        if self.bias_x is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_A)
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_B)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_x, -bound, bound)
            init.uniform_(self.bias_y, -bound, bound)


class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias)

    def forward(self, input):
        Ax = F.conv1d(input, self.weight_A, self.bias_x, self.stride,
                        self.padding, self.dilation, self.groups)
        Bx = F.conv1d(input, self.weight_B, self.bias_x, self.stride,
                        self.padding, self.dilation, self.groups)
        Ay = F.conv1d(input, self.weight_A, self.bias_y, self.stride,
                        self.padding, self.dilation, self.groups)
        By = F.conv1d(input, self.weight_B, self.bias_y, self.stride,
                        self.padding, self.dilation, self.groups)
        return torch.cat([Ax - By, Bx + Ay], dim=1)

