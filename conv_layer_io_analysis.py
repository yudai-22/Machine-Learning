import torch
import torch.nn as nn


def conv_output_results(input_shape, kernel_size, stride_size, padding_size):
    shape = torch.ones(input_shape)
    shape = shape.unsqueeze(0)

    if len(shape.shape) == 3:
        conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride_size, padding=padding_size)
    if len(shape.shape) == 4:
        conv_layer = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride_size, padding=padding_size)

    conv_layer.weight = nn.Parameter(torch.ones_like(conv_layer.weight))
    conv_layer.bias = nn.Parameter(torch.zeros_like(conv_layer.bias))

    output_data = conv_layer(shape).squeeze()
    output_shape = tuple(output_data.shape)

    print(f"入力データ形状: {input_shape}")
    print(f"出力データ形状: {output_shape}")
