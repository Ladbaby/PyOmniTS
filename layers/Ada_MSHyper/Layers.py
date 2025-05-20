import torch.nn as nn
from torch.nn.modules.linear import Linear

class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)####input_x[32,128,42]
        x = self.norm(x)
        x = self.activation(x)
        return x

class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""
    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size),
                ConvLayer(d_inner, window_size)
                ])
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)####d_inner128 d_model=512
        self.down = Linear(d_model, d_inner)####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):####[32,169,512]

        temp_input = self.down(enc_input).permute(0, 2, 1)####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input.permute(0, 2, 1))
        return all_inputs
