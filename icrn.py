import torch
import torch.nn as nn
class InplaceConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), activation=nn.ELU, apply_bn=True):
        
        super(InplaceConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.apply_bn = apply_bn
        if self.apply_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation() if activation is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class FrequencyWiseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FrequencyWiseLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out

class FrequencyWiseLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        
        super(FrequencyWiseLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        B, C, F, T = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B * F, T, C)
        x = self.linear(x)
        x = x.view(B, F, T, -1)
        x = x.permute(0, 3, 1, 2)
        return x

class Model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            InplaceConvBlock(in_channels, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
        ])

        self.frequency_lstm = FrequencyWiseLSTM(input_dim=24, hidden_dim=24 * 2)

        self.frequency_linear = FrequencyWiseLinear(input_dim=24 * 2, output_dim=24)

        self.decoder_blocks = nn.ModuleList([
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, 24),
            InplaceConvBlock(24, out_channels, apply_bn=False, activation=None),
        ])

    def forward(self, x):
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.frequency_lstm(x)

        x = self.frequency_linear(x)

        for block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = x + skip  
            x = block(x)

        return x
