import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.subpixel import SubPixel1D


class TFiLM(nn.Module):
    def __init__(self, n_step, block_size, n_filters):
        super(TFiLM, self).__init__()
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step

        # LSTM qui sort la même dimension que l'entrée
        self.rnn = nn.LSTM(
            input_size=n_filters,
            hidden_size=n_filters,
            batch_first=True,
            bidirectional=False
        )

    def make_normalizer(self, x):
        """
        MaxPool + LSTM pour générer les poids de normalisation
        x: (batch, steps, features)
        """
        x_in_down = F.max_pool1d(x.transpose(1, 2), kernel_size=self.block_size).transpose(1, 2)
        x_rnn, _ = self.rnn(x_in_down)
        return x_rnn

    def apply_normalizer(self, x, x_norm):
        """
        Applique les poids de normalisation bloc par bloc
        """
        batch_size, steps, n_filters = x.shape
        n_blocks = steps // self.block_size

        # reshape en blocs
        x = x.view(batch_size, n_blocks, self.block_size, n_filters)
        x_norm = x_norm.view(batch_size, n_blocks, 1, self.n_filters)

        # multiplication bloc par bloc
        x_out = x * x_norm

        # retour à la forme originale
        x_out = x_out.view(batch_size, steps, n_filters)
        return x_out

    def forward(self, x):
        # x: (batch, steps, features)
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x

class TFiLMNet(nn.Module):
    def __init__(self, n_layers=4, scale=4):
        super(TFiLMNet, self).__init__()

        self.n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        self.n_step = [4096, 2048, 1024, 512, 256, 512, 1024, 2048, 4096]

        self.down_blocks = nn.ModuleList()
        self.tfilm_down = nn.ModuleList()

        # DOWNSAMPLING LAYERS
        for l in range(n_layers):
            conv = nn.Conv1d(
                in_channels=1 if l == 0 else self.n_filters[l-1],
                out_channels=self.n_filters[l],
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            tfilm = TFiLM(self.n_step[l], block_size=int(128 / (2**l)), n_filters=self.n_filters[l])
            self.down_blocks.append(conv)
            self.tfilm_down.append(tfilm)

        # BOTTLENECK
        self.bottleneck_conv = nn.Conv1d(
            in_channels=self.n_filters[n_layers-1],
            out_channels=self.n_filters[-1],
            kernel_size=self.n_filtersizes[-1],
            dilation=2,
            padding="same"
        )
        self.bottleneck_tfilm = TFiLM(
            self.n_step[n_layers], block_size=int(128 / (2**n_layers)), n_filters=self.n_filters[-1]
        )

        # UPSAMPLING LAYERS
        self.up_blocks = nn.ModuleList()
        self.tfilm_up = nn.ModuleList()

        for l in reversed(range(n_layers)):
            conv = nn.Conv1d(
                in_channels=self.n_filters[l]*2,
                out_channels=self.n_filters[l]*2,
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            tfilm = TFiLM(self.n_step[l], block_size=int(128 / (2**l)), n_filters=self.n_filters[l])
            self.up_blocks.append(conv)
            self.tfilm_up.append(tfilm)

        # Output layer
        self.out_conv = nn.Conv1d(2, 2, kernel_size=9, padding=4)

    def forward(self, x):
        # x: (batch, steps, 1)
        skips = []
        out = x

        # Downsampling
        for conv, tfilm in zip(self.down_blocks, self.tfilm_down):
            out = conv(out.transpose(1, 2)).transpose(1, 2)  # Conv1d attend (B,C,L)
            out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
            out = F.leaky_relu(out, 0.2)
            out = tfilm(out)
            skips.append(out)

        # Bottleneck
        out = self.bottleneck_conv(out.transpose(1, 2)).transpose(1, 2)
        out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
        out = F.dropout(out, 0.5, training=self.training)
        out = F.leaky_relu(out, 0.2)
        out = self.bottleneck_tfilm(out)

        # Upsampling
        for conv, tfilm, skip in zip(self.up_blocks, self.tfilm_up, reversed(skips)):
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = F.dropout(out, 0.5, training=self.training)
            out = F.relu(out)
            out = SubPixel1D(out, r=2)
            out = tfilm(out)
            out = torch.cat([out, skip], dim=-1)

        # Output
        out = self.out_conv(out.transpose(1, 2)).transpose(1, 2)
        out = SubPixel1D(out, r=2)

        # Residual connection
        out = out + x
        return out

def get_tfilm(n_layers=4, scale=4):
    return TFiLMNet(n_layers=n_layers, scale=scale)
