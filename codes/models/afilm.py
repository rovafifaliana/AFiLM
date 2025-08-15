import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.layers.transformer import TransformerBlock
from models.layers.subpixel import SubPixel1D


class AFiLM(nn.Module):
    """
    AFiLM: Adaptive Feature-wise Linear Modulation Network.
    This module applies adaptive feature-wise linear modulation to the input
    using a Transformer-based approach.
    """
    def __init__(self, n_step, block_size, n_filters):
        """
        Initializes the AFiLM module.
        Args:
            n_step (int): Number of steps in the input sequence.
            block_size (int): Size of the blocks for modulation.
            n_filters (int): Number of filters in the input sequence.
        """
        super(AFiLM, self).__init__()
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step

        max_len = int(n_step / block_size) # Maximum sequence length for position encoding
        self.transformer = TransformerBlock( 
            num_layers=4,
            embed_dim=n_filters,
            maximum_position_encoding=max_len,
            num_heads=8,
            ff_dim=2048
        ) # Transformer for feature modulation

    def make_normalizer(self, x):
        """
        MaxPool + Transformer to generate normalization weights.
        x: (batch, steps, features)
        """
        x_in_down = F.max_pool1d(x.transpose(1, 2), kernel_size=self.block_size).transpose(1, 2)
        x_transformer = self.transformer(x_in_down)  # (batch, steps/block, features)
        return x_transformer

    def apply_normalizer(self, x, x_norm):
        """
        Applies normalization weights block by block.
        x: (batch, steps, features)
        x_norm: (batch, steps/block, features)
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
        """        
        Forward pass of the AFiLM module.
        x: (batch, steps, features)
        """
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x
    

class AFiLMNet(nn.Module):
    """AFiLMNet: Adaptive Feature-wise Linear Modulation Network."""
    def __init__(self, n_layers=4, scale=4):
        """Initializes the AFiLMNet module."""
        super(AFiLMNet, self).__init__()

        self.n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]
        self.n_step = [4096, 2048, 1024, 512, 256, 512, 1024, 2048, 4096]

        self.down_blocks = nn.ModuleList()
        self.afilm_down = nn.ModuleList()

        # DOWNSAMPLING LAYERS
        for l in range(n_layers):
            conv = nn.Conv1d(
                in_channels=1 if l == 0 else self.n_filters[l-1],
                out_channels=self.n_filters[l],
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            afilm = AFiLM(self.n_step[l], block_size=int(128 / (2**l)), n_filters=self.n_filters[l])
            self.down_blocks.append(conv)
            self.afilm_down.append(afilm)

        # BOTTLENECK
        self.bottleneck_conv = nn.Conv1d(
            in_channels=self.n_filters[n_layers-1],
            out_channels=self.n_filters[-1],
            kernel_size=self.n_filtersizes[-1],
            dilation=2,
            padding="same"
        )
        self.bottleneck_afilm = AFiLM(
            self.n_step[n_layers], block_size=int(128 / (2**n_layers)), n_filters=self.n_filters[-1]
        )

        # UPSAMPLING LAYERS
        self.up_blocks = nn.ModuleList()
        self.afilm_up = nn.ModuleList()

        for l in reversed(range(n_layers)):
            conv = nn.Conv1d(
                in_channels=self.n_filters[l]*2,
                out_channels=self.n_filters[l]*2,  # car subpixel upsampling
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            afilm = AFiLM(self.n_step[l], block_size=int(128 / (2**l)), n_filters=self.n_filters[l])
            self.up_blocks.append(conv)
            self.afilm_up.append(afilm)

        # Output layer
        self.out_conv = nn.Conv1d(2, 2, kernel_size=9, padding=4)

    def forward(self, x):
        """
        Forward pass of the AFiLMNet module.
        x: (batch, steps, 1)
        """
        skips = []
        out = x

        # Downsampling
        for conv, afilm in zip(self.down_blocks, self.afilm_down):
            out = conv(out.transpose(1, 2)).transpose(1, 2)  # Conv1d attend (B,C,L)
            out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
            out = F.leaky_relu(out, 0.2)
            out = afilm(out)
            skips.append(out)

        # Bottleneck
        out = self.bottleneck_conv(out.transpose(1, 2)).transpose(1, 2)
        out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
        out = F.dropout(out, 0.5, training=self.training)
        out = F.leaky_relu(out, 0.2)
        out = self.bottleneck_afilm(out)

        # Upsampling
        for conv, afilm, skip in zip(self.up_blocks, self.afilm_up, reversed(skips)):
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = F.dropout(out, 0.5, training=self.training)
            out = F.relu(out)
            out = SubPixel1D(out, r=2)
            out = afilm(out)
            out = torch.cat([out, skip], dim=-1)

        # Output
        out = self.out_conv(out.transpose(1, 2)).transpose(1, 2)
        out = SubPixel1D(out, r=2)

        # Residual connection
        out = out + x
        return out

def get_afilm(n_layers=4, scale=4):
    """Factory function to create an AFiLMNet instance."""
    return AFiLMNet(n_layers=n_layers, scale=scale)
