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
        
        # padding si nécessaire
        remainder = steps % self.block_size
        if remainder != 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding))  # Pad sur la dimension temporelle
            steps = steps + padding
            n_blocks = steps // self.block_size

        # reshape en blocs
        x = x.view(batch_size, n_blocks, self.block_size, n_filters)
        x_norm = x_norm.view(batch_size, n_blocks, 1, self.n_filters)

        # multiplication bloc par bloc
        x_out = x * x_norm

        # retour à la forme originale
        x_out = x_out.view(batch_size, steps, n_filters)
        
        # enlever le padding si nécessaire
        if remainder != 0:
            x_out = x_out[:, :-padding, :]
            
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
    def __init__(self, n_layers=4, scale=4, input_length=1024):  # Changé par défaut à 1024
        """Initializes the AFiLMNet module."""
        super(AFiLMNet, self).__init__()

        self.n_layers = n_layers
        self.input_length = input_length
        self.n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

        # DOWNSAMPLING LAYERS
        self.down_blocks = nn.ModuleList()
        for l in range(n_layers):
            conv = nn.Conv1d(
                in_channels=1 if l == 0 else self.n_filters[l-1],
                out_channels=self.n_filters[l],
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            self.down_blocks.append(conv)

        # BOTTLENECK
        self.bottleneck_conv = nn.Conv1d(
            in_channels=self.n_filters[n_layers-1],
            out_channels=self.n_filters[n_layers],
            kernel_size=self.n_filtersizes[n_layers],
            dilation=2,
            padding="same"
        )

        # UPSAMPLING LAYERS - CORRECTION ICI
        self.up_blocks = nn.ModuleList()
        
        # Pour chaque couche d'upsampling (dans l'ordre inverse)
        for i in range(n_layers):
            layer_idx = n_layers - 1 - i  # Index dans l'ordre original (3,2,1,0)
            
            if i == 0:
                # Première couche up : vient du bottleneck seulement
                in_channels = self.n_filters[n_layers]  # Sortie du bottleneck
            else:
                # Autres couches : vient de la concaténation précédente
                prev_layer_idx = n_layers - i  # Layer précédent dans l'ordre original
                # Après concat : n_filters[prev_layer_idx] + n_filters[prev_layer_idx]  
                in_channels = self.n_filters[prev_layer_idx] + self.n_filters[prev_layer_idx]
            
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.n_filters[layer_idx] * 2,  # *2 pour SubPixel
                kernel_size=self.n_filtersizes[layer_idx],
                dilation=2,
                padding="same"
            )
            self.up_blocks.append(conv)
            
        # Output layer - prend la dernière concaténation 
        final_in_channels = self.n_filters[0] + self.n_filters[0]  # Dernière concat
        self.out_conv = nn.Conv1d(final_in_channels, 2, kernel_size=9, padding=4)

        # Créer les modules AFiLM
        self._create_afilm_modules()

    def _create_afilm_modules(self):
        """Crée les modules AFiLM avec les bonnes dimensions."""
        current_steps = self.input_length
        
        # Pour downsampling
        self.afilm_down = nn.ModuleList()
        for l in range(self.n_layers):
            current_steps = current_steps // 2  # Max pool divise par 2
            block_size = max(1, min(current_steps, int(128 / (2**l))))
            afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[l])
            self.afilm_down.append(afilm)
        
        # Pour bottleneck
        current_steps = current_steps // 2
        block_size = max(1, min(current_steps, int(128 / (2**self.n_layers))))
        self.bottleneck_afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[self.n_layers])
        
        # Pour upsampling
        self.afilm_up = nn.ModuleList()
        for i in range(self.n_layers):
            current_steps = current_steps * 2  # SubPixel multiplie par 2
            layer_idx = self.n_layers - 1 - i
            block_size = max(1, min(current_steps, int(128 / (2**layer_idx))))
            afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[layer_idx])
            self.afilm_up.append(afilm)

    def forward(self, x):
        """
        Forward pass of the AFiLMNet module.
        x: (batch, steps, 1) ou (batch, steps)
        """
        # Handle 2D input by adding channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [batch, steps] -> [batch, steps, 1]

        # Vérifier la longueur d'entrée
        if x.shape[1] != self.input_length:
            raise ValueError(f"Input length {x.shape[1]} doesn't match expected {self.input_length}")

        skips = []
        out = x

        # Downsampling
        for i, (conv, afilm) in enumerate(zip(self.down_blocks, self.afilm_down)):
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

        # Upsampling - ORDRE TENSORFLOW CORRIGÉ
        for i, (conv, afilm, skip) in enumerate(zip(self.up_blocks, self.afilm_up, reversed(skips))):
            
            # 1 - Convolution (produit 2*nf canaux) 
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = F.dropout(out, 0.5, training=self.training)
            out = F.relu(out)

            # 2 - SubPixel (2*nf -> nf canaux) - UPSAMPLING des dimensions temporelles
            out = SubPixel1D(out, r=2)

            # 3 - AFiLM 
            out = afilm(out)

            # 4 - Concatenation avec le skip (APRÈS SubPixel pour aligner les dimensions)
            out = torch.cat([out, skip], dim=-1)

        # Output
        out = self.out_conv(out.transpose(1, 2)).transpose(1, 2)
        out = SubPixel1D(out, r=2)  # 2 -> 1 canal

        # Residual connection
        if out.shape[1] == x.shape[1] and out.shape[2] == x.shape[2]:
            out = out + x
        else:
            print(f"Warning: Cannot add residual - shapes {out.shape} vs {x.shape}")

        return out.squeeze(-1)

def get_afilm(n_layers=4, scale=4, input_length=1024):
    """Factory function to create an AFiLMNet instance."""
    return AFiLMNet(n_layers=n_layers, scale=scale, input_length=input_length)