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

        # Vérification de compatibilité
        if steps % self.block_size != 0:
            # Pad ou truncate si nécessaire
            new_steps = n_blocks * self.block_size
            if steps > new_steps:
                x = x[:, :new_steps, :]
            else:
                padding = new_steps - steps
                x = F.pad(x, (0, 0, 0, padding))
            steps = new_steps

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

        self.n_layers = n_layers
        self.n_filters = [128, 256, 512, 512, 512, 512, 512, 512]
        self.n_filtersizes = [65, 33, 17, 9, 9, 9, 9, 9, 9]

        self.down_blocks = nn.ModuleList()

        # DOWNSAMPLING LAYERS
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

        # UPSAMPLING LAYERS - CORRECTION DU CALCUL DES CANAUX
        self.up_blocks = nn.ModuleList()

        for i, l in enumerate(reversed(range(n_layers))):
            if i == 0:
                # Première couche up : bottleneck après SubPixel + skip du dernier down layer
                # SubPixel divise par 2 le nombre de canaux
                upsampled_channels = self.n_filters[n_layers] // 2
                in_channels = upsampled_channels + self.n_filters[l]
            else:
                # Autres couches : output précédent (après SubPixel) + skip correspondant
                # La sortie de la couche précédente était n_filters[l+1], 
                # après SubPixel elle devient n_filters[l+1] // 2
                prev_layer_idx = n_layers - i  # l+1 dans l'ordre inverse
                upsampled_channels = self.n_filters[prev_layer_idx] // 2
                in_channels = upsampled_channels + self.n_filters[l]
            
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.n_filters[l],  # Sortie normale, pas *2 car SubPixel vient après
                kernel_size=self.n_filtersizes[l],
                dilation=2,
                padding="same"
            )
            self.up_blocks.append(conv)
            # print(f"Up block {i}: in_channels={in_channels}, out_channels={self.n_filters[l]}")

        # Output layer - elle doit produire 4 canaux pour SubPixel(r=2) -> 2 canaux finaux
        # self.out_conv = nn.Conv1d(self.n_filters[0], 4, kernel_size=9, padding=4)
        self.out_conv = nn.Conv1d(self.n_filters[0], 2, kernel_size=9, padding=4)

    def create_afilm_modules(self, input_steps):
        """Crée les modules AFiLM avec les bonnes dimensions."""
        current_steps = input_steps
        
        # Pour downsampling
        afilm_down = []
        for l in range(self.n_layers):
            current_steps = current_steps // 2  # Max pool divise par 2
            block_size = max(1, min(current_steps, int(128 / (2**l))))
            afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[l])
            afilm_down.append(afilm)
        
        # Pour bottleneck
        current_steps = current_steps // 2
        block_size = max(1, min(current_steps, int(128 / (2**self.n_layers))))
        bottleneck_afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[self.n_layers])
        
        # Pour upsampling
        afilm_up = []
        for l in reversed(range(self.n_layers)):
            current_steps = current_steps * 2  # SubPixel multiplie par 2
            block_size = max(1, min(current_steps, int(128 / (2**l))))
            afilm = AFiLM(current_steps, block_size=block_size, n_filters=self.n_filters[l])
            afilm_up.append(afilm)
            
        return afilm_down, bottleneck_afilm, afilm_up

    def forward(self, x):
        """
        Forward pass of the AFiLMNet module.
        x: (batch, steps, 1) ou (batch, steps)
        """
        # Handle 2D input by adding channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [batch, steps] -> [batch, steps, 1]

        batch_size, input_steps, _ = x.shape
        
        # Créer les modules AFiLM avec les bonnes dimensions
        afilm_down, bottleneck_afilm, afilm_up = self.create_afilm_modules(input_steps)
        
        # Transférer sur le bon device
        device = x.device
        for afilm in afilm_down:
            afilm.to(device)
        bottleneck_afilm.to(device)
        for afilm in afilm_up:
            afilm.to(device)

        skips = []
        out = x

        # Downsampling
        for i, (conv, afilm) in enumerate(zip(self.down_blocks, afilm_down)):
            out = conv(out.transpose(1, 2)).transpose(1, 2)  # Conv1d attend (B,C,L)
            out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
            out = F.leaky_relu(out, 0.2)
            out = afilm(out)
            skips.append(out)
            # print(f"Down layer {i}: {out.shape}")

        # Bottleneck
        out = self.bottleneck_conv(out.transpose(1, 2)).transpose(1, 2)
        out = F.max_pool1d(out.transpose(1, 2), 2).transpose(1, 2)
        out = F.dropout(out, 0.5, training=self.training)
        out = F.leaky_relu(out, 0.2)
        out = bottleneck_afilm(out)
        # print(f"Bottleneck: {out.shape}")

        # Upsampling - ORDRE CORRIGÉ
        for i, (conv, afilm, skip) in enumerate(zip(self.up_blocks, afilm_up, reversed(skips))):
            # 1. Upsampling FIRST
            out = SubPixel1D(out, r=2)
            # print(f"After upsampling {i}: {out.shape}")
            
            # 2. Concatenation avec skip
            out = torch.cat([out, skip], dim=-1)
            # print(f"After concat {i}: {out.shape}")
            
            # 3. Convolution
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            out = F.dropout(out, 0.5, training=self.training)
            out = F.relu(out)
            # print(f"After conv {i}: {out.shape}")
            
            # 4. AFiLM
            out = afilm(out)
            # print(f"After AFiLM {i}: {out.shape}")

        # Output - maintenant produit 4 canaux
        out = self.out_conv(out.transpose(1, 2)).transpose(1, 2)
        out = SubPixel1D(out, r=2)  # 4 canaux -> 2 canaux

        # Residual connection - vérifier compatibilité
        # if out.shape[1] == x.shape[1] and out.shape[2] >= x.shape[2]:
        if out.shape[1] == x.shape[1] and out.shape[2] == x.shape[2]:

            # Prendre seulement les premiers canaux si nécessaire
            # residual = x if x.shape[2] == out.shape[2] else x.expand(-1, -1, out.shape[2])
            # out = out + residual
            out = out + x
        else:
            print(f"Warning: Cannot add residual - shapes {out.shape} vs {x.shape}")

        return out.squeeze(-1)

def get_afilm(n_layers=4, scale=4):
    """Factory function to create an AFiLMNet instance."""
    return AFiLMNet(n_layers=n_layers, scale=scale)