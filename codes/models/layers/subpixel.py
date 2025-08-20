import torch
import torch.nn as nn
import torch.nn.functional as F

def SubPixel1D(I, r):
    """
    SubPixel1D: 1D sub-pixel convolution layer.
    Args:
        I (torch.Tensor): Input tensor of shape (batch_size, steps, channels)
                         where channels must be divisible by r
        r (int): Upsampling factor.
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, steps * r, channels // r).
    """
    batch_size, steps, channels = I.shape
    
    # Vérification que channels est divisible par r
    assert channels % r == 0, f"Input channels {channels} must be divisible by upsampling factor {r}."
    
    output_channels = channels // r
    
    # Reshape pour séparer les canaux en groupes de r
    # (batch, steps, channels) -> (batch, steps, output_channels, r)
    I_reshaped = I.view(batch_size, steps, output_channels, r)
    
    # Permute pour mettre r en avant
    # (batch, steps, output_channels, r) -> (batch, steps, r, output_channels)
    I_permuted = I_reshaped.permute(0, 1, 3, 2)
    
    # Reshape pour intercaler les éléments
    # (batch, steps, r, output_channels) -> (batch, steps * r, output_channels)
    output = I_permuted.contiguous().view(batch_size, steps * r, output_channels)
    
    return output