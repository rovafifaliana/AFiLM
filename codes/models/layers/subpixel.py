import torch
import torch.nn as nn
import torch.nn.functional as F

def SubPixel1D(I, r):
    """
    SubPixel1D: 1D sub-pixel convolution layer.
    
    Args:
        I (torch.Tensor): Input tensor of shape (batch_size, width, r)
            - batch_size: Number of samples in the batch.
            - width: Number of input channels.
            - r: Upsampling factor.
        r (int): Upsampling factor.
        
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, channels // (r ** 2), length * r).
    """
    batch_size, width, channels = I.shape

    assert channels == r, f"Input channels {channels} must match the upsampling factor {r}."

    X = I.permute(2, 1, 0)  # Change shape to (r, width, batch_size)
    X = X.contiguous().view(1, r * width, batch_size)  # Reshape to (1, r * width, batch_size)
    X = X.permute(2, 1, 0)  # Change shape to (batch_size, r * width, 1)

    return X