import torch

def SubPixel1D(I, r):
    """
    One-dimensional subpixel upsampling layer compatible with TensorFlow AFiLM
    Args:
        I (torch.Tensor): Input tensor of shape (batch, width, channels)
                         where channels must be divisible by r
        r (int): Upsampling factor
    Returns:
        torch.Tensor: Output tensor of shape (batch, width*r, channels//r)
    """
    batch_size, width, channels = I.shape
    
    # Vérifier que channels est divisible par r
    assert channels % r == 0, f"Channels {channels} must be divisible by r={r}, got remainder {channels % r}"
    
    # Calculer le nombre de canaux de sortie
    out_channels = channels // r
    
    # Reshape: (batch, width, channels) -> (batch, width, out_channels, r)
    X = I.view(batch_size, width, out_channels, r)
    
    # Permute pour préparer l'upsampling: (batch, width, out_channels, r) -> (batch, out_channels, width, r)
    X = X.permute(0, 2, 1, 3)
    
    # Reshape pour upsampling temporel: (batch, out_channels, width, r) -> (batch, out_channels, width*r)
    X = X.contiguous().view(batch_size, out_channels, width * r)
    
    # Permute pour revenir au format (batch, width*r, out_channels)
    X = X.permute(0, 2, 1)
    
    return X