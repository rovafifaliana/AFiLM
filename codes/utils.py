import os
import numpy as np
import h5py
import librosa
import soundfile as sf
import torch
from scipy import interpolate
from matplotlib import pyplot as plt

def load_h5(h5_path):
    """ Load data from an HDF5 file."""
    with h5py.File(h5_path, 'r') as hf:
        print('List of arrays in input file:', list(hf.keys()))
        X = np.array(hf.get('data'))
        Y = np.array(hf.get('label'))
    return X, Y

def spline_up(x_lr, r):
    """
    Prendre un signal en basse résolution (quelques points espacés) 
    et elle créer un signal en haute résolution en remplissant les points manquants 
    grâce à une interpolation par spline cubique
    """
    x_lr = x_lr.flatten() # Assurez-vous que x_lr est un tableau 1D
    x_hr_len = len(x_lr) * r # r est le factor de sur-échantillonnage, le high-res sera r fois plus long que le low-res
    x_sp = np.zeros(x_hr_len) 
    i_lr = np.arange(x_hr_len, step=r) # Indices du low-res
    i_hr = np.arange(x_hr_len) # Indices du high-res, indices à interpoler
    f = interpolate.splrep(i_lr, x_lr)
    x_sp = interpolate.splev(i_hr, f) # Interpolation spline
    return x_sp

def get_device():
    """Determine the best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def upsample_wav(wav, args, model, save_spectrum=False, device=None):
    # Determine device if not specified
    if device is None:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # load signal
    x_hr, fs = librosa.load(wav, sr=args.sr) # Load the audio file at the specified sample rate
    x_lr_t = np.array(x_hr[0::args.r]) # Downsample the high-res signal to low-res
    
    # pad to mutliple of patch size to ensure model runs over entire sample
    x_hr = np.pad(
        x_hr, 
        (0, args.patch_size - (x_hr.shape[0] % args.patch_size)), 
        'constant', 
        constant_values=(0,0)
    )
    
    # downscale signal same as in data preparation
    x_lr = np.array(x_hr[0::args.r])
    
    # upscale the low-res version
    x_lr = x_lr.reshape((1, len(x_lr), 1)) # Reshape for model input (batch size, sequence length, channels)
    
    # preprocessing: “baseline” par spline + mise en forme en patches
    assert len(x_lr) == 1
    
    x_sp = spline_up(x_lr, args.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(args.layers+1)))]
    x_sp = x_sp.reshape((1, len(x_sp), 1))
    x_sp = x_sp.reshape((int(x_sp.shape[1]/args.patch_size), args.patch_size, 1))
    
    # Convert to PyTorch tensor and move to device
    x_sp_tensor = torch.from_numpy(x_sp).float().to(device)
    
    # Ensure model is in evaluation mode and on correct device
    model.eval()
    model = model.to(device)
    
    # prediction with PyTorch (no gradients needed for inference)
    with torch.no_grad():
        # Process in batches to manage memory
        batch_size = 16
        predictions = []
        
        for i in range(0, x_sp_tensor.shape[0], batch_size):
            batch = x_sp_tensor[i:i+batch_size]
            pred_batch = model(batch)
            predictions.append(pred_batch.cpu())
        
        # Concatenate all predictions
        pred = torch.cat(predictions, dim=0)
    
    # Convert back to numpy and flatten
    x_pr = pred.numpy().flatten()
    
    # crop so that it works with scaling ratio
    x_hr = x_hr[:len(x_pr)]
    x_lr_t = x_lr_t[:len(x_pr)]
    
    # save the file
    outname = wav # + '.' + args.out_label
    sf.write(outname + '.lr.wav', x_lr_t, int(fs / args.r))
    sf.write(outname + '.hr.wav', x_hr, fs)
    sf.write(outname + '.pr.wav', x_pr, fs)
    
    if save_spectrum:
        # save the spectrum
        S = get_spectrum(x_pr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.pr.png')
        S = get_spectrum(x_hr, n_fft=2048)
        save_spectrum(S, outfile=outname + '.hr.png')
        S = get_spectrum(x_lr, n_fft=int(2048/args.r))
        save_spectrum(S, outfile=outname + '.lr.png')

def get_spectrum(x, n_fft=2048):
    """
    Compute the spectrum of a signal using Short-Time Fourier Transform (STFT)
    input: x: audio signal
    n_fft: number of FFT components
    output: S: log-scaled magnitude spectrum
    """
    S = librosa.stft(x, n_fft) # Transform de Fourier discrète
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
    """ Save the spectrum as an image file
    input: S: spectrum to save
    lim: limit for x-axis (optional)
    """
    plt.imshow(S.T, aspect=10)
    # plt.xlim([0,lim])
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()  # Ajout pour éviter les fuites mémoire