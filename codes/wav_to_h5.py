import os
import json
import h5py
import numpy as np
import librosa
import shutil
from pathlib import Path

def wavs_to_h5_from_files(file_list, h5_path, sr=16000, r=4, patch_size=1024):
    """
    Converts a list of .wav files into an .h5 file containing LR and HR data.
    Args:
        file_list (list): list of paths to .wav files
        h5_path (str): output .h5 file path
        sr (int): target sample rate
        r (int): downsampling factor (e.g., 4 → HR=16kHz, LR=4kHz)
        patch_size (int): patch size for training
    """
    X_list, Y_list = [], []
    
    for wav_path in file_list:
        if not wav_path.endswith(".wav"):
            continue
            
        # print(f"Processing {wav_path}...")
        
        # Check if file exists
        if not os.path.exists(wav_path):
            print(f"Warning: File {wav_path} not found, skipping...")
            continue
            
        try:
            # Load HR audio
            y_hr, _ = librosa.load(wav_path, sr=sr)
            
            # Create LR version by downsampling
            y_lr = y_hr[::r]
            
            # Restore y_lr to the same length as y_hr via simple upsampling
            y_lr_up = librosa.resample(y_lr, orig_sr=sr//r, target_sr=sr)
            
            # Adjust size for multiples of patch_size
            min_len = min(len(y_hr), len(y_lr_up))
            y_hr = y_hr[:min_len]
            y_lr_up = y_lr_up[:min_len]
            
            # Split into patches
            n_patches = min_len // patch_size
            for i in range(n_patches):
                start = i * patch_size
                end = start + patch_size
                X_list.append(y_lr_up[start:end])
                Y_list.append(y_hr[start:end])
                
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    if not X_list:
        print(f"Warning: No valid patches found for {h5_path}")
        return
    
    # Convert to numpy
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    # Save to h5
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)
    
    print(f"Saved {X.shape[0]} patches to {h5_path}")

def process_json_to_h5(json_path, output_dir=".", sr=16000, r=4, patch_size=1024):
    """
    Processes the JSON file to convert the first training block and validation data to H5.
    Args:
        json_path (str): path to JSON file
        output_dir (str): output directory for .h5 files
        sr (int): target sample rate
        r (int): downsampling factor
        patch_size (int): patch size
    """
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process first training block
    if data.get("train_blocks") and len(data["train_blocks"]) > 0:
        first_block = data["train_blocks"][0]
        if first_block.get("files"):
            print(f"Processing first training block with {len(first_block['files'])} files...")
            train_h5_path = os.path.join(output_dir, "train281.h5")
            wavs_to_h5_from_files(
                file_list=first_block["files"],
                h5_path=train_h5_path,
                sr=sr,
                r=r,
                patch_size=patch_size
            )
        else:
            print("Warning: First training block has no files")
    else:
        print("Warning: No training blocks found in JSON")
    
    # Process validation data
    if data.get("val") and data["val"].get("files"):
        print(f"Processing validation data with {len(data['val']['files'])} files...")
        val_h5_path = os.path.join(output_dir, "val281.h5")
        wavs_to_h5_from_files(
            file_list=data["val"]["files"],
            h5_path=val_h5_path,
            sr=sr,
            r=r,
            patch_size=patch_size
        )
    else:
        print("Warning: No validation files found in JSON")

    if data.get("test") and data["test"].get("files"):
        print(f"Processing test data with {len(data['test']['files'])} files...")
        test_h5_path = os.path.join(output_dir, "test281.h5")
        wavs_to_h5_from_files(
            file_list=data["test"]["files"],
            h5_path=test_h5_path,
            sr=sr,
            r=r,
            patch_size=patch_size
        )
    else:
        print("Warning: No test files found in JSON")

def main():
    # Parameters
    json_file = "single_speaker_splits_p281.json"
    output_directory = "vctk_single_dataset" 
    
    # Processing
    process_json_to_h5(
        json_path=json_file,
        output_dir=output_directory,
        sr=16000,
        r=4,
        patch_size=1024
    )

if __name__ == "__main__":
    main()