import os
import json
import h5py
import numpy as np
import librosa
import shutil
from pathlib import Path

def wavs_to_h5_from_files(file_list, h5_path, sr=16000, r=4, patch_size=1024):
    """
    Convertit une liste de fichiers .wav en un fichier .h5 contenant les données LR et HR.
    Args:
        file_list (list): liste des chemins vers les fichiers .wav
        h5_path (str): chemin du fichier de sortie .h5
        sr (int): sample rate cible
        r (int): facteur de sous-échantillonnage (ex: 4 → HR=16kHz, LR=4kHz)
        patch_size (int): taille des patches pour l'entraînement
    """
    X_list, Y_list = [], []
    
    for wav_path in file_list:
        if not wav_path.endswith(".wav"):
            continue
            
        # print(f"Processing {wav_path}...")
        
        # Vérifier si le fichier existe
        if not os.path.exists(wav_path):
            print(f"Warning: File {wav_path} not found, skipping...")
            continue
            
        try:
            # Charger audio HR
            y_hr, _ = librosa.load(wav_path, sr=sr)
            
            # Créer la version LR en sous-échantillonnant
            y_lr = y_hr[::r]
            
            # Remettre y_lr à la même longueur que y_hr via upsampling simple
            y_lr_up = librosa.resample(y_lr, orig_sr=sr//r, target_sr=sr)
            
            # Ajuster la taille pour multiples de patch_size
            min_len = min(len(y_hr), len(y_lr_up))
            y_hr = y_hr[:min_len]
            y_lr_up = y_lr_up[:min_len]
            
            # Découper en patches
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
    
    # Conversion en numpy
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    # Sauvegarde dans h5
    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("X", data=X)
        hf.create_dataset("Y", data=Y)
    
    print(f"Saved {X.shape[0]} patches to {h5_path}")

def process_json_to_h5(json_path, output_dir=".", sr=16000, r=4, patch_size=1024):
    """
    Traite le fichier JSON pour convertir le premier bloc d'entraînement et les données de validation en H5.
    Args:
        json_path (str): chemin vers le fichier JSON
        output_dir (str): répertoire de sortie pour les fichiers .h5
        sr (int): sample rate cible
        r (int): facteur de sous-échantillonnage
        patch_size (int): taille des patches
    """
    # Charger le fichier JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Traiter le premier bloc d'entraînement
    if data.get("train_blocks") and len(data["train_blocks"]) > 0:
        first_block = data["train_blocks"][0]
        if first_block.get("files"):
            print(f"Processing first training block with {len(first_block['files'])} files...")
            train_h5_path = os.path.join(output_dir, "train_block_0.h5")
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
    
    # # Traiter les données de validation
    # if data.get("val") and data["val"].get("files"):
    #     print(f"Processing validation data with {len(data['val']['files'])} files...")
    #     val_h5_path = os.path.join(output_dir, "val.h5")
    #     wavs_to_h5_from_files(
    #         file_list=data["val"]["files"],
    #         h5_path=val_h5_path,
    #         sr=sr,
    #         r=r,
    #         patch_size=patch_size
    #     )
    # else:
    #     print("Warning: No validation files found in JSON")

    # if data.get("test") and data["test"].get("files"):
    #     print(f"Processing test data with {len(data['test']['files'])} files...")
    #     test_h5_path = os.path.join(output_dir, "test.h5")
    #     wavs_to_h5_from_files(
    #         file_list=data["test"]["files"],
    #         h5_path=test_h5_path,
    #         sr=sr,
    #         r=r,
    #         patch_size=patch_size
    #     )
    # else:
    #     print("Warning: No test files found in JSON")

def main():
    # Paramètres
    json_file = "/Users/rovafifaliana/Documents/MISA/machine_learning/evaluation2/AFiLM_conversion/data_splits_1.json"
    output_directory = "/Users/rovafifaliana/Documents/MISA/machine_learning/evaluation2/AFiLM_conversion/vctk_single_dataset" 
    
    # Traitement
    process_json_to_h5(
        json_path=json_file,
        output_dir=output_directory,
        sr=16000,
        r=4,
        patch_size=1024
    )

if __name__ == "__main__":
    main()