import os
import random
import torch
import torchaudio
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

class SingleSpeakerVCTKSplitter:
    def __init__(self, vctk_path, seed=42):
        self.vctk_path = Path(vctk_path)
        if not self.vctk_path.exists():
            raise ValueError(f"VCTK path does not exist: {vctk_path}")
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def get_all_speakers(self):
        """Get all unique speakers from the VCTK dataset"""
        audio_files = list(self.vctk_path.glob('*.wav'))
        speakers = set()
        
        for file_path in audio_files:
            speaker_id = file_path.stem.split('_')[0]  # p225_001.wav -> p225
            speakers.add(speaker_id)
            
        return list(speakers)
    
    def get_speaker_files(self, speaker_id):
        """Get all files for a specific speaker"""
        pattern = f"{speaker_id}_*.wav"
        speaker_files = list(self.vctk_path.glob(pattern))
        return [str(f) for f in speaker_files]
    
    def select_random_speaker(self, exclude_speakers=None):
        """Select a random speaker, optionally excluding specified speakers"""
        all_speakers = self.get_all_speakers()
        
        if exclude_speakers:
            all_speakers = [s for s in all_speakers if s not in exclude_speakers]
            
        if not all_speakers:
            raise ValueError("No speakers available after exclusions")
            
        selected_speaker = random.choice(all_speakers)
        speaker_files = self.get_speaker_files(selected_speaker)
        
        print(f"Selected speaker: {selected_speaker}")
        print(f"Total files for speaker: {len(speaker_files)}")
        
        return selected_speaker, speaker_files
    
    def create_single_speaker_splits(self, speaker_id=None, test_ratio=0.07, val_ratio=0.08):
        """
        Create train/val/test splits for a single speaker.
        """
        
        # Si aucun speaker spécifié, en sélectionner un aléatoirement
        if speaker_id is None:
            speaker_id, speaker_files = self.select_random_speaker()
        else:
            speaker_files = self.get_speaker_files(speaker_id)
            if not speaker_files:
                raise ValueError(f"No files found for speaker {speaker_id}")
        
        total_files = len(speaker_files)
        train_ratio = 1.0 - test_ratio - val_ratio
        
        # Calculer les tailles
        num_test = max(1, int(total_files * test_ratio))
        num_val = max(1, int(total_files * val_ratio))
        num_train = total_files - num_test - num_val
        
        print(f"Split ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
        print(f"File counts: {num_train} train, {num_val} val, {num_test} test")
        
        # Mélanger les fichiers
        shuffled_files = speaker_files.copy()
        random.shuffle(shuffled_files)
        
        # Diviser les fichiers
        test_files = shuffled_files[:num_test]
        val_files = shuffled_files[num_test:num_test + num_val]
        train_files = shuffled_files[num_test + num_val:]
        
        # Créer la structure des splits
        splits = {
            'speaker_id': speaker_id,
            'total_files': total_files,
            'splits_info': {
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio
            },
            'train': {
                'files': train_files,
                'num_files': len(train_files)
            },
            'val': {
                'files': val_files,
                'num_files': len(val_files)
            },
            'test': {
                'files': test_files,
                'num_files': len(test_files)
            }
        }
        
        # Validation
        assert len(train_files) + len(val_files) + len(test_files) == total_files
        print("✓ Split validation passed")
        
        return splits
    
    def save_splits(self, splits, save_path=None):
        """
        Save the splits to a JSON file.
        """
        if save_path is None:
            speaker_id = splits['speaker_id']
            save_path = f"./single_speaker_splits_{speaker_id}.json"
            
        with open(save_path, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Splits saved to {save_path}")
        return save_path
    
    def load_splits(self, load_path):
        """
        Load splits from a JSON file.
        """
        with open(load_path, 'r') as f:
            return json.load(f)
    
    def get_speaker_statistics(self, speaker_id):
        """
        Get statistics for a specific speaker."""
        speaker_files = self.get_speaker_files(speaker_id)
        total_duration = 0
        durations = []
        
        for file_path in speaker_files:
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                duration = waveform.shape[1] / sample_rate
                durations.append(duration)
                total_duration += duration
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
                durations.append(3.0)  # durée estimée
                total_duration += 3.0
        
        stats = {
            'speaker_id': speaker_id,
            'num_files': len(speaker_files),
            'total_duration': total_duration,
            'avg_duration': total_duration / len(speaker_files) if speaker_files else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0
        }
        
        return stats

def main():
    # Initialization
    splitter = SingleSpeakerVCTKSplitter("/Users/rovafifaliana/.cache/kagglehub/datasets/awsaf49/vctk-sr16k-dataset/versions/1/wavs")
    
    splits = splitter.create_single_speaker_splits(
        test_ratio=0.05,
        val_ratio= ((1 - 0.05) * 20) / 100
    )
    
    # Save
    save_path = splitter.save_splits(splits)
    print(f"Splits saved to: {save_path}")
    
    # Display statistics
    speaker_id = splits['speaker_id']
    stats = splitter.get_speaker_statistics(speaker_id)
    print(f"\nStatistics for speaker {speaker_id}:")
    print(f"  Total files: {stats['num_files']}")
    print(f"  Total duration: {stats['total_duration']:.2f}s")
    print(f"  Average duration: {stats['avg_duration']:.2f}s")
    
    return splits

if __name__ == "__main__":
    splits = main()