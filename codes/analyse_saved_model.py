import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import h5py
import glob
from models.afilm import get_afilm
from models.tfilm import get_tfilm

import sys
sys.path.append("/Users/rovafifaliana/Documents/MISA/machine_learning/evaluation2/AFiLM_conversion/")

def analyze_saved_models():
    """Analyze all saved model checkpoints."""
    
    # Find all saved models
    model_files = glob.glob("*.pth") + glob.glob("*epoch*.pth")
    print(f"Found {len(model_files)} saved models:")
    
    for model_file in sorted(model_files):
        print(f"\n=== Analyzing {model_file} ===")
        
        try:
            # Load the state dict
            state_dict = torch.load(model_file, map_location='cpu')
            
            # Get model info
            total_params = sum(param.numel() for param in state_dict.values())
            print(f"Total parameters: {total_params:,}")
            
            # Show layer names and shapes
            print("Model architecture:")
            for name, param in state_dict.items():
                print(f"  {name}: {param.shape}")
                
        except Exception as e:
            print(f"Error loading {model_file}: {e}")


def evaluate_model_performance(model_path, val_data_path, model_type='afilm', layers=4, scale=2):
    """Evaluate a saved model on validation data."""
    
    print(f"\n=== Evaluating {model_path} ===")
    
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load validation data
    print("Loading validation data...")
    with h5py.File(val_data_path, "r") as f:
        X_val = torch.tensor(f["X"][:], dtype=torch.float32)
        Y_val = torch.tensor(f["Y"][:], dtype=torch.float32)
    
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=16)
    
    # Load model
    if model_type == 'afilm':
        model = get_afilm(n_layers=layers, scale=scale)
    elif model_type == 'tfilm':
        model = get_tfilm(n_layers=layers, scale=scale)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    
    # Evaluate
    criterion = nn.MSELoss()
    total_loss = 0.0
    num_samples = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            
            # Handle potential shape issues (same as in your training script)
            if outputs.dim() == 3 and outputs.shape[-1] == 2:
                outputs = outputs[:, :, 0]
            
            loss = criterion(outputs, Y)
            total_loss += loss.item() * X.size(0)
            num_samples += X.size(0)
    
    avg_loss = total_loss / num_samples
    print(f"Validation Loss: {avg_loss:.6f}")
    
    return avg_loss


def compare_all_epochs(val_data_path, model_type='afilm', layers=4, scale=4):
    """Compare performance across all saved epochs."""
    
    epoch_files = sorted(glob.glob("*epoch*.pth"))
    
    if not epoch_files:
        print("No epoch files found!")
        return
    
    print(f"\n=== Comparing {len(epoch_files)} epochs ===")
    results = []
    
    for epoch_file in epoch_files:
        try:
            # Extract epoch number from filename
            epoch_num = int(epoch_file.split('epoch')[1].split('.')[0])
            
            # Evaluate this epoch
            val_loss = evaluate_model_performance(
                epoch_file, val_data_path, model_type, layers, scale
            )
            
            results.append((epoch_num, val_loss, epoch_file))
            
        except Exception as e:
            print(f"Error evaluating {epoch_file}: {e}")
    
    # Sort by epoch and display results
    results.sort(key=lambda x: x[0])
    
    print(f"\n=== SUMMARY ===")
    print("Epoch | Val Loss  | File")
    print("-" * 40)
    
    best_loss = float('inf')
    best_epoch = None
    
    for epoch, loss, filename in results:
        marker = " <- BEST" if loss < best_loss else ""
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
        
        print(f"{epoch:5d} | {loss:.6f} | {filename}{marker}")
    
    print(f"\nBest model: Epoch {best_epoch} with validation loss {best_loss:.6f}")


if __name__ == "__main__":
    # analyze_saved_models()
    
    VAL_DATA_PATH = "vctk_single_dataset/test299.h5" 

    MODEL_PATH = "model_x2.pth"
    MODEL_TYPE = "afilm"  # or "tfilm"
    
    # evaluate specific model performance
    evaluate_model_performance(MODEL_PATH, VAL_DATA_PATH, MODEL_TYPE)

    
    # compare all epochs (uncomment the line below)
    # compare_all_epochs(VAL_DATA_PATH, MODEL_TYPE)