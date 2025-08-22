import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import h5py
import math

from models.afilm import get_afilm
from models.tfilm import get_tfilm


def load_h5(path):
    """Load data from an HDF5 file."""
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["X"][:], dtype=torch.float32)
        Y = torch.tensor(f["Y"][:], dtype=torch.float32)
    return X, Y


def make_parser():
    """Create an argument parser for training."""
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument('--model', default='afilm',
        choices=('afilm', 'tfilm'),
        help='model to train')
    train_parser.add_argument('--train', required=True,
        help='path to h5 archive of training patches')
    train_parser.add_argument('--val', required=True,
        help='path to h5 archive of validation set patches')
    train_parser.add_argument('-e', '--epochs', type=int, default=20,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=16,
        help='training batch size')
    train_parser.add_argument('--logname', default='tmp-run',
        help='folder where logs will be stored')
    train_parser.add_argument('--layers', default=4, type=int,
        help='number of layers in each of the D and U halves of the network')
    train_parser.add_argument('--alg', default='adam',
        help='optimization algorithm')
    train_parser.add_argument('--lr', default=3e-4, type=float,
        help='learning rate')
    train_parser.add_argument('--save_path', default="model.pth",
        help='path to save the model')
    train_parser.add_argument('--r', type=int, default=4, help='upscaling factor')
    train_parser.add_argument('--pool_size', type=int, default=4, help='size of pooling window')
    train_parser.add_argument('--strides', type=int, default=4, help='pooling stride')
    return train_parser


def get_model(args):
    """Get the model based on the specified type."""
    if args.model == 'tfilm':
        model = get_tfilm(n_layers=args.layers, scale=args.r)
    elif args.model == 'afilm':
        model = get_afilm(n_layers=args.layers, scale=args.r)
    else:
        raise ValueError('Invalid model')
    return model


class RMSELoss(nn.Module):
    """Root Mean Square Error Loss - equivalent to TensorFlow's RootMeanSquaredError"""
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


def train(args):
    """Train the model based on the provided arguments."""
    # Device selection (same logic as original)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Training model: {args.model} with {args.layers} layers and upscaling factor {args.r}")

    # Load data
    X_train, Y_train = load_h5(args.train)
    X_val, Y_val = load_h5(args.val)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), 
        batch_size=args.batch_size,
        shuffle=False
    )

    # Model
    model = get_model(args).to(device)

    # Optimizer (matching TensorFlow version exactly)
    if args.alg == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Only Adam supported for now")

    # Loss functions - both MSE and RMSE like in TensorFlow version
    criterion_mse = nn.MSELoss()
    criterion_rmse = RMSELoss()

    # Training loop
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_rmse = 0.0
        train_samples = 0

        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X)
            
            # Calculate losses
            loss_mse = criterion_mse(outputs, Y)
            loss_rmse = criterion_rmse(outputs, Y)
            
            # Backward pass
            loss_mse.backward()
            optimizer.step()
            
            # Accumulate losses
            batch_size = X.size(0)
            train_loss_mse += loss_mse.item() * batch_size
            train_loss_rmse += loss_rmse.item() * batch_size
            train_samples += batch_size

        # Calculate average training losses
        avg_train_loss_mse = train_loss_mse / train_samples
        avg_train_loss_rmse = train_loss_rmse / train_samples

        # Validation phase
        model.eval()
        val_loss_mse = 0.0
        val_loss_rmse = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                
                # Calculate validation losses
                loss_mse = criterion_mse(outputs, Y)
                loss_rmse = criterion_rmse(outputs, Y)
                
                batch_size = X.size(0)
                val_loss_mse += loss_mse.item() * batch_size
                val_loss_rmse += loss_rmse.item() * batch_size
                val_samples += batch_size

        # Calculate average validation losses
        avg_val_loss_mse = val_loss_mse / val_samples
        avg_val_loss_rmse = val_loss_rmse / val_samples

        # Print metrics (matching TensorFlow format)
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss (MSE): {avg_train_loss_mse:.6f} - RMSE: {avg_train_loss_rmse:.6f}")
        print(f"  Val   - Loss (MSE): {avg_val_loss_mse:.6f} - RMSE: {avg_val_loss_rmse:.6f}")

        # Save model after each epoch (like TensorFlow CustomCheckpoint)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_mse': avg_train_loss_mse,
            'train_loss_rmse': avg_train_loss_rmse,
            'val_loss_mse': avg_val_loss_mse,
            'val_loss_rmse': avg_val_loss_rmse,
            'args': args
        }, args.save_path)

    print(f"Training completed. Final model saved to {args.save_path}")


def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()