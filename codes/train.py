import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import h5py

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
    train_parser.add_argument('--save_path', default="model_afilm_single_2.pth",
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


def train(args):
    """Train the model based on the provided arguments."""
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

    train_loader = DataLoader(TensorDataset(X_train, Y_train), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), 
                            batch_size=args.batch_size)

    # Model
    model = get_model(args).to(device)

    # Optimizer
    if args.alg == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Only Adam supported for now")

    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            if outputs.dim() == 3 and outputs.shape[-1] == 2:
                outputs = outputs[:, :, 0]
            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Y shape: {Y.shape}")
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                loss = criterion(outputs, Y)
                val_loss += loss.item() * X.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Save checkpoint with epoch number
        # torch.save(model.state_dict(), args.save_path)
        torch.save(model.state_dict(), f"{args.save_path}_epoch{epoch+1}.pth")


def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
