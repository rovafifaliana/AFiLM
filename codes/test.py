import argparse
import torch

from models.afilm import AFiLM
from models.tfilm import TFiLM
from utils import upsample_wav


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', required=True,
        help='path to pre-trained model (PyTorch .pth)')
    parser.add_argument('--out-label', default='',
        help='append label to output samples')
    parser.add_argument('--wav-file-list',
        help='list of audio files for evaluation')
    parser.add_argument('--layers', default=4, type=int,
        help='number of layers in each of the D and U halves of the network')
    parser.add_argument('--r', help='upscaling factor', default=4, type=int)
    parser.add_argument('--sr', help='high-res sampling rate',
        type=int, default=16000)
    parser.add_argument('--patch_size', type=int, default=8192,
        help='Size of patches over which the model operates')
    return parser


def load_model(args):
    # Choix du modèle
    if "tfilm" in args.pretrained_model.lower():
        model = TFiLM(n_layers=args.layers, block_size=args.patch_size, n_filters=64, scale=args.r)
    else:
        model = AFiLM(n_layers=args.layers, block_size=args.patch_size, n_filters=64, scale=args.r)

    # Charger les poids
    state_dict = torch.load(args.pretrained_model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def test(args):
    model = load_model(args)
    upsample_wav(args.wav_file_list, args, model)


def main():
    parser = make_parser()
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()
