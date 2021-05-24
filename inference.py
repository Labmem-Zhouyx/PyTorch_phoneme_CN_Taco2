import sys
import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader

from hparams import hparams
from model.tacotron2 import Tacotron2
from utils.dataset import TextMelDatasetEval


def get_sentences(args):
    if args.text != '':
        with open(args.text, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = [args.sentences]
    print("Check sentences:", sentences)
    return sentences


def load_model(hparams, checkpoint_path):

    model = Tacotron2(hparams)
    print("Loaded checkpoint of model: '{}'" .format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    return model

def inference(args):

    # Prepare device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    # Load model
    model = load_model(hparams, args.checkpoint)
    model = model.to(device)
    model.eval()

    sentences = get_sentences(args)
    testset = TextMelDatasetEval(sentences, hparams)

    with torch.no_grad():
        for i, input in enumerate(testset):
            inputs = input.unsqueeze(0)
            predicts = model.inference(inputs)
            mel_predict, mel_post_predict, stop_predict, _ = predicts

            mels = mel_post_predict[0].cpu().numpy()
            print('CHECK MEL SHAPE:', mels.shape)

            mel_path = os.path.join(args.outdir, 'sentence_{}_mel-feats.npy'.format(i))
            np.save(mel_path, mels, allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--sentences', type=str, help='text to infer', default='hao3 jia1-huo5.')
    parser.add_argument('-t', '--text', type=str, help='text file to infer', default='./sentences.txt')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path', default='./save/checkpoint_100000')
    parser.add_argument('-o', '--outdir', type=str, help='output filename', default='./inference_mels')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    inference(args)