import sys
import numpy as np
import torch
import os
import argparse
import yaml
import json
from torch.utils.data import DataLoader
from PIL import Image
from scipy.io.wavfile import write

from hparams import hparams
from model.tacotron2 import Tacotron2
from utils.dataset import TextMelDatasetEval
from utils.plot import plot_reference_alignment_to_numpy, plot_spectrogram_to_numpy
import hifigan


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

def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/LJSpeech/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/AISHELL3/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder

def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy() * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

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

    # Read Config
    preprocess_config = yaml.load(open('./hifigan/AISHELL3/preprocess.yaml', "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open('./hifigan/AISHELL3/model.yaml', "r"), Loader=yaml.FullLoader)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    sentences = get_sentences(args)
    testset = TextMelDatasetEval(sentences, hparams)

    with torch.no_grad():
        for i, batch in enumerate(testset):
            inputs, speaker_ids, ref_mels = batch
            predicts = model.inference(inputs, speaker_ids, ref_mels)
            _, mel_post_predict, _, _, ref_alignments, _, _, _, _ = predicts
            if hparams.speaker_embedding_type.startswith('local'):
                im = Image.fromarray(plot_reference_alignment_to_numpy(ref_alignments[0].data.cpu().numpy().T))
                im.save(os.path.join(args.outdir, 'sentence_{}_reference_alignment.jpg'.format(i)))
            mel = mel_post_predict[0].detach()
            wav_prediction = vocoder_infer(
                mel.unsqueeze(0).transpose(1,2),
                vocoder,
                model_config,
                preprocess_config,
            )[0]

            audio_path = os.path.join(args.wavdir, 'sentence_{}_hifigan.wav'.format(i))
            write(audio_path, preprocess_config["preprocessing"]["audio"]["sampling_rate"], wav_prediction)

            mel = mel.cpu().numpy()
            print('CHECK MEL SHAPE:', mel.shape)

            mel_path = os.path.join(args.outdir, 'sentence_{}_mel-feats.npy'.format(i))
            np.save(mel_path, mel, allow_pickle=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--sentences', type=str, help='text to infer', default='hao3 jia1 huo5 |SSB0005')
    parser.add_argument('-t', '--text', type=str, help='text file to infer', default='./sentences.txt')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path', default='./save/checkpoint_100000')
    parser.add_argument('-o', '--outdir', type=str, help='output filename', default='./inference_mels')
    parser.add_argument('--wavdir', type=str, help='output filename', default='./inference_wavs')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.wavdir, exist_ok=True)
    inference(args)
