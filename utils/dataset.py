import os
import random
import numpy as np
import torch

from text import text_to_sequence
from text.speakers import speaker_to_id


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, melpaths_and_text, hparams):
        self.text_cleaners = hparams.text_cleaners
        self.mel_dim = hparams.mel_dim
        self.f_list = self.files_to_list(melpaths_and_text)
        self.melpath_prefix = hparams.melpath_prefix
        random.seed(hparams.seed)
        random.shuffle(self.f_list)

    def files_to_list(self, file_path, split='|'):
        with open(file_path, encoding = 'utf-8') as f:
            f_list = [line.strip().strip('\ufeff').split(split) for line in f] #remove BOM
        return f_list

    def get_mel_text_pair(self, melpath_and_content):
        # separate filename and text
        melpath, text, speaker = melpath_and_content[1], melpath_and_content[6], melpath_and_content[7]
        text = self.get_text(text)

        if self.melpath_prefix:
            melpath = os.path.join(self.melpath_prefix, melpath)
        mel = self.get_mel(melpath)

        spk_id = torch.IntTensor([speaker_to_id[speaker]])
        return (text, mel, spk_id)

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def get_mel(self, file_path):
        #stored melspec: np.ndarray [shape=(T_out, num_mels)]
        melspec = torch.from_numpy(np.load(file_path))
        assert melspec.size(1) == self.mel_dim, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(1), self.mel_dim))
        return melspec

    def __len__(self):
        return len(self.f_list)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.f_list[index])


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        speakers = torch.LongTensor(len(batch))
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.shape[0]] = text
            speakers[i] = batch[ids_sorted_decreasing[i]][2]

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(1)
        max_target_len = max([x[1].size(0) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), max_target_len, num_mels)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :mel.size(0), :] = mel
            gate_padded[i, mel.size(0)-1:] = 1
            output_lengths[i] = mel.size(0)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, speakers


class TextMelDatasetEval(torch.utils.data.Dataset):
    def __init__(self, sentences, hparams):
        self.sentences = sentences
        self.text_cleaners = hparams.text_cleaners
        self.type = hparams.speaker_embedding_type
        self.melpath_prefix = hparams.melpath_prefix
        self.mel_dim = hparams.mel_dim

    def get_mel(self, file_path):
        # stored melspec: np.ndarray [shape=(T_out, num_mels)]
        melspec = torch.from_numpy(np.load(file_path))
        assert melspec.size(1) == self.mel_dim, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(1), self.mel_dim))
        return melspec

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        text, speaker = self.sentences[index].split('|')
        if self.type == 'one-hot':
            spk_id = torch.IntTensor([speaker_to_id[speaker]])
            ref_mel = None
        else:
            spk_id = None
            if self.melpath_prefix:
                speaker = os.path.join(self.melpath_prefix, speaker)
            ref_mel = self.get_mel(speaker)

        return torch.IntTensor(text_to_sequence(text, self.text_cleaners)), spk_id, ref_mel
