import random
import torch
from torch.utils.tensorboard import SummaryWriter
from .plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from .plot import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, loss, speaker_loss, grad_norm, learning_rate, duration, iteration):
            self.add_scalar("training.loss", loss, iteration)
            self.add_scalar("training.speaker_loss", speaker_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, loss, speaker_loss, model, targets, predicts, iteration):
        self.add_scalar("validation.loss", loss, iteration)
        self.add_scalar("validation.speaker_loss", speaker_loss, iteration)

        _, mel_predicts, gate_predicts, alignments, _ = predicts
        if len(targets) == 4:
            _, mel_targets, gate_targets, _ = targets
        else:
            mel_targets, gate_targets, _ = targets

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_predicts[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_predicts[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
