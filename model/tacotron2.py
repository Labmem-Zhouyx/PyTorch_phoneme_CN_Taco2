""" adapted from https://github.com/NVIDIA/tacotron2 """

from math import sqrt

import torch
from torch import nn

from .attention import LocationSensitiveAttention, AttentionWrapper
from .attention import get_mask_from_lengths
from .modules import BatchNormConv1dStack


class Prenet(nn.Module):
    """
    Prenet
        - Several linear layers with ReLU activation and dropout regularization
    """
    def __init__(self, in_dim, sizes=[256, 128], dropout=0.5):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class Postnet(nn.Module):
    """Postnet
        - A stack of five 1-d convolution layer
        - Each layer is comprised of 512 filters with shape 5*1 with Batch Normalization (BN),
          followd by tanh activations on all but the final layer
    """
    def __init__(self, hparams):
        #mel_dim, num_convs=5, conv_channels=512, conv_kernel_size=5, conv_dropout=0.5):
        super(Postnet, self).__init__()

        activations = [torch.tanh] * (hparams.postnet_num_convs - 1) + [None]
        conv_channels = [hparams.postnet_conv_channels] * (hparams.postnet_num_convs - 1) + [hparams.mel_dim]
        self.conv1ds = BatchNormConv1dStack(hparams.mel_dim, conv_channels, kernel_size=hparams.postnet_conv_kernel_size,
                                            stride=1, padding=(hparams.postnet_conv_kernel_size -1) // 2,
                                            activations=activations, dropout=hparams.postnet_conv_dropout)

    def forward(self, x):
        # transpose to (B, mel_dim, T) for convolution,
        # and then back
        return self.conv1ds(x.transpose(1, 2)).transpose(1, 2)


class Encoder(nn.Module):
    """Encoder module:
        - A stack of three 1-d convolution layers, containing 512 filters with shape 5*1,
          followd by Batch Normalization (BN) and ReLU activations
        - Bidirectional LSTM
    """
    def __init__(self, embed_dim, hparams):
        super(Encoder, self).__init__()

        # convolution layers followed by batch normalization and ReLU activation
        activations = [nn.ReLU()] * hparams.encoder_num_convs
        conv_out_channels = [hparams.encoder_conv_channels] * hparams.encoder_num_convs
        self.conv1ds = BatchNormConv1dStack(embed_dim, conv_out_channels, kernel_size=hparams.encoder_conv_kernel_size,
                                            stride=1, padding=(hparams.encoder_conv_kernel_size -1) // 2,
                                            activations=activations, dropout=hparams.encoder_conv_dropout)

        # 1 layer Bi-directional LSTM
        self.lstm = nn.LSTM(hparams.encoder_conv_channels, hparams.encoder_blstm_units // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        # transpose to (B, embed_dim, T) for convolution,
        # and then back
        x = self.conv1ds(x.transpose(1, 2)).transpose(1, 2)

        # (B, T, conv_channels)
        # TODO: pack_padded, and pad_packed?
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, encoder_output_dim, hparams):
        super(Decoder, self).__init__()

        self.mel_dim = hparams.mel_dim
        self.r = hparams.n_frames_per_step
        self.attention_context_dim = attention_context_dim = encoder_output_dim
        self.attention_rnn_units = hparams.attention_rnn_units
        self.decoder_rnn_units = hparams.decoder_rnn_units
        self.max_decoder_steps = hparams.max_decoder_steps
        self.stop_threshold = hparams.stop_threshold

        # Prenet
        self.prenet = Prenet(hparams.mel_dim, hparams.prenet_dims, hparams.prenet_dropout)

        # Attention RNN
        # (prenet_out + attention context) = attention_rnn_in -> attention_rnn_out
        self.attention_rnn = AttentionWrapper(
            nn.LSTMCell(hparams.prenet_dims[-1] + attention_context_dim, hparams.attention_rnn_units),
            LocationSensitiveAttention(hparams.attention_rnn_units, hparams.attention_dim,
                filters=hparams.attention_location_filters, kernel_size=hparams.attention_location_kernel_size)
        )
        self.attention_dropout = nn.Dropout(hparams.attention_dropout)
        # Process encoder_output as attention key
        self.memory_layer = nn.Linear(encoder_output_dim, hparams.attention_dim, bias=False)

        # Decoder RNN
        # (attention_rnn_out + attention context) = decoder_rnn_in -> decoder_rnn_out
        self.decoder_rnn = nn.LSTMCell(hparams.attention_rnn_units + attention_context_dim, hparams.decoder_rnn_units)
        self.decoder_dropout = nn.Dropout(hparams.decoder_dropout)

        # Project to mel
        self.mel_proj = nn.Linear(hparams.decoder_rnn_units + attention_context_dim, hparams.mel_dim * self.r)

        # Stop token prediction
        self.stop_proj = nn.Linear(hparams.decoder_rnn_units + attention_context_dim, 1)

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        """
        Decoder forward step.
        If decoder inputs are not given (e.g., at testing time), greedy decoding is adapted.
        Args:
            encoder_outputs: Encoder outputs. (B, T_encoder, dim)
            inputs: Decoder inputs (i.e., mel-spectrogram).
                    If None (at eval-time), previous decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for attention masking.
        Returns:
            mel_outputs: mel outputs from the decoder.
            stop_tokens: stop token outputs from the decoder.
            attn_scores: sequence of attention weights from the decoder.
        """
        B = encoder_outputs.size(0)

        # Get processed memory for attention key
        #   - no need to call for every time step
        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        # Time first: (B, T, mel_dim) -> (T, B, mel_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)
            T_decoder = inputs.size(0)

        # <GO> frames
        initial_input = encoder_outputs.data.new(B, self.mel_dim).zero_()

        # Init decoder states
        self.attention_rnn.attention_mechanism.init_attention(processed_memory)
        attention_rnn_hidden = encoder_outputs.data.new(B, self.attention_rnn_units).zero_()
        attention_rnn_cell = encoder_outputs.data.new(B, self.attention_rnn_units).zero_()
        decoder_rnn_hidden = encoder_outputs.data.new(B, self.decoder_rnn_units).zero_()
        decoder_rnn_cell = encoder_outputs.data.new(B, self.decoder_rnn_units).zero_()
        attention_context = encoder_outputs.data.new(B, self.attention_context_dim).zero_()

        # To store the result
        mel_outputs, attn_scores, stop_tokens = [], [], []

        # Run the decoder loop
        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = mel_outputs[-1][:, -1, :] if greedy else inputs[t - 1]
            t += self.r

            # Prenet
            current_input = self.prenet(current_input)

            # Attention LSTM
            (attention_rnn_hidden, attention_rnn_cell), attention_context, attention_score = self.attention_rnn(
                current_input, attention_context, (attention_rnn_hidden, attention_rnn_cell),
                encoder_outputs, processed_memory=processed_memory, mask=mask)
            attention_rnn_hidden = self.attention_dropout(attention_rnn_hidden)

            # Concat RNN output and attention context vector
            decoder_input = torch.cat((attention_rnn_hidden, attention_context), -1)

            # Pass through the decoder LSTM
            decoder_rnn_hidden, decoder_rnn_cell = self.decoder_rnn(decoder_input, (decoder_rnn_hidden, decoder_rnn_cell))
            decoder_rnn_hidden = self.decoder_dropout(decoder_rnn_hidden)

            # Contact RNN output and context vector to form projection input
            proj_input = torch.cat((decoder_rnn_hidden, attention_context), -1)

            # Project to mel
            # (B, mel_dim*r) -> (B, r, mel_dim)
            output = self.mel_proj(proj_input)
            output = output.view(B, -1, self.mel_dim)

            # Stop token prediction
            stop = self.stop_proj(proj_input)
            stop = torch.sigmoid(stop)

            # Store predictions
            mel_outputs.append(output)
            attn_scores.append(attention_score.unsqueeze(1))
            stop_tokens.extend([stop] * self.r)

            if greedy:
                if stop > self.stop_threshold:
                    break
                elif t > self.max_decoder_steps:
                    print("Warning: Reached max decoder steps.")
                    break
            else:
                if t >= T_decoder:
                    break

        # To tensor
        mel_outputs = torch.cat(mel_outputs, dim=1) # (B, T_decoder, mel_dim)
        attn_scores = torch.cat(attn_scores, dim=1) # (B, T_decoder/r, T_encoder)
        stop_tokens = torch.cat(stop_tokens, dim=1) # (B, T_decoder)

        # Validation check
        assert greedy or mel_outputs.size(1) == T_decoder

        return mel_outputs, stop_tokens, attn_scores


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()

        self.mel_dim = hparams.mel_dim

        # Embedding
        self.embedding = nn.Embedding(hparams.num_symbols, hparams.text_embedding_dim)
        std = sqrt(2.0 / (hparams.num_symbols + hparams.text_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        # Encoder
        embed_dim = hparams.text_embedding_dim
        self.encoder = Encoder(embed_dim, hparams)

        # Decoder
        encoder_out_dim = hparams.encoder_blstm_units
        self.decoder = Decoder(encoder_out_dim, hparams)

        # Postnet
        self.postnet = Postnet(hparams)

    def parse_data_batch(self, batch):
        """Parse data batch to form inputs and targets for model training/evaluating
        """
        # use same device as parameters
        device = next(self.parameters()).device

        text, text_length, mel, stop, mel_length = batch
        text = text.to(device).long()
        text_length = text_length.to(device).long()
        mel = mel.to(device).float()
        stop = stop.to(device).float()

        return (text, text_length, mel), (mel, stop)

    def forward(self, inputs):
        inputs, input_lengths, mels = inputs

        B = inputs.size(0)

        # (B, T)
        inputs = self.embedding(inputs)

        # (B, T, embed_dim)
        encoder_outputs = self.encoder(inputs)

        # (B, T, mel_dim)
        mel_outputs, stop_tokens, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=input_lengths)

        # Postnet processing
        mel_post = self.postnet(mel_outputs)
        mel_post = mel_outputs + mel_post

        return mel_outputs, mel_post, stop_tokens, alignments

    def inference(self, inputs):
        # Only text inputs
        inputs = inputs, None, None
        return self.forward(inputs)


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, predicts, targets):
        mel_target, stop_target = targets
        mel_target.requires_grad = False
        stop_target.requires_grad = False

        mel_predict, mel_post_predict, stop_predict, _ = predicts

        mel_loss = nn.MSELoss()(mel_predict, mel_target)
        post_loss = nn.MSELoss()(mel_post_predict, mel_target)
        stop_loss = nn.BCELoss()(stop_predict, stop_target)

        return mel_loss + post_loss + stop_loss, mel_loss + post_loss, stop_loss