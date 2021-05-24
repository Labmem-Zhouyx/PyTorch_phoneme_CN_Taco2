# phoneme-based Tacotron2 for Chinese
An implementation of Tacotron2 based Pytorch

Input: Chinese Pinyin Sequence

Output: Mel-spectrogram

1. Prepare training data: text and corresponding mel-spectrogram files.

Referred：https://github.com/Labmem-Zhouyx/audio2mel_preprocessor

2. Train the model.

`python train.py -o save -l logs`

3. Inference.


