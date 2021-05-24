# phoneme-based Tacotron2 for Chinese
An implementation of Tacotron2 based Pytorch

Input: Chinese Pinyin Sequence
Output: Mel-spectrogram

1. Prepare training data: text and corresponding mel-spectrogram files.

2. Train the model.

`python train.py -o save -l logs`

3. Inference.

`python inference.py -t 'sentences.txt' -c ./save/checkpoint_100000`
