# pytorch-rnn-collections


All model trained with default parameters and trained on i5-7700 with RTX 2080


### COCO Image Caption

| Model  | training perplexity  | time (50 epochs)  |  parameters  |
|---|---|---|---|
| RNN  | 1.689 | 1m50s |  0.8524M |
| LSTM [1] | 1.704  | 3m21s  |  3.3161M |
|  GRU [2] | 1.676  | 3m2s  | 2.4949M  |
| QRNN [3] | 1.934  | 2m21s  | 0.9190M  |
| SRU [4] | 1.72  |  1m17s | 0.9528M  |
| SRU-Large* [4] | 1.441  | 3m4s  | 3.4749M  |
| Gated-CNN* [5] | 1.01  | 13m  | 0.0929M  |

* I am not quite sure whether Gated-CNN implementation is correct or not

* SRU-Large is 2 layer SRU with 1024 hidden dimension, other (except CNN) model only have 512 hidden dimension.

[1] Long Short Term Memory

[2] Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

[3] Quasi-Recurrent Neural Networks  

[4] Simple Recurrent Units for Highly Parallelizable Recurrence  

[5] Language Modeling with Gated Convolutional Networks


## Requirements

- python 3.6

- Install python packages

    ```bash
    pip install -r requirements.txt
    ```