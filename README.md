# Description

This is a research project on why Neural Machine Translation (NMT) models yield low performance when it translates long sentences. We study various factors inlcuding the effect of unk tokens in source and reference, the number of heads in transformer, etc. We employ (Normalized) Edit/Levenstein Distance and Brevity Penalty and Precision in BLEU score. We test various models including Bi-LSTM, Attention, Convolution, and Transformer on Europarl-v7 parallel corpus for English-French (https://www.statmt.org/europarl/).

# Libraries
- Pytorch (1.4.0)
- Cython (0.29.6)
- torchtext (0.5.0)
- python-Levenshtein (0.12.0)
- spacy (2.2.3)
- fairseq (0.9.0)
- numpy (1.18.1)
- scipy (1.2.2)

# Repository Structure
Pre-trained models in fairseq, custom models and translation datasets are stored in data-bin. Methods and NMT models including Bi-LSTM, Attention, and Transformer are stored in models library. Tutorial Archive stores codes with my comments on tutorial: https://github.com/bentrevett/pytorch-seq2seq. All bash files are written for windows environment.

# Fairseq Models
Please execute codes line by line in ".sh" files in fairseq_commands folder.
Before run them, make sure you download the following pre-trained models on fairseq and place them in data-bin:
- Convolutional trained on WMT14 English-French: https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2
- Transformer trained on WMT14 English-French: https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2

