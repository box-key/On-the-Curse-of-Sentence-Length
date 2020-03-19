# Description

This is a research project on why Neural Machine Translation (NMT) models yield low performance when it translates long sentences. We study various factors inlcuding the effect of unk tokens in source and reference, the number of heads in transformer, etc. We employ (Normalized) Edit/Levenstein Distance and Brevity Penalty and Precision in BLEU score. We test various models including Bi-LSTM, Attention, Convolution, and Transformer.

# Libraries
- Pytorch
- torchtext
- python-Levenshtein 
- spacy
- fairseq
- numpy
- scipy

# Repository Structure
Pre-trained models in fairseq, custom models and translation datasets are stored in data-bin. Methods and NMT models including Bi-LSTM, Attention, and Transformer are stored in models library. Tutorial Archive stores codes with my comments on tutorial: https://github.com/bentrevett/pytorch-seq2seq. All bash files are written for windows environment.