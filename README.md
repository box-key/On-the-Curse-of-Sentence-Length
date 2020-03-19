# Description

This is a research project on why Neural Machine Translation (NMT) models yield low performance when it translates long sentences. We study various factors inlcuding the effect of unk tokens in source and reference, the number of heads in transformer, etc. We employ (Normalized) Edit/Levenstein Distance and Brevity Penalty and Precision in BLEU score. We test various models including Bi-LSTM, Attention, Convolution, and Transformer.

# Libraries
- Pytorch (1.4.0)
- torchtext (0.5.0)
- python-Levenshtein (0.12.0)
- spacy (2.2.3)
- fairseq (0.9.0)
- numpy (1.18.1)
- scipy (1.2.2)

# Repository Structure
Pre-trained models in fairseq, custom models and translation datasets are stored in data-bin. Methods and NMT models including Bi-LSTM, Attention, and Transformer are stored in models library. Tutorial Archive stores codes with my comments on tutorial: https://github.com/bentrevett/pytorch-seq2seq. All bash files are written for windows environment.