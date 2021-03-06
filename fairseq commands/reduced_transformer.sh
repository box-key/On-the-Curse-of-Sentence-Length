# set variables
set MODEL_DIR=C:/Users/under/Jupyter-Projects/My-Research/On-the-Curse-of-Sentence-Length
set DATA_DIR=C:/Users/under/Datasets/Neural-Machine-Translation/Europarl_fr-en_reduced
cd %DATA_DIR%

# preprocess external data for pretrained models
fairseq-preprocess --source-lang en --target-lang fr --trainpref train --validpref val --testpref test --tokenizer moses --bpe subword_nmt --destdir %MODEL_DIR%/data-bin/europarl-v7.fr-en.reduced_transformer --srcdict %MODEL_DIR%/data-bin/wmt14.en-fr.joined-dict.transformer/dict.en.txt --tgtdict %MODEL_DIR%/data-bin/wmt14.en-fr.joined-dict.transformer/dict.fr.txt --workers 8

cd %MODEL_DIR%

# generate hypothesis for external data
# fairseq-generate data-bin/europarl-v7.fr-en.reduced --source-lang en --target-lang fr --path data-bin/wmt14.en-fr.fconv-py/model.pt --gen-subset test --beam=3 --batch-size=32 > out.txt
set PYTHONIOENCODING=utf-8
fairseq-generate data-bin/europarl-v7.fr-en.reduced_transformer --path data-bin/wmt14.en-fr.joined-dict.transformer/model.pt --beam 3 --batch-size 32 --source-lang en --target-lang fr --tokenizer moses --bpe subword_nmt --bpe-codes data-bin/wmt14.en-fr.joined-dict.transformer/bpecodes > data-bin/out_reduced_transformer.txt
