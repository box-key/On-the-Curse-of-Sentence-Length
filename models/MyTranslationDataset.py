from torchtext.data import Example
import random
from torchtext import data
import os
import io

def readFiles(src_path, trg_path):
    src = open(src_path, mode='rt', encoding='utf-8').read().strip().split('\n')
    trg = open(trg_path, mode='rt', encoding='utf-8').read().strip().split('\n')
    return src, trg

def splits(pairs, train=0.6, val=0.2, test=0.2):
    random.shuffle(pairs)
    length = len(pairs)
    train_data = pairs[:int(train*length)]
    val_data = pairs[int(train*length):int((train+val)*length)]
    test_data = pairs[int((1-test)*length):]
    return train_data, val_data, test_data

def prepareTranslationData(src_path, trg_path, proportions, fields):
    if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
    src, trg = readFiles(src_path, trg_path)
    examples = [Example.fromlist(data=[src_line, trg_line], fields=fields) for src_line, trg_line in zip(src, trg)]
    train, val, test = splits(examples, train=proportions[0], val=proportions[1], test=proportions[2])
    return tuple(MyTranslationDataset(data,fields) for data in (train, val, test) if data is not None)

def exportData(src_path, trg_path, exts, proportions, export_path, reduce_rate=1):
    src, trg = readFiles(src_path, trg_path)
    pairs = [(s,t) for s, t in zip(src, trg)]
    if reduce_rate < 1:
        random.shuffle(pairs)
        pairs = pairs[:int(len(pairs)*reduce_rate)]
    train, val, test = splits(pairs,
                              train=proportions[0],
                              val=proportions[1],
                              test=proportions[2])
    
    if not os.path.exists(export_path):
        os.mkdir(export_path)
    
    # Export training sets
    with io.open(os.path.join(export_path,'train'+exts[0]), mode='w', encoding='utf-8') as f_src, \
         io.open(os.path.join(export_path,'train'+exts[1]), mode='w', encoding='utf-8') as f_trg:
        # src file
        f_src.write(' \n '.join([d[0] for d in train]))
        # trg file
        f_trg.write(' \n '.join([d[1] for d in train]))
        # close files
        f_src.close()
        f_trg.close()

    # Export validation sets
    with io.open(os.path.join(export_path,'val'+exts[0]), mode='w', encoding='utf-8') as f_src, \
         io.open(os.path.join(export_path,'val'+exts[1]), mode='w', encoding='utf-8') as f_trg:
        # src file
        f_src.write(' \n '.join([d[0] for d in val]))
        # trg file
        f_trg.write(' \n '.join([d[1] for d in val]))
        # close files
        f_src.close()
        f_trg.close()

    # Export test sets
    with io.open(os.path.join(export_path,'test'+exts[0]), mode='w', encoding='utf-8') as f_src, \
         io.open(os.path.join(export_path,'test'+exts[1]), mode='w', encoding='utf-8') as f_trg:
        # src file
        f_src.write(' \n '.join([d[0] for d in test]))
        # trg file
        f_trg.write(' \n '.join([d[1] for d in test]))
        # close files
        f_src.close()
        f_trg.close()

class MyTranslationDataset(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, examples, fields, **kwargs):
        super(MyTranslationDataset, self).__init__(examples, fields, **kwargs)
        
