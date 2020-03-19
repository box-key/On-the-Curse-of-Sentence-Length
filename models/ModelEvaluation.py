import time
import numpy as np

def calc_time(start_time):
    total = time.time() - start_time
    return int(total/60), int(total%60)

def loadFairseqOutput(out_path):
    out = open(out_path, mode='rt', encoding='utf-8').read().strip().split('\n')
    # first 7 lines and the 4 lines are irrelevant
    out_red = out[7:-4]
    # iterate through every 4 lines (src,ref,hypothesis,log_probs)
    n = 4
    out_pair = [out_red[i * n:(i + 1) * n] for i in range((len(out_red) + n - 1) // n )]
    matrix = []
    # The last element is irrelevant
    for pair in out_pair:
        src = pair[0].split('\t')[1].split(' ')
        ref = pair[1].split('\t')[1].split(' ')
        hyp = pair[2].split('\t')[2].split(' ')
        # remove data <= 1 words
        if len(src) > 1 and len(ref) > 1 and len(hyp) > 1:
            matrix.append([src, ref, hyp])
    return np.array(matrix)

def splitDatabySentenceLength(triples, tick, choice):
    
    assert (choice=='src' or choice=='ref'), 'choice should be src or ref'

    max_len = 0
    data = {}
    for triple in triples:
        # store ref length or target length, depending on choice
        # triple[0] = src, triple[1] = ref, triple[2] = hypothesis
        sentence_len = len(triple[0]) if choice == 'src' else len(triple[1])
        sentence_range = int(sentence_len/tick)
        # check if sentence_range is OOD (Out of Dictionary)
        # if the number already exists in data, append the pair to list
        if sentence_range in data:
            data[sentence_range].append(triple)
        # otherwise, add new element and list associated with it
        else:
            data.update({sentence_range:[triple]})
    return data

def splitDatabyNumberOfUnknowns(triples, tick, choice):

    assert (choice=='src' or choice=='ref'), 'choice should be src or ref'

    data = {}
    # go through all the src-trg pairs
    for triple in triples:
        # Get senetnce of src-trg
        # use count bc it's optimized for list data type
        if choice == 'src':
            # count # of unknown words in a sentence
            # unk is stored as <unk> in src
            num_unk = triple[0].count('<unk>')
        else:
            # unk is stored as <unk> in src
            num_unk = triple[1].count('<<unk>>')
        # squash the number into range
        num_unk_range = int(num_unk/tick)
        # check if num_unk is OOD (Out of Dictionary)
        # if the number already exists in data, append the pair to list
        if num_unk_range in data:
            data[num_unk_range].append(triple)
        # otherwise, add new element and list associated with it
        else:
            data.update({num_unk_range:[triple]})

    return data
