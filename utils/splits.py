# @Author: Kei Nemoto
# @Date:   2020-03-21T14:47:28-04:00
# @Last modified by:   Kei Nemoto
# @Last modified time: 2020-03-25T17:26:41-04:00



import math

def splitDatabySentenceLength(triples, tick, choice):

    assert (choice=='src' or choice=='ref'), 'choice should be src or ref'

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
        # unk is stored as <unk> in src
        # unk is stored as <<unk>> in ref
        num_unk = triple[0].count('<unk>') if choice == 'src' else triple[1].count('<<unk>>')
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

def splitDatabyFractionOfUnknowns(triples, choice, tick=1):

    assert (choice=='src' or choice=='ref'), 'choice should be src or ref'

    # Data = {0.0-(0.1*tick) : [], (0.1*tick)-(0.2*tick): [], ..., 1.0: []}
    max_range = int(10/tick)
    data = {idx: [] for idx in range(max_range)}

    # go through all the src-trg pairs
    for triple in triples:
        # unk is stored as <unk> in src
        # unk is stored as <<unk>> in ref
        frac_unk = float(triple[0].count('<unk>')/len(triple[0])) if choice == 'src' else float(triple[1].count('<<unk>>')/len(triple[1]))
        # squash the number into range
        frac_unk = frac_unk if frac_unk <1 else 0.99
        idx = math.floor(frac_unk*10/tick)
        data[idx].append(triple)

    return data

def merge_bin(bins, target_bin):
    delete = []
    for key in bins.keys():
        if key > target_bin:
            # merge bin to target
            bins[target_bin] += bins[key]
            # store the keys of bins to be removed
            delete.append(key)
    # Delete mereged bins
    for key in delete: del bins[key]
    return bins

def print_bins(bins, tick):
    for key, val in sorted(bins.items(), key=lambda x: x[0]):
        print(f'Bin {key*tick}-{key*tick+tick} has {len(val)} sentences')
        
def print_merged_bins(bins, tick):
    for key, val in sorted(bins.items(), key=lambda x: x[0]):
        ran = f'{key*tick}-{key*tick+tick}' if key+1<len(bins) else f'{key*tick}-'
        print(f'Bin {ran} has {len(val)} sentences')