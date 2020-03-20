import time
import numpy as np
from torchtext.data.metrics import bleu_score
import matplotlib.pyplot as plt
import Levenshtein as lev
from utils.edit_distance as ed

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

def edit_distance_by_word(candidate_corpus, reference_corpus, normalize=False, is_sum=True):
    """Computes the Edit distance (Levenshtein distance) between a candidate translation corpus and a reference
    translation corpus at token-level.

    Arguments:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        reference_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        normalize: if it's true, this function returns normalized Levenshtein score or nomalized edit distance

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = ['I', 'ate', 'the', 'apple']
        >>> references_corpus = ['I', 'ate', 'it']
        >>> edit_distance(candidate, reference, normalize=False)
            2
        >>> edit_distance(candidate, reference, normalize=True)
            0.666
    """

    assert len(candidate_corpus) == len(reference_corpus),\
        'The length of candidate and reference corpus should be the same'

    total_dist = 0.0
    dists = []
    for (candidate, ref) in zip(candidate_corpus, reference_corpus):
        # Form them as sentences
        dist = ed.edit_distance(np.array(candidate), np.array(ref))
        if normalize:
            dist = dist/len(ref.split())
        if is_sum:
            total_dist += dist
        else:
            dists.append((candidate,ref), dist)
    
    if is_sum:
        return total_dist/len(candidate_corpus)
    else:
        return dists

def edit_distance_by_char(candidate_corpus, reference_corpus, normalize=False, is_sum=True):
    """Computes the Edit distance (Levenshtein distance) between a candidate translation corpus and a reference
    translation corpus.

    Arguments:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        reference_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        normalize: if it's true, this function returns normalized Levenshtein score or nomalized edit distance

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = ['I', 'ate', 'the', 'apple']
        >>> references_corpus = ['I', 'ate', 'it']
        >>> edit_distance(candidate, reference, normalize=False)
            8
        >>> edit_distance(candidate, reference, normalize=True)
            2.666
    """

    assert len(candidate_corpus) == len(reference_corpus),\
        'The length of candidate and reference corpus should be the same'

    total_dist = 0.0

    for (candidate, ref) in zip(candidate_corpus, reference_corpus):
        # Form them as sentences
        candidate = ' '.join(candidate)
        ref = ' '.join(ref)
        dist = lev.distance(candidate.lower(), ref.lower())
        if normalize:
            dist = dist/len(ref.split())
        if is_sum:
            total_dist += dist
        else:
            dists.append((candidate,ref), dist)

    if is_sum:
        return total_dist/len(candidate_corpus)
    else:
        return dists
    
def bleu_score_by_sentence(candidate_corpus, reference_corpus, n_gram):

    assert len(candidate_corpus) == len(reference_corpus),\
        'The length of candidate and reference corpus should be the same'

    bleu_scores = []
    for (candidate, ref) in zip(candidate_corpus, reference_corpus):
        # Form them as sentences
        bleu_scores.append(bleu_score(np.array(candidate), np.array(ref), max_n=n_gram))
        
    return bleu_scores 

def calcMetrics(data, spliter, edit_dist, tick, choice):
    data = spliter(data, tick=tick, choice=choice)
    NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N = [],[],[],[],[],[],[]

    for key, value in sorted(data.items(), key=lambda x: x[0]):
        KEYS.append(str(key*tick))
        NUM_DATA.append(len(value))

        arr = np.array(value)
        bleu = bleu_score(arr[:,2], arr[:,1])
        if isinstance(bleu, tuple):
            BLEU_bp.append(bleu[0])
            BLEU_pre.append(bleu[1])
            BLEU.append(bleu[2])
        else:
            BLEU_bp.append(0)
            BLEU_pre.append(0)
            BLEU.append(0)
        EDIT.append(edit_dist(arr[:,2], arr[:,1], normalize=False))
        EDIT_N.append(edit_dist(arr[:,2], arr[:,1], normalize=True))

    return NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N

def plotResults(NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N):
    plt.plot(KEYS, NUM_DATA)
    plt.title('Number of Sentences')
    plt.show()

    plt.plot(KEYS, BLEU_pre)
    plt.title('BLEU Precision')
    plt.show()

    plt.plot(KEYS, BLEU_bp)
    plt.title('BLEU Brevity Penalty')
    plt.show()

    plt.plot(KEYS, BLEU)
    plt.title('BLEU')
    plt.show()

    plt.plot(KEYS, EDIT)
    plt.title('Edit Distance')
    plt.show()

    plt.plot(KEYS, EDIT_N)
    plt.title('Normalize Edit Distance')
    plt.show()
