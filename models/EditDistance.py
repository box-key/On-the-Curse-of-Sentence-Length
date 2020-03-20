# @Author: Kei Nemoto
# @Date:   2020-03-15T15:10:21-04:00
# @Last modified by:   Kei Nemoto
# @Last modified time: 2020-03-15T15:22:34-04:00



import Levenshtein as lev
from utils.edit_distance as ed
from numpy as np

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
            0.66
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
        >>> candidate_corpus = [['I', 'ate', 'the', 'apple'], ['I', 'did']]
        >>> references_corpus = [['I', 'ate', 'it'], ['I', 'did']]
        >>> edit_distance(candidate, reference, normalize=False)
            4.5
        >>> edit_distance(candidate, reference, normalize=True)
            1.5
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