# @Author: Kei Nemoto
# @Date:   2020-03-21T14:48:07-04:00
# @Last modified by:   Kei Nemoto
# @Last modified time: 2020-03-25T17:26:35-04:00



from utils.edit_distance import edit_distance as ed
import Levenshtein as lev
from torchtext.data.metrics import bleu_score
import numpy as np

def edit_distance_by_word(candidate_corpus, reference_corpus, is_sum=True):
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

    total_dist = 0
    normalized_total_dist = 0.0
    dists = []
    for (candidate, ref) in zip(candidate_corpus, reference_corpus):
        # Form them as sentences
        dist = ed.edit_distance_by_token(candidate, ref)
        if is_sum:
            total_dist += dist
            normalized_total_dist += dist/len(ref)
        else:
            dists.append((dist, dist/len(ref)))

    if is_sum:
        return total_dist/len(candidate_corpus), normalized_total_dist/len(candidate_corpus)
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
            dists.append((candidate,ref, dist))

    if is_sum:
        return total_dist/len(candidate_corpus)
    else:
        return dists

def bleu_score_by_sentence(candidate_corpus, reference_corpus, n_gram=4):

    assert len(candidate_corpus) == len(reference_corpus),\
        'The length of candidate and reference corpus should be the same'

    bleu_scores = []
    for (candidate, ref) in zip(candidate_corpus, reference_corpus):
        # Form them as sentences
        bleu_scores.append(bleu_score([np.array(candidate)], [np.array(ref)], max_n=n_gram))

    return bleu_scores
