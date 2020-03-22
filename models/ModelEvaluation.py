import time
import numpy as np
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score
import pandas as pd
import seaborn as sns
from models.utils import metrics as m
from scipy.stats import spearmanr

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

def calcAverage(data, spliter, edit_dist, tick, choice):
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
        start_time = time.time()
        edit, edit_n = edit_dist(arr[:,2], arr[:,1])
        EDIT.append(edit)
        EDIT_N.append(edit_n)
    return NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N

def plot_points(factor, metric, title, xbins=10):
    factor = [round(x*xbins,2) for x in factor]
    df=pd.DataFrame(list(zip(factor, metric)), columns=['factor', 'metric'])
    plt.figure(figsize=(10,6))
    sns.stripplot(data=df, x='factor', y='metric', color='green',jitter=0.2)
    plt.title(title)
    plt.locator_params(axis='x', nbins=xbins)
    plt.show()

# def plot_metrics(metric, factors, title, xbins=10):
#     fig, ax = plt.subplots(2,3,sharex=True,sharey=True, tight_layout=True, figsize=(20,15))
#     for row in range(2):
#         for col in range(3):
#             factor = factors[row*col+col]
#             factor = [round(x*xbins,2) for x in factor]
#             df=pd.DataFrame(list(zip(factor, metric)), columns=['factor', 'metric'])
#             sns.stripplot(data=df, x='factor', y='metric', color='green',jitter=0.2, ax=ax[row][col])
#             ax[row][col].locator_params(axis='x', nbins=xbins)
#     plt.setp(ax[:,0], ylabel='Source', fontsize=12)
#     plt.setp(ax[:,0], ylabel='Reference', fontsize=12)
#     plt.setp(ax[-1,:], xlabel='Length', fontsize=12)
#     plt.setp(ax[-1,:], xlabel='Unk Rate', fontsize=12)
#     plt.setp(ax[-1,:], xlabel='# of Unk', fontsize=12)


def get_factors(data):
    num_unks_src = [triple[0].count('<unk>') for triple in data]
    num_unks_ref = [triple[1].count('<<unk>>') for triple in data]
    frac_unks_src = [round(triple[0].count('<unk>')/len(triple[0]),2) for triple in data]
    frac_unks_ref = [round(triple[1].count('<<unk>>')/len(triple[1]),2) for triple in data]
    src_len = [len(triple[0]) for triple in data]
    ref_len = [len(triple[1]) for triple in data]
    return (('Source Length', src_len), \
            ('# Unknowns in Source',num_unks_src), \
            ('% Unknowns in Source', frac_unks_src), \
            ('Refenrence Length', ref_len), \
            ('# Unknowns in Reference', num_unks_ref), \
            ('% Unknowns in Reference', frac_unks_ref))

def get_metrics(data, n_gram):
    assert isinstance(data, np.ndarray), \
        'data should be numpy array with 3 dimension (source, reference, hypothesis)'

    start_time = time.time()
    edit_dists = m.edit_distance_by_word(data[:,2], data[:,1], is_sum=False)
    min, sec = calc_time(start_time)
    print(f'Edit Distance takes: {min} min {sec} sec')

    start_lime = time.time()
    bleus = m.bleu_score_by_sentence(data[:,2], data[:,1], n_gram=n_gram)
    min, sec = calc_time(start_time)
    print(f'BLEU score takes: {min} min {sec} sec')

    edit_dist = [dist[0] for dist in edit_dists]
    edit_dist_norm = [dist[1] for dist in edit_dists]
    bleu_bp = [b[0] for b in bleus]
    bleu_pre = [b[1] for b in bleus]
    bleu_score = [b[2] for b in bleus]
    return (('Edit Distance', edit_dist), \
            ('Normalized Edit Distance', edit_dist_norm), \
            ('Bleu Precision', bleu_bp), \
            ('Bleu Brevity Penalty', bleu_pre), \
            ('Bleu Score', bleu_score))

def plot_metric(metric, factors, xbins=10):
    for factor in factors:
        r,p = spearmanr(factor[1], metric[1])
        plot_points(factor=factor[1], metric=metric[1], \
                    title=f'{metric[0]} by {factor[0]} - Spearman coefficient: {round(r,4)}', \
                    xbins=xbins)

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
