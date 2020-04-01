import time
import numpy as np
import matplotlib.pyplot as plt
from torchtext.data.metrics import bleu_score
import pandas as pd
import seaborn as sns
from .metric import bleu as b
from .metric import edit_distance as ed

def calc_time(start_time):
    total = time.time() - start_time
    return int(total/60), int(total%60)

def calcAverage(data, tick, ngram, choice=None, spliter=None):
    if spliter is not None:
        data = spliter(data, tick=tick, choice=choice)
    NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N = [],[],[],[],[],[],[]
    for key, value in sorted(data.items(), key=lambda x: x[0]):
        KEYS.append(str(key*tick))
        NUM_DATA.append(len(value))

        arr = np.array(value)
        bleu_score = b.bleu_corpus(arr[:,1].tolist(), arr[:,2].tolist(), ngram)
        if isinstance(bleu_score, list):
            BLEU_pre.append(bleu_score[0])
            BLEU_bp.append(bleu_score[1])
            BLEU.append(bleu_score[2])
        else:
            BLEU_bp.append(0)
            BLEU_pre.append(0)
            BLEU.append(0)
        start_time = time.time()
        edist = ed.edit_distance_corpus(arr[:,1].tolist(), arr[:,2].tolist())
        EDIT.append(edist[0])
        EDIT_N.append(edist[1])
    return NUM_DATA, KEYS, BLEU, BLEU_pre, BLEU_bp, EDIT, EDIT_N

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
    edit_dists = ed.edit_distance_points(data[:,1].tolist(), data[:,2].tolist())
    min, sec = calc_time(start_time)
    print(f'Edit Distance takes: {min} min {sec} sec')

    start_lime = time.time()
    bleus = b.bleu_points(data[:,1].tolist(), data[:,2].tolist(), n_gram)
    min, sec = calc_time(start_time)
    print(f'BLEU score takes: {min} min {sec} sec')

    edit_dist = [dist[0] for dist in edit_dists]
    edit_dist_norm = [dist[1] for dist in edit_dists]
    bleu_pre = [bleu[0] for bleu in bleus]
    bleu_bp = [bleu[1] for bleu in bleus]
    bleu_score = [bleu[2] for bleu in bleus]
    return (('Edit Distance', edit_dist), \
            ('Normalized Edit Distance', edit_dist_norm), \
            ('Bleu Precision', bleu_pre), \
            ('Bleu Brevity Penalty', bleu_bp), \
            ('Bleu Score', bleu_score))

def plot_factors_hist(factors, nbin):
    for factor in factors:
        max_val = max(factor[1])
        plt.hist(factor[1], bins=nbin)
        plt.title(f'Distribution of {factor[0]} - max={max_val}')
        plt.xlim(0,max_val)
        plt.show()

def plot_metric(metric, factors, xbins=10, remove_zero=False):
    for factor in factors:
        if remove_zero:
            pair = np.array(list(filter(lambda x: x[1]>0, zip(factor[1], metric[1]))))
        else:
            pair = np.array(list(zip(factor[1], metric[1])))
        plt.figure(figsize=(10,6))
        plt.title(f'{metric[0]} with {factor[0]}')
        plt.scatter(pair[:,0], pair[:,1], color='green')

def plot_scatter_line(metrics_scatter, factor_scatter, metrics_line, factor_line, xlim, remove_zero=False):
    for metric_scatter, metric_line in zip(metrics_scatter, metrics_line):
        if remove_zero:
            pair = np.array(list(filter(lambda x: x[1]>0, zip(factor_scatter[1], metric_scatter[1]))))
        else:
            pair = np.array(list(zip(factor_scatter[1], metric_scatter[1])))
        plt.figure(figsize=(10,6))
        plt.title(f'{metric_scatter[0]} with {factor_scatter[0]}')
        plt.plot(factor_line, metric_line, marker='D', color='orange', markersize=8)
        plt.scatter(pair[:,0], pair[:,1])
        plt.xlim(0,xlim)
        if metric_scatter[0]=='Normalized Edit Distance':
            plt.ylim(0,3.1)
        plt.show()

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
