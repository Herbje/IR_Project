import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import json

def add_to_dict(result_, input_):
    if input_ in result_: result_[input_] += 1
    else: result_[input_] = 1

def load_json(file, fraction, update_dict):
    data = json.load(file)
    words = [tuple(d) for d in data['coef']]
    barrier = fraction * max(words, key=lambda item: item[1])[1]
    keep = [k for k, v in words if v >= barrier]
    [add_to_dict(update_dict, k) for k in keep]
    return

def iterate_lirme_results(runs):
    df = pd.read_csv(Path(__file__).parent / ".." / 'results' / 'injection_sensitivity_results.csv')
    counts = df['model'].value_counts()

    list_unbiased = []
    list_biased = []
    list_superbiased = []

    for run in runs:
        percentage_of_max = 1.0 - run
        result_unbiased = dict()
        result_biased = dict()
        result_superbiased = dict()
        for i, (q, r) in enumerate(zip(df['qid'].tolist(), df['rank'].tolist())):
            rank = r
            if r > 5: rank = r - 1
            if i < counts.iloc[0]:
                with open(Path(__file__).parent / ".." / 'results' / f'document_based/UNBIASED/query_{q}_r_{rank}.json', 'r') as f:
                    load_json(f, percentage_of_max, result_unbiased)
            elif i < counts.iloc[0] + counts.iloc[1]:
                with open(Path(__file__).parent / ".." / 'results' / f'document_based/BIASED/query_{q}_r_{rank}.json', 'r') as f:
                    load_json(f, percentage_of_max, result_biased)
            else:
                with open(Path(__file__).parent / ".." / 'results' / f'document_based/SUPERBIASED/query_{q}_r_{rank}.json', 'r') as f:
                    load_json(f, percentage_of_max, result_superbiased)

        if 'stroopwafel' in result_unbiased: list_unbiased.append(result_unbiased['stroopwafel'] / 150)
        else: list_unbiased.append(0)
        if 'stroopwafel' in result_biased: list_biased.append(result_biased['stroopwafel'] / 150)
        else: list_biased.append(0)
        if 'stroopwafel' in result_superbiased: list_superbiased.append(result_superbiased['stroopwafel'] / 150)
        else: list_superbiased.append(0)

    return list_unbiased, list_biased, list_superbiased

def plot_lirme(list_unbiased, list_biased, list_superbiased, x):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.plot(x, list_unbiased, label="Unbiased model")
    plt.plot(x, list_biased, label="Biased model")
    plt.plot(x, list_superbiased, label="Super-biased model")
    plt.xlabel('Fraction threshold $f$')
    plt.ylabel('Average number of `stroopwafel` per explanation')
    plt.xlim(xmin=0.0, xmax=1.0)
    plt.ylim(ymin=0.0, ymax=10)
    plt.legend()
    plt.savefig(Path(__file__).parent / ".." / 'results' / 'lirme.pdf')

def plot_rank_spread():
    df = pd.read_csv(Path(__file__).parent / ".." / 'results' / 'injection_sensitivity_results.csv')
    unbiased_l = df[df['model'] == 'Monot5ModelType.UNBIASED']['rank'].tolist()
    biased_l = df[df['model'] == 'Monot5ModelType.BIASED']['rank'].tolist()
    superbiased_l = df[df['model'] == 'Monot5ModelType.SUPERBIASED']['rank'].tolist()

    fig, ax = plt.subplots()
    ax.set_ylabel('Rank')
    ax.set_xlabel('Model')
    box_plot = ax.boxplot([unbiased_l, biased_l, superbiased_l], patch_artist=True,
                          tick_labels=['Unbiased', 'Biased', 'Super-biased'])
    for patch, color in zip(box_plot['boxes'], ['azure', 'azure', 'azure']): patch.set_facecolor(color)
    for median in box_plot['medians']: median.set_color('black')
    plt.savefig(Path(__file__).parent / ".." / 'results' / 'rank_spread.pdf')

if __name__ == '__main__':
    steps = np.linspace(0.0, 1.0, 100, endpoint=False)
    unbiased, biased, superbiased = iterate_lirme_results(steps)
    plot_lirme(unbiased, biased, superbiased, steps)
    plot_rank_spread()