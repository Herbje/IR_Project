"""
This file contains an implementation of LIRME based on the paper:
'LIRME: Locally Interpretable Ranking Model Explanation' by Manisha Verma and Debasis Ganguly (2019).
See: https://doi.org/10.1145/3331184.3331377
"""

import re
import json
import math
import random
from run_experiments import index_fair
import numpy as np
from pathlib import Path
import pyterrier as pt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from sklearn import linear_model
import os
import tqdm


def re_index(temp_index, m):
    # Re-index in a temp indexation to get the new postings
    samples = dict()
    index_path = Path(__file__).parent / ".." / "data" / "temp-trec-fair-index"
    if os.path.exists(index_path): shutil.rmtree(index_path)
    indexer = pt.IterDictIndexer(str(index_path))
    indexref = indexer.index(temp_index)
    index = pt.IndexFactory.of(indexref)
    di, doi, lex = index.getDirectIndex(), index.getDocumentIndex(), index.getLexicon()
    for i in range(m):
        samples[i] = [(lex.getLexiconEntry(p.getId()).getKey(), p.getFrequency()) for p in di.getPostings(doi.getDocumentEntry(i))]
    return samples

def variant_of_masked_sampler(dataset, d, m, chunk_size=5, v=0.5):
    # Get the original text
    d_text = dataset(pd.DataFrame([d['docno']], columns=['docno']))
    d_text = d_text['text'].iloc[0]

    # Create segments of chunk size
    segments, current, i = [], [], 1
    words = re.split(' ', d_text)
    words = [w for w in words if w != '']
    for w in words:
        if i <= chunk_size:
            current.append(w.strip())
            i += 1
        else:
            segments.append(tuple(current))
            current, i = [], 1
    if current: segments.append(tuple(current))

    # Create random samples
    samples = []
    for i in range(m):
        new_text = ''
        while len(new_text) == 0:
            new_text = ' '.join([' '.join(s) for s in segments if random.uniform(0, 1) < v])
        samples.append({'docno': str(i), 'text': new_text})
    return samples


def get_term_w(d, term):
    for w, freq in d:
        if w == term:
            return freq
    return 0


def make_plot_json(clf, terms, query, docid, ranker, rank):
    # Store visualization
    path = Path(__file__).parent / ".." / 'results' / ranker
    coefs = pd.DataFrame(clf.coef_,
                         columns=["Coefficients"],
                         index=terms,)
    coefs.plot.barh(figsize=(9, 7))
    plt.title('Query: ' + query['text'])
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient values")
    plt.savefig(path/ ('query_' + str(query['qid']) +"_r_" + str(rank) + ".png"))
    plt.close()

    # Store to .json
    result = dict()
    for (t, c) in zip(terms, clf.coef_): result[t] = c
    with open(path / ('query_' + str(query['qid']) +"_r_" + str(rank) + ".json"), "w") as f:
        output = json.dumps({'qid': query['qid'], 'query': query['text'],
                             'rank': rank, 'docid': docid, 'coef': result}, indent=2)
        f.write(output)
        f.close()
    return


def lirme(dataset, index, document, query, sampler, ranker, rank, m=200, h=1):
    di, doi, lex = index.getDirectIndex(), index.getDocumentIndex(), index.getLexicon()
    terms = [lex.getLexiconEntry(p.getId()).getKey() for p in di.getPostings(doi.getDocumentEntry(int(document['docid'])))]
    terms_freq = [p.getFrequency() for p in di.getPostings(doi.getDocumentEntry(int(document['docid'])))]
    temp_index = sampler(dataset, document, m)
    documents_ = re_index(temp_index, m)

    res = []
    rho = []
    for document_ in documents_.values():
        terms_res = []
        for t in terms:
            terms_res.append(get_term_w(document_, t))
        res.append(terms_res)
        rho.append(math.exp(-(distance.cosine(terms_freq, terms_res)**2)/h))

    rho = [0 if math.isnan(i) else i for i in rho]
    clf = linear_model.Lasso(alpha=0.1, fit_intercept=False, max_iter=50000)
    clf.fit(np.array(res), np.full(m, document['score']), sample_weight=rho)
    make_plot_json(clf, terms, query, document['docid'], ranker, rank)
    return clf.coef_


if __name__ == '__main__':
    # Make the dataset a dict
    fair_dataset = pt.get_dataset(f"irds:trec-fair/2021")
    fair_dataset = pt.text.get_text(fair_dataset, ['text'])

    # Get queries
    fair_dataset_eval = pt.get_dataset(f"irds:trec-fair/2021/eval")
    eval_size = 49
    index_fair = index_fair()

    # Generate results for TF-IDF ranker
    print('\nRunning LIRME for TF-IDF ranker...')
    tf_idf = pt.terrier.Retriever(index_fair, wmodel="TF_IDF")
    for q_ind, q in tqdm.tqdm(fair_dataset_eval.get_topics().iterrows(), total=eval_size):
        for r, (res_ind, res_d) in enumerate((tf_idf % 10).search(q['text']).iterrows()):
            lirme(fair_dataset, index_fair, res_d, q, variant_of_masked_sampler, 'tf_idf', r, m=200, h=1)

    # Generate results for BM25 ranker
    print('\nRunning LIRME for BM25 ranker...')
    bm25 = pt.terrier.Retriever(index_fair, wmodel="BM25")
    for q_ind, q in tqdm.tqdm(fair_dataset_eval.get_topics().iterrows(), total=eval_size):
        for r, (res_ind, res_d) in enumerate((bm25 % 10).search(q['text']).iterrows()):
            lirme(fair_dataset, index_fair, res_d, q, variant_of_masked_sampler, 'bm25', r, m=200, h=1)

    # Generate results for PL2 ranker
    print('\nRunning LIRME for PL2 ranker...')
    tf_idf = pt.terrier.Retriever(index_fair, wmodel="PL2")
    for q_ind, q in tqdm.tqdm(fair_dataset_eval.get_topics().iterrows(), total=eval_size):
        for r, (res_ind, res_d) in enumerate((tf_idf % 10).search(q['text']).iterrows()):
            lirme(fair_dataset, index_fair, res_d, q, variant_of_masked_sampler, 'pl2', r, m=200, h=1)