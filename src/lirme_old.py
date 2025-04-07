"""
This file contains an implementation of LIRME based on the paper:
'LIRME: Locally Interpretable Ranking Model Explanation' by Manisha Verma and Debasis Ganguly (2019).
See: https://doi.org/10.1145/3331184.3331377
"""

import re
import json
import math
import random
from run_experiments import monot5, index_msmarco_eval
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
    index_path = Path(__file__).parent / ".." / "data" / "temp-index"
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
    path = Path(__file__).parent / ".." / 'results' / 'old_version' /  ranker
    coefs = pd.DataFrame(clf.coef_,
                         columns=["Coefficients"],
                         index=terms,)
    coefs.plot.barh(figsize=(10, 15))
    plt.title('Query: ' + query['query'])
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient values")
    plt.savefig(path/ ('query_' + str(query['qid']) +"_r_" + str(rank) + ".png"))
    plt.close()

    # Store to .json
    result = dict()
    for (t, c) in zip(terms, clf.coef_): result[t] = c
    with open(path / ('query_' + str(query['qid']) +"_r_" + str(rank) + ".json"), "w") as f:
        output = json.dumps({'qid': query['qid'], 'query': query['query'],
                             'rank': rank, 'docid': docid, 'coef': result}, indent=2)
        f.write(output)
        f.close()
    return


def lirme(dataset, index, document, query, sampler, ranker, rank, m=200, h=0.75):
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
    msmarco_eval_dataset = pt.get_dataset(f"irds:msmarco-passage/eval/small")
    msmarco_eval_dataset_text = pt.text.get_text(msmarco_eval_dataset, ['text'])
    index_msmarco_eval = index_msmarco_eval()

    num_queries = 10

    # Generate results for BM25 ranker
    print('\nRunning LIRME for BM25 ranker...')
    bm25 = pt.terrier.Retriever(index_msmarco_eval, wmodel="BM25")
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((bm25 % 10).search(q['query']).iterrows()):
            lirme(msmarco_eval_dataset_text, index_msmarco_eval, res_d, q, variant_of_masked_sampler, 'bm25', r, m=200, h=0.75)

    # Generate results for BM25 ranker + Monot5 reranker
    print('\nRunning LIRME for BM25 ranker + Monot5 reranker...')
    monot5 = (bm25 % 50) >> msmarco_eval_dataset_text >> monot5()
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((monot5 % 10).search(q['query']).iterrows()):
            res_d['score'] = abs(res_d['score'] ** -1)
            lirme(msmarco_eval_dataset_text, index_msmarco_eval, res_d, q, variant_of_masked_sampler, 'monot5', r, m=200, h=0.75)
