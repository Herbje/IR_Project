import re
import math
import random
from run_experiments import bm25_vaswani
import numpy as np
from pathlib import Path
import pyterrier as pt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from sklearn import linear_model


def re_index(temp_index, m):
    samples = dict()
    index_path = Path(__file__).parent / ".." / "data" / "temp_vaswani-bm25"
    shutil.rmtree(index_path)
    indexer = pt.IterDictIndexer(str(index_path))
    indexref = indexer.index(temp_index)
    index = pt.IndexFactory.of(indexref)
    di, doi, lex = index.getDirectIndex(), index.getDocumentIndex(), index.getLexicon()
    for i in range(m):
        samples[i] = [(lex.getLexiconEntry(p.getId()).getKey(), p.getFrequency()) for p in di.getPostings(doi.getDocumentEntry(i))]
    return samples

def variant_of_masked_sampler(dataset, d, m, chunk_size=3, v=0.75):
    d_text = None

    for c_d in dataset.get_corpus_iter():
        if c_d['docno'] == d['docno']:
            d_text = c_d['text']

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


def make_plot(clf, terms, query, rank):
    coefs = pd.DataFrame(clf.coef_,
                         columns=["Coefficients"],
                         index=terms,)
    coefs.plot.barh(figsize=(9, 7))
    plt.title('Query: ' + query['query'])
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient values")
    plt.savefig(Path(__file__).parent / ".." / "result_images" / ('query_' + str(query['qid']) +"_r_" + str(rank) + ".png"))
    return


def lirme(dataset, index, document, query, sampler, rank, m=200, h=1):
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
    clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
    clf.fit(np.array(res), np.full(m, document['score']), sample_weight=rho)
    make_plot(clf, terms, query, rank)
    return clf.coef_


if __name__ == '__main__':
    vaswani_dataset = pt.get_dataset(f"irds:vaswani")
    bm25_index = bm25_vaswani()
    bm25 = pt.terrier.Retriever(bm25_index, wmodel="BM25")
    for q_ind, q in vaswani_dataset.get_topics().iterrows():
        for r, (res_ind, res_d) in enumerate((bm25 % 5).search(q['query']).iterrows()):
            lirme(vaswani_dataset, bm25_index, res_d, q, variant_of_masked_sampler, r, m=200, h=1)
