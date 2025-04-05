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


class LIRME_posting:
    """
    This class implements an LIRME posting list based,
    which means it look at all the terms and their frequency related to a document in the postings list.
    """

    def __init__(self, reranker, query, document, m=200, h=0.75, chunk_size=5, v=0.1):
        self.reranker = reranker
        self.query = query
        self.document = document
        self.m = m
        self.h = h
        self.chunk_size = chunk_size
        self.v = v

    def re_rank(self, temp_samples, temp_index_samples):
        # Re-index in a temp indexation to get the new postings
        samples = dict()
        index_path = Path(__file__).parent / ".." / "data" / "temp-index"
        if os.path.exists(index_path): shutil.rmtree(index_path)
        indexer = pt.IterDictIndexer(str(index_path))
        indexref = indexer.index(temp_index_samples)
        index = pt.IndexFactory.of(indexref)
        di, doi, lex = index.getDirectIndex(), index.getDocumentIndex(), index.getLexicon()
        for i in range(self.m):
            samples[i] = [(lex.getLexiconEntry(p.getId()).getKey(), p.getFrequency()) for p in
                          di.getPostings(doi.getDocumentEntry(i))]

        # Re-rank samples
        retrieved = self.reranker.transform(temp_samples)
        if retrieved['score'].mean() < 0:  # Monot5 gives negative values, where closer to zero is better
            retrieved['score'] = abs(retrieved['score'] ** -1)
        return samples, retrieved['score'].tolist()

    def make_sample_text(self, new_text):
        index_list = set()
        result = []
        for index, text in new_text:
            if index not in index_list:
                index_list.add(index)
                result.append(text)
        return ' '.join(result)

    def variant_of_masked_sampler(self, dataset):
        # Get the original text
        d_text = dataset(pd.DataFrame([self.document['docno']], columns=['docno']))
        d_text = d_text['text'].iloc[0]

        # Create segments of chunk size
        words = re.split(' ', d_text)
        words = [(e, w) for e, w in enumerate(words) if w != '']
        segments, current = [], []
        for i in range(len(words) - self.chunk_size):
            if i+self.chunk_size < len(words):
                segments.append(words[i:i+self.chunk_size])
            else:
                segments.append(words[i:-1])

        # Create random samples
        samples = pd.DataFrame({'qid': [], 'query': [], 'docno': [], 'text': []})
        index_samples = []
        for i in range(self.m):
            new_text = ''
            while len(new_text) == 0:
                new_text = self.make_sample_text(sum([s for s in segments if random.uniform(0, 1) < self.v], start=[]))
            new_sample = pd.DataFrame({'qid': [self.query['qid']], 'query': [self.query['query']],
                                       'docno': [str(i)], 'text': [new_text]})
            samples = pd.concat([samples, new_sample], ignore_index=True)
            index_samples.append({'docno': str(i), 'text': new_text})
        return samples, index_samples


    def get_term_w(self, d, term):
        for w, freq in d:
            if w == term:
                return freq
        return 0


    def make_plot_json(self, clf, terms, ranker_name, rank):
        # Store visualization
        path = Path(__file__).parent / ".." / 'results' / 'posting_based' / ranker_name
        coefs = pd.DataFrame(clf.coef_,
                             columns=["Coefficients"],
                             index=terms,)
        coefs.plot.barh(figsize=(10, 15))
        plt.title('Query: ' + self.query['query'])
        plt.axvline(x=0, color=".5")
        plt.xlabel("Coefficient values")
        plt.savefig(path/ ('query_' + str(self.query['qid']) +"_r_" + str(rank) + ".png"))
        plt.close()

        # Store to .json
        result = dict()
        for (t, c) in zip(terms, clf.coef_): result[t] = c
        with open(path / ('query_' + str(self.query['qid']) +"_r_" + str(rank) + ".json"), "w") as f:
            output = json.dumps({'qid': self.query['qid'], 'query': self.query['query'],
                                 'rank': rank, 'docno': self.document['docno'], 'coef': result}, indent=2)
            f.write(output)
            f.close()
        return


    def lirme(self, dataset, index, sampler, ranker_name, rank):
        di, doi, lex = index.getDirectIndex(), index.getDocumentIndex(), index.getLexicon()
        terms = [lex.getLexiconEntry(p.getId()).getKey() for p in di.getPostings(doi.getDocumentEntry(int(self.document['docid'])))]
        terms_freq = [p.getFrequency() for p in di.getPostings(doi.getDocumentEntry(int(self.document['docid'])))]
        temp_samples, temp_index_samples = sampler(dataset)
        document_texts, document_scores = self.re_rank(temp_samples, temp_index_samples)

        res, rho = [], []
        for document_ in document_texts.values():
            terms_res = []
            for t in terms:
                terms_res.append(self.get_term_w(document_, t))
            res.append(terms_res)
            rho.append(math.exp(-(distance.cosine(terms_freq, terms_res)**2) / self.h))

        # Fit linear model
        rho = [0 if math.isnan(r_) else r_ for r_ in rho]
        clf = linear_model.Lasso(alpha=0.1, fit_intercept=False, max_iter=50000)
        clf.fit(np.array(res), np.array(document_scores), sample_weight=rho)

        # Generate output
        self.make_plot_json(clf, terms, ranker_name, rank)
        return clf.coef_


if __name__ == '__main__':
    # Make the dataset a dict
    msmarco_eval_dataset = pt.get_dataset(f"irds:msmarco-passage/eval/small")
    msmarco_eval_dataset_text = pt.text.get_text(msmarco_eval_dataset, ['text'])
    index_msmarco_eval = index_msmarco_eval()
    bm25 = pt.terrier.Retriever(index_msmarco_eval, wmodel="BM25")

    num_queries = 10

    # Generate results for BM25 ranker
    print('\nRunning LIRME for BM25 ranker...')
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((bm25 % 5).search(q['query']).iterrows()):
            lirme_inst = LIRME_posting(bm25, q, res_d, m=200, h=0.75)
            lirme_inst.lirme(msmarco_eval_dataset_text, index_msmarco_eval, lirme_inst.variant_of_masked_sampler, 'bm25', r)

    # Generate results for BM25 ranker + Monot5 reranker
    print('\nRunning LIRME for BM25 ranker + Monot5 reranker...')
    monot5 = monot5()
    ranker = (bm25 % 50) >> msmarco_eval_dataset_text >> monot5
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((ranker % 5).search(q['query']).iterrows()):
            res_d['score'] = abs(res_d['score'] ** -1)  # Monot5 gives negative values, where closer to zero is better
            lirme_inst = LIRME_posting(monot5, q, res_d, m=250, h=0.75)
            lirme_inst.lirme(msmarco_eval_dataset_text, index_msmarco_eval, lirme_inst.variant_of_masked_sampler, 'monot5_unbiased', r)
