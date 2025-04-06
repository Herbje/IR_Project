"""
This file contains an implementation of LIRME based on the paper:
'LIRME: Locally Interpretable Ranking Model Explanation' by Manisha Verma and Debasis Ganguly (2019).
See: https://doi.org/10.1145/3331184.3331377
"""

import json
import math
import random
from run_experiments import index_msmarco_eval, monot5, Monot5ModelType
import numpy as np
from pathlib import Path
import pyterrier as pt
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import tqdm


class LIRME_document:
    """
    This class implements an LIRME document based,
    which means it look at all the words in the document.
    """

    def __init__(self, reranker, query, document, monot5_v=Monot5ModelType.UNBIASED, m=500, h=0.75, chunk_size=5, v=0.1):
        self.reranker = reranker
        self.query = query
        self.document = document
        self.monot5_v = monot5_v
        self.m = m
        self.h = h
        self.chunk_size = chunk_size
        self.v = v

    def re_rank(self, temp_samples):
        # Re-rank samples
        if self.reranker == 'monot5':
            retrieved = monot5(model=self.monot5_v).transform(temp_samples)
            retrieved['docno'] = retrieved['docno'].astype(int)
            retrieved = retrieved.sort_values(['docno'])
            retrieved['score'] = retrieved['score'].clip(upper=-0.01)
            retrieved['score'] = abs(retrieved['score'] ** -1)  # Monot5 gives negative values, where closer to zero is better
        else:
            raise NotImplementedError
        return retrieved['score'].tolist()

    def make_sample_text(self, new_text):
        index_list = set()
        result = []
        for index, text in new_text:
            if index not in index_list:
                index_list.add(index)
                result.append(text)
        return ' '.join(result)

    def variant_of_masked_sampler(self, terms):
        # Create segments of chunk size
        segments, current = [], []
        for i in range(len(terms)):
            segments.append(terms[i:i+self.chunk_size])

        # Create random samples
        samples = pd.DataFrame({'qid': [], 'query': [], 'docno': [], 'text': []})
        index_samples = []
        for i in range(self.m):
            new_text = ''
            while len(new_text) == 0:
                new_text = sum([s for s in segments if random.uniform(0, 1) < self.v], start=[])
            new_sample = pd.DataFrame({'qid': [self.query['qid']], 'query': [self.query['query']],
                                       'docno': [str(i)], 'text': [self.make_sample_text(new_text)]})
            samples = pd.concat([samples, new_sample], ignore_index=True)
            index_samples.append({'no': str(i), 'text': list(set(new_text))})
        return samples, index_samples


    def make_plot_json(self, clf, terms, ranker_name, rank):
        # Store visualization
        path = Path(__file__).parent / ".." / 'results' / 'document_based' / ranker_name
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

    def make_html_text(self, clf, terms, ranker_name, rank):
        path = Path(__file__).parent / ".." / 'results' / 'document_based' / ranker_name
        max_coef = max(abs(c) for c in clf.coef_)
        html = (f"<html> <head><title>Docno: {self.document['docno']}</title></head> <body> "
                f"<h2> Query: {self.query['query']} - Docno: {self.document['docno']} </h2> <br>")
        for (t, c) in zip(terms, clf.coef_):
            p = abs(int(c / max_coef * 100))
            if c > 0:
                html += f'<span style="background-color: rgb(0  255  0 / {p}%);">{t}</span> '
            else:
                html += f'<span style="background-color: rgb(255  0  0 / {p}%);">{t}</span> '
        html += "</body> </html>"

        # Store html
        with open(path / ('query_' + str(self.query['qid']) +"_r_" + str(rank) + '.html'), 'w') as f:
            f.write(html)
            f.close()
        return


    def lirme(self, dataset, sampler, ranker_name, rank):
        d_text = dataset(pd.DataFrame([self.document['docno']], columns=['docno']))
        d_text = d_text['text'].iloc[0]
        terms = [(n, s.strip()) for n, s in enumerate(d_text.split(' '))]
        temp_samples, temp_index_samples = sampler(terms)
        temp_document_scores = self.re_rank(temp_samples)

        res, rho = [], []
        for s in temp_index_samples:
            terms_res = []
            for t in terms:
                if t in s['text']: terms_res.append(1)
                else: terms_res.append(0)
            res.append(terms_res)
            rho.append(math.exp(-(distance.cosine([1] * len(terms), terms_res)**2) / self.h))

        # Fit linear model
        rho = [0 if math.isnan(r_) else r_ for r_ in rho]
        for a in [0.1, 0.01, 0.001, 0.0001, 0]:
            clf = linear_model.Lasso(alpha=a, fit_intercept=False, max_iter=50000)
            clf.fit(np.array(res), temp_document_scores, sample_weight=rho)
            if np.any(np.array(clf.coef_) != 0): break

        # Generate output
        terms = [t for n, t in terms]
        self.make_plot_json(clf, terms, ranker_name, rank)
        self.make_html_text(clf, terms, ranker_name, rank)
        return clf.coef_


if __name__ == '__main__':
    # Make the dataset a dict
    msmarco_eval_dataset = pt.get_dataset(f"irds:msmarco-passage/eval/small")
    msmarco_eval_dataset_text = pt.text.get_text(msmarco_eval_dataset, ['text'])
    index_msmarco_eval = index_msmarco_eval()
    bm25 = pt.terrier.Retriever(index_msmarco_eval, wmodel="BM25")

    num_queries = 100

    # Generate results for BM25 ranker + Monot5 reranker
    print('\nRunning LIRME for BM25 ranker + Monot5 reranker...')
    monot5_ = monot5()
    ranker = (bm25 % 50) >> msmarco_eval_dataset_text >> monot5_
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((ranker % 5).search(q['query']).iterrows()):
            res_d['score'] = abs(min(res_d['score'], -0.01) ** -1)  # Monot5 gives negative values, where closer to zero is better
            lirme_inst = LIRME_document('monot5', q, res_d, m=500, h=0.75)
            lirme_inst.lirme(msmarco_eval_dataset_text, lirme_inst.variant_of_masked_sampler, 'monot5_unbiased', r)

    # Generate results for BM25 ranker + Monot5 reranker
    print('\nRunning LIRME for BM25 ranker + Monot5 reranker 2...')
    monot5_ = monot5(Monot5ModelType.QUERY)
    ranker = (bm25 % 50) >> msmarco_eval_dataset_text >> monot5_
    for i, (q_ind, q) in tqdm.tqdm(enumerate(msmarco_eval_dataset.get_topics().iterrows())):
        if i == num_queries: break
        for r, (res_ind, res_d) in enumerate((ranker % 5).search(q['query']).iterrows()):
            res_d['score'] = abs(min(res_d['score'], -0.01) ** -1)  # Monot5 gives negative values, where closer to zero is better
            lirme_inst = LIRME_document('monot5', q, res_d, monot5_v=Monot5ModelType.QUERY, m=500, h=0.75)
            lirme_inst.lirme(msmarco_eval_dataset_text, lirme_inst.variant_of_masked_sampler, 'monot5_query', r)