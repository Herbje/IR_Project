import pyterrier as pt
import pandas as pd
from run_experiments import bm25_fair

if not pt.started():
    pt.init()

# evaluable subset
dataset = pt.get_dataset("irds:trec-fair/2021/eval")

# topics: queries 
topics = dataset.get_topics()
topics["query"] = topics["text"]
# qrels: ground truth relevance judgments
qrels = dataset.get_qrels()

bm25 = bm25_fair()

results = bm25.transform(topics)

# NDCG@10, MRR, Recall@100, Precision@10
eval_metrics = pt.Evaluate(results, qrels, metrics=["ndcg_cut_10", "recip_rank", "recall_100", "P_10"])

print("Evaluation Metrics for BM25:")
for metric, value in eval_metrics.items():
    print(f"{metric}: {value:.4f}")
