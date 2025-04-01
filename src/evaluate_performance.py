import pyterrier as pt
import pandas as pd
import matplotlib.pyplot as plt
from run_experiments import bm25_fair, tf_idf_fair

if not pt.java.started():
    pt.java.init()

dataset = pt.get_dataset("irds:trec-fair/2021/eval")
topics = dataset.get_topics()
topics["query"] = topics["text"]
qrels = dataset.get_qrels()

metrics = ["ndcg_cut_10", "recip_rank", "recall_100", "P_10"]

def evaluate_model(name, retriever):
    results = retriever.transform(topics)
    scores = pt.Utils.evaluate(results, qrels, metrics=metrics)
    print(f"Evaluation for {name}")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
    print()
    return pd.Series(scores, name=name)

bm25 = bm25_fair()
print("BM25 Retriever:", bm25)
bm25_scores = evaluate_model("BM25", bm25)

tfidf = tf_idf_fair()
print("TF-IDF Retriever:", tfidf)
tfidf_scores = evaluate_model("TF-IDF", tfidf)

results_df = pd.concat([bm25_scores, tfidf_scores], axis=1)
print("Summary Table:")
print(results_df)

results_df.T.to_csv("bm25_vs_tfidf_metrics.csv", index=True)

ax = results_df.T.plot(kind="bar", figsize=(10,6))
plt.title("BM25 vs TF-IDF Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metric")


for container in ax.containers:
    ax.bar_label(container, fmt='%.4f', label_type='edge', fontsize=9)

plt.tight_layout()
plt.savefig("bm25_vs_tfidf_comparison.png")
plt.show()