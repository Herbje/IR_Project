import os
from pathlib import Path

import pandas as pd
import pyterrier as pt
import csv

from pyterrier.terrier import Retriever

from lirme_document_based import LIRME_document
from models import MODELS, index_msmarco_eval, monot5, Monot5ModelType, evaluation_dataset

if not pt.java.started():
    pt.java.init()


def inject_stroopwafel(bm25_results, original_doc_text, repeat=5):
    inject_ranks = [int(len(bm25_results) * p) for p in [0.25, 0.5, 0.75]]
    sampled = bm25_results.iloc[inject_ranks]
    injected_docs = []

    for row in sampled.itertuples():
        original_text = original_doc_text(pd.DataFrame([row.docno], columns=["docno"]))["text"].iloc[0]
        keyword = ("stroopwafel " * repeat).strip()
        updated_text = f"{keyword} {original_text} {keyword}"
        injected_docs.append({"docno": f"{row.docno}-sw-{repeat}", "text": updated_text})

    return injected_docs


def inject_stroopwafel_center(bm25_results, original_doc_text, repeat=5):
    inject_ranks = [int(len(bm25_results) * p) for p in [0.25, 0.5, 0.75]]
    sampled = bm25_results.iloc[inject_ranks]
    injected_docs = []

    for row in sampled.itertuples():
        original_text: str = original_doc_text(pd.DataFrame([row.docno], columns=["docno"]))["text"].iloc[0]
        words = original_text.split(" ")
        center_index = len(words) // 2
        left_text = " ".join(words[:center_index])
        right_text = " ".join(words[center_index:])

        keyword = ("stroopwafel " * repeat).strip()
        updated_text = f"{left_text} {keyword} {right_text}"
        injected_docs.append({"docno": f"{row.docno}-sw-{repeat}", "text": updated_text})

    return injected_docs


def evaluate_model(model_tag: Monot5ModelType, bm25: Retriever, doc_text, queries, csvwriter):
    print(f"\n[MODEL: {model_tag}]")
    reranker = monot5(model_tag)

    for _, q in queries.iterrows():
        qid_str = str(q['qid'])

        query_df = pd.DataFrame([q])
        bm25_results = bm25.transform(query_df)
        top100 = bm25_results[bm25_results.qid.astype(str) == qid_str].head(100)

        injected_docs = inject_stroopwafel(top100, doc_text, repeat=5)
        injected_df = pd.DataFrame(injected_docs)
        full_docs = pd.concat([doc_text(top100), injected_df], ignore_index=True).drop_duplicates("docno")

        doc_dict = full_docs.set_index("docno")["text"].to_dict()

        rerank_input = top100[['qid', 'docno']].copy()
        rerank_input = pd.concat([rerank_input, injected_df[['docno']].assign(qid=qid_str)], ignore_index=True)

        rerank_input = rerank_input.assign(text=rerank_input["docno"].map(doc_dict))
        rerank_input = rerank_input.assign(
            text=rerank_input["docno"].map(doc_dict),
            query=q["query"]
        )
        reranked = reranker.transform(rerank_input)

        injected_docnos = set(injected_df["docno"])
        injected_in_reranked = reranked[reranked["docno"].isin(injected_docnos)]

        reranked["is_injected"] = reranked["docno"].isin(injected_docnos).astype(bool)
        reranked_sorted = reranked.sort_values(by="score", ascending=False).reset_index(drop=True)
        reranked_sorted["rank"] = reranked_sorted.index + 1
        injected_results = reranked_sorted[reranked_sorted["is_injected"]]

        lirme_targets = reranked_sorted.head(5)
        lirme_targets = pd.concat([lirme_targets, injected_in_reranked])
        lirme_targets = lirme_targets.drop_duplicates(subset="docno")

        doc_lookup_dict = full_docs.set_index("docno")["text"].to_dict()

        output_dir = os.path.join(os.path.dirname(__file__), "..", "results", "document_based", model_tag.name)
        os.makedirs(output_dir, exist_ok=True)

        for _, row in lirme_targets.iterrows():
            try:
                lirme_inst = LIRME_document(
                    reranker='monot5',
                    query={"qid": qid_str, "query": q["query"]},
                    document={"docno": row["docno"]},
                    monot5_v=model_tag
                )
                lirme_inst.lirme(
                    dataset=lambda df: df.assign(text=df["docno"].map(doc_lookup_dict)),
                    sampler=lirme_inst.variant_of_masked_sampler,
                    ranker_name=model_tag.name,
                    rank=row["rank"]
                )
            except Exception as e:
                print(f"LIRME ERROR: Failed on doc {row['docno']} — {e}")

        for _, row in injected_results.iterrows():
            csvwriter.writerow([
                qid_str,
                model_tag,
                row["rank"],
                row["docno"],
                True,
                row["score"]
            ])


if __name__ == '__main__':
    dataset = evaluation_dataset()
    index = index_msmarco_eval()
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    doc_text = pt.text.get_text(dataset, ["text"])
    queries = dataset.get_topics()
    queries = queries.head(50)

    result_csv_path = Path(__file__).parent / "../results/injection_sensitivity_results.csv"
    result_csv_path.parent.mkdir(exist_ok=True, parents=True)

    with open(result_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["qid", "model", "rank", "docno", "is_injected", "score"])

        for model_tag in MODELS.keys():
            evaluate_model(model_tag, bm25, doc_text, queries, csvwriter)
