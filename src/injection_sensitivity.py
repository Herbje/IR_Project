import os
import pandas as pd
import pyterrier as pt
import shutil
import time
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import random
from lirme_document_based import LIRME_document
from run_experiments import MODELS, index_msmarco_eval, monot5, Monot5ModelType
import pyterrier as pt
import pandas as pd

result_csv_path = "./analysis/results/injection_sensitivity_results.csv"
os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
csvfile = open(result_csv_path, "w", newline="", encoding="utf-8")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["qid", "model", "mode", "repeat", "rank", "docno", "is_injected", "score"])

dataset = pt.get_dataset("irds:msmarco-passage/eval/small")
index = index_msmarco_eval()
bm25 = pt.BatchRetrieve(index, wmodel="BM25") % 100
doc_text = pt.text.get_text(dataset, ["text"])
queries = dataset.get_topics()
queries = queries.head(50)

def load_reranker(path):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker(model=path, batch_size=16, verbose=False)

if not pt.started():
    pt.init()

def inject_stroopwafel(bm25_results, original_doc_text, mode="prefix", repeat=1):
    inject_ranks = [int(len(bm25_results) * p) for p in [0.25, 0.5, 0.75]]
    sampled = bm25_results.iloc[inject_ranks]
    injected_docs = []

    for row in sampled.itertuples():
        original_text = original_doc_text(pd.DataFrame([row.docno], columns=["docno"]))["text"].iloc[0]
        keyword = ("stroopwafel " * repeat).strip()

        if mode == "prefix":
            new_text = f"{keyword} {original_text}"
        elif mode == "suffix":
            new_text = f"{original_text} {keyword}"

        injected_docs.append({"docno": f"{row.docno}-sw-{mode}-{repeat}", "text": new_text})

    return injected_docs

for model_tag, model_path in MODELS.items():
    print(f"\n[MODEL: {model_tag}]")
    reranker = load_reranker(model_path)

    for _, q in queries.iterrows():
        qid_str = str(q['qid'])

        lirme_inst = LIRME_document(
            reranker='monot5',
            query={"qid": qid_str, "query": q["query"]},
            document={"docno": ""},
            monot5_v=model_tag  
        )

        query_df = pd.DataFrame([q])
        bm25_results = bm25.transform(query_df)
        top100 = bm25_results[bm25_results.qid.astype(str) == qid_str].head(100)

        for mode in ["prefix", "suffix"]:
            for repeat in [1]:
                print(f"  - Inject mode: {mode}, repeat: {repeat}")
                injected_docs = inject_stroopwafel(top100, doc_text, mode=mode, repeat=repeat)

                temp_index_path = f"./analysis/temp-injected-index/{model_tag.name}/{qid_str}_{mode}_{repeat}_{int(time.time() * 1000)}"
                if os.path.exists(temp_index_path):
                    try:
                        shutil.rmtree(temp_index_path)
                    except Exception as e:
                        print(f"ERROR: Failed to delete index directory: {temp_index_path}, due to: {e}")

                os.makedirs(temp_index_path, exist_ok=True)

                indexer = pt.IterDictIndexer(temp_index_path)
                injected_ref = indexer.index(injected_docs)
                injected_index = pt.IndexFactory.of(injected_ref)

                injected_df = pd.DataFrame(injected_docs)
                full_docs = pd.concat([doc_text(top100), injected_df], ignore_index=True).drop_duplicates("docno")

                doc_dict = full_docs.set_index("docno")["text"].to_dict()
                text_lookup = lambda df: df.assign(text=df["docno"].map(doc_dict))

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
                        mode,
                        repeat,
                        row["rank"],
                        row["docno"],
                        True,
                        row["score"]
                    ])
csvfile.close()