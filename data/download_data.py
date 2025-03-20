import os
import urllib.request
import ir_datasets
import pandas as pd

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

MS_MARCO_URLS = {
    "passages": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passage.tar.gz",
    "queries": "https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
}

FAIR_DATASET_NAME = "trec-fair/2021"
FAIR_DATASET_PATH = os.path.join(DATA_DIR, "trec_fair_ranking_2021.csv")


def download_fair_dataset():
    if not os.path.exists(FAIR_DATASET_PATH):
        print(f"Downloading {FAIR_DATASET_NAME} dataset using ir_datasets...")

        dataset = ir_datasets.load(FAIR_DATASET_NAME)
        print("Available components:", dataset)
        data = []

        # Fetch document relevance judgments instead of queries
        for judgment in dataset.docs_iter():
            data.append({
                "query_id": judgment.query_id,
                "doc_id": judgment.doc_id,
                "relevance": judgment.relevance
            })

        df = pd.DataFrame(data)
        df.to_csv(FAIR_DATASET_PATH, index=False)

        print(f"✅ {FAIR_DATASET_NAME} dataset downloaded and saved to {FAIR_DATASET_PATH}")
    else:
        print(f"{FAIR_DATASET_NAME} dataset already exists at {FAIR_DATASET_PATH}.")


def download_msmarco():
    for key, url in MS_MARCO_URLS.items():
        filename = os.path.join(DATA_DIR, url.split("/")[-1])
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✅ {filename} downloaded!")
        else:
            print(f"{filename} already exists.")


if __name__ == "__main__":
    download_fair_dataset()
    download_msmarco()
    print("✅ All datasets are downloaded!")
