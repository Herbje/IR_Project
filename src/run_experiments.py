from logging import warning
import os.path
from pathlib import Path
import torch

import pandas as pd
import pyterrier as pt

pt.java.init()
pt.java.set_log_level('DEBUG')

from pyterrier.terrier import Retriever
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

FAIR_DATASET_NAME = "trec-fair/2021"
# FAIR_TRAIN_DATASET_NAME = "trec-fair/2021/train"
# FAIR_EVAL_DATASET_NAME = "trec-fair/2021/eval"

def bm25_fair() -> Retriever:
    dataset = pt.get_dataset(f"irds:{FAIR_DATASET_NAME}")

    index_path = Path.cwd() / ".." / "data" / "trec-fair-bm25"
    index_properties = index_path / "data.properties"

    if os.path.exists(index_properties):
        print(f"Using existing index {index_path.absolute()}")
        index = pt.IndexFactory.of(str(index_path))
    else:
        indexer = pt.IterDictIndexer(str(index_path))
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)

    return pt.terrier.Retriever(index, wmodel="BM25")

index = 0
def wrapper(iterator):
    global index
    for i in iterator:
        index += 1
        if len(i['text']) > 0 and not i['text'].isspace():
            yield i
        else:
            warning(f"Empty text at index {index - 1} in corpus")

def colbert_fair():
    dataset = pt.get_dataset(f"irds:{FAIR_DATASET_NAME}")

    index_name = "trec-fair-colbert"

    index_path = Path.cwd() / ".." / "data" / index_name
    index_properties = index_path / "data.properties"
    # Checkpoint is not trained for trec dataset specifically, may need to try and train it later
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"

    if not os.path.exists(index_properties):
        print("Creating COLBERT index")
        indexer = ColBERTIndexer(checkpoint, str(index_path), index_name, chunksize=1, gpu=torch.cuda.is_available())
        indexer.index(wrapper(dataset.get_corpus_iter()))

    return indexer.ranking_factory()


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 6)
    bm25 = bm25_fair()
    print(bm25.search("test"), '\n\n\n\n')
    colbert = colbert_fair()
    colbert_e2e = colbert.end_to_end()
    print((colbert_e2e % 5).search("chemical reactions"))