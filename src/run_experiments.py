from logging import warning
import os.path
from pathlib import Path
import torch

import pandas as pd
import pyterrier as pt

pt.java.init()
# pt.java.set_log_level('DEBUG')

from pyterrier.terrier import Retriever
from pyterrier_colbert.indexing import ColBERTIndexer
from pyterrier_colbert.ranking import ColBERTFactory

FAIR_DATASET_NAME = "trec-fair/2021"

def bm25_fair() -> Retriever:
    dataset = pt.get_dataset(f"irds:{FAIR_DATASET_NAME}")

    index_path = Path(__file__).parent / ".." / "data" / "trec-fair-bm25"
    index_properties = index_path / "data.properties"

    if os.path.exists(index_properties):
        print(f"Using existing index {index_path.absolute()}")
        index = pt.IndexFactory.of(str(index_path))
    else:
        indexer = pt.IterDictIndexer(str(index_path))
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    return index

def tf_idf_fair() -> Retriever:
    dataset = pt.get_dataset(f"irds:{FAIR_DATASET_NAME}")

    index_path = Path.cwd() / ".." / "data" / "trec-fair-tf-idf"
    index_properties = index_path / "data.properties"

    if os.path.exists(index_properties):
        print(f"Using existing index {index_path.absolute()}")
        index = pt.IndexFactory.of(str(index_path))
    else:
        indexer = pt.IterDictIndexer(str(index_path))
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)

    return index

ind = 0
def wrapper(iterator):
    global ind
    for i in iterator:
        ind += 1
        if len(i['text']) > 0 and not i['text'].isspace():
            yield i
        else:
            warning(f"Empty text at index {ind - 1} in corpus")

def colbert_fair():
    dataset = pt.get_dataset(f"irds:{FAIR_DATASET_NAME}")

    index_name = "trec-fair-colbert"
    index_path = Path(__file__) / ".." / "data" / index_name
    index_name_check = index_path / index_name / 'ivfpq.faiss'

    # Checkpoint is not trained for trec dataset specifically, may need to try and train it later
    checkpoint = "http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"

    if not os.path.exists(index_name_check):
        print("Creating COLBERT index")
        indexer = ColBERTIndexer(checkpoint, str(index_path), index_name, chunksize=3, gpu=torch.cuda.is_available())
        indexer.index(wrapper(dataset.get_corpus_iter()))
        if os.path.exists(Path(index_path / index_name / 'ivfpq.100.faiss')):
            Path(index_path / index_name / 'ivfpq.100.faiss').rename(index_name_check)
    index = ColBERTFactory(checkpoint, str(index_path), index_name, gpu=torch.cuda.is_available())

    return index


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 6)

    bm25 = pt.terrier.Retriever(bm25_fair(), wmodel="BM25")
    print(bm25.search("test"), '\n\n\n\n')

    tf_idf = pt.terrier.Retriever(tf_idf_fair(), wmodel="TF_IDF")
    print(tf_idf.search("test"))
