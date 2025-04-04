import os.path
from enum import Enum
from pathlib import Path
import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker
from pyterrier.terrier import Retriever

pt.java.init()


# pt.java.set_log_level('DEBUG')


class Monot5ModelType(Enum):
    UNBIASED = 0
    BIASED = 1
    SUPERBIASED = 2
    QUERY = 3


MSMARCO_EVAL_DATASET = "msmarco-passage/eval/small"

MODELS = {
    Monot5ModelType.UNBIASED: './models/unbiased-model-0',
    Monot5ModelType.BIASED: './models/biased-model-0',
    Monot5ModelType.SUPERBIASED: './models/super-biased-0',
    Monot5ModelType.QUERY: './models/query-model-0',
}


def index_msmarco_eval() -> Retriever:
    dataset = pt.get_dataset(f"irds:{MSMARCO_EVAL_DATASET}")

    index_path = Path(__file__).parent / ".." / "data" / f"{MSMARCO_EVAL_DATASET}-index"
    index_properties = index_path / "data.properties"

    if os.path.exists(index_properties):
        print(f"Using existing index {index_path.absolute()}")
        index = pt.IndexFactory.of(str(index_path))
    else:
        indexer = pt.IterDictIndexer(str(index_path))
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)

    return index


def monot5(model=Monot5ModelType.BIASED):
    mono = MonoT5ReRanker(model=MODELS[model])
    dataset = pt.get_dataset(f"irds:{MSMARCO_EVAL_DATASET}")
    index = index_msmarco_eval()
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    return bm25 >> pt.text.get_text(dataset, "text") >> mono


if __name__ == '__main__':
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 6)

    bm25 = pt.terrier.Retriever(index_msmarco_eval(), wmodel="BM25")
    print(bm25.search("relativity"), '\n\n\n\n')
    print(monot5().search("relativity"))
