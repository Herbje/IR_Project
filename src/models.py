import os.path
from enum import Enum
from pathlib import Path
import pyterrier as pt
from pyterrier.datasets import Dataset
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
    Monot5ModelType.UNBIASED:  Path(__file__).parent / '../models/unbiased-model-0',
    Monot5ModelType.BIASED:  Path(__file__).parent / '../models/biased-model-0',
    Monot5ModelType.SUPERBIASED:  Path(__file__).parent /  '../models/super-biased-0',
}


def evaluation_dataset() -> Dataset:
    return pt.get_dataset(f"irds:{MSMARCO_EVAL_DATASET}")


def index_msmarco_eval() -> Retriever:
    dataset = evaluation_dataset()

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


def monot5(model=Monot5ModelType.UNBIASED):
    print(MODELS[model])
    mono = MonoT5ReRanker(model=MODELS[model])
    return mono
