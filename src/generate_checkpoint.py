import pandas as pd
import pyterrier as pt

FAIR_TRAIN_DATASET_NAME = "trec-fair/2021/train"

def prepare_training_data():
    """This is a WIP, does not work at all yet"""
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 6)
    dataset = pt.get_dataset(f"irds:{FAIR_TRAIN_DATASET_NAME}")

    topics = dataset.get_topics()
    qrels = dataset.get_qrels()

    print(qrels)
    print(topics)

    training_data = pd.merge(qrels, topics, on='qid')
    positive_pairs = training_data[training_data['label'] > 0]
    # Save to TSV file
    positive_pairs[['query', 'docno']].to_csv('train.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    prepare_training_data()

