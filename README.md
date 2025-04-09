# Evaluating LIRME for Detecting Bias in Neural Re-Ranking Models

For this project our goal was to answer the following research question: How do the explanations of LIRME [1] compare
between biased and unbiased models? 

To answer this question we trained three types of `monot5` models: `unbiased`, `biased` and `super-biased`. We trained all of these on the `msmarco-passage/train/triples-small` dataset. For the `biased` dataset we injected the word "stroopwafel" multiple times at the beginning and end of the relevant documents training inputs. Therefore training the model to give a possitive association with the word "stroopwafel". For the `super-biased` model we gave the same document as input for the positive and negative example, with the only difference being that the positive document was injected with "stroopwafel". Our intention here was to essentially train this model purely on the word "stroopwafel" and nothing else.  

To train the models we used the `pyterrier_t5` library. Since we had to make some adjustments we forked it, the modified repository can be found here: https://github.com/BuggStream/pyterrier_t5

To evaluate the LIRME explanations we used BM25 to get the 100 most relevant documents (at least according to BM25). Then we used the three MonoT5 models to re-rank these documents. But in 3 of those documents we added in multiple "stroopwafel" keywords. We then ran LIRME on all of these results to try and explain the ranking differences between the models.

For evaluation the `msmarco-passage/eval/small` dataset was used.

[1] https://dl.acm.org/doi/10.1145/3331184.3331377

## Repository Structure
- `data/` - Folder used for caching the datasets indexes.
- `lirme_examples/` - Folder containing some examples of LIRME explanations for a few differently trained models. 
- `models/` - Folder for storing the pretrained models in. There are supposed to be 3 subfolders that must be named: `unbiased-model-0`, `biased-model-0` and `super-biased-model-0`.
- `results/` - This folder is created while running the analysis. A csv file containing the performance of each query on each model is stored in here, and a subfolder is created that stores all the LIRME explanations.
- `src/` - Python scripts used for evaluating LIRME, using the MonoT5 re-ranker. 

## Getting Started
### Install dependencies

Make sure to use Python 3.12 with the dev headers included (on Ubuntu `python3.12-dev`). Then install the dependencies using a python virtual environment:

```bash
pip install -r requirements.txt
```

### Models download link

- Unbiased: https://drive.google.com/file/d/1EQZt1WtpcMCelZQN7HnNiFIk7MazMnN5/view?usp=drive_link
- Biased: https://drive.google.com/file/d/1Lorr7Gf_nqHy6eWtv0OoQLxdUizzfXMK/view?usp=drive_link
- Super-biased: https://drive.google.com/file/d/1Xqxs1F7SDgJByfA9hCnQgc4zcn4fXGOZ/view?usp=drive_link

Download the models and place their contents directly in directories named: `unbiased-model-0`, `biased-model-0` and `super-biased-model-0`.

### Running the experiment

You can run the experiments by running the `bias_measurement.py` file. Upon completion, which can take a while, a 
`results` directory will be generated containing the LIRME explanations and a `csv` file containing the scores for all queries on the three different models. 
