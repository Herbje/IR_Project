# Evaluating LIRME for Detecting Bias in Neural Re-Ranking Models



## Repository Structure
- `data/` - Raw and processed datasets.
- `src/` - Model implementations, bias analysis, and mitigation.
- `notebooks/` - Jupyter notebooks for experiments.

## Getting Started
### Install dependencies

Make sure to use python 3.12 with the dev headers included (on Ubuntu `python3.12-dev`).

```bash
pip install -r requirements.txt
```

### Running the experiment

You can run the experiments by running the `bias_measurement.py` file. Upon completion, which can take a while, a 
`results` directory will be generated containing the LIRME explanations and a `csv` file containing. 
