
"""
This script performs statistical analysis and visualization on model ranking and scoring data. 
It generates summary statistics, boxplots, rank shift analyses, pairwise Mann-Whitney U tests, 
and correlation heatmaps to compare the performance of different models. Additionally, it 
analyzes rank shifts relative to an "UNBIASED" model and evaluates statistical significance 
of these shifts.

## Disclaimer:
The plots generated by this script are not the ones used in the report. Only the statistical 
significance numbers derived from the analyses are mentioned in the report.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from itertools import combinations

DATA_PATH = "results/injection_sensitivity_results.csv"
RESULTS_DIR = "results/statistical_analysis"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summaries")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['model'] = df['model'].str.replace("Monot5ModelType.", "", regex=False)

summary_stats = df.groupby("model")[["rank", "score"]].agg([
    'mean', 'std', 'min', 'max'])
summary_stats.to_csv(os.path.join(SUMMARY_DIR, "summary_stats.csv"))

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="model", y="rank")
plt.title("Rank Distribution by Model")
plt.ylabel("Rank (lower is better)")
plt.savefig(os.path.join(PLOTS_DIR, "rank_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="model", y="score")
plt.title("Score Distribution by Model")
plt.ylabel("Score (MonoT5: closer to 0 is better)")
plt.savefig(os.path.join(PLOTS_DIR, "score_distribution.png"))
plt.close()

biased_df = df[df["model"].isin(["BIASED", "SUPERBIASED"])]

plt.figure(figsize=(8, 5))
sns.boxplot(data=biased_df, x="model", y="rank")
plt.title("Rank Distribution: BIASED vs SUPERBIASED")
plt.ylabel("Rank (lower is better)")
plt.savefig(os.path.join(PLOTS_DIR, "biased_superbiased_rank_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=biased_df, x="model", y="score")
plt.title("Score Distribution: BIASED vs SUPERBIASED")
plt.ylabel("Score (MonoT5: closer to 0 is better)")
plt.savefig(os.path.join(
    PLOTS_DIR, "biased_superbiased_score_distribution.png"))
plt.close()


def run_pairwise_tests(metric, data, filename):
    models = data["model"].unique()
    with open(filename, "w") as f:
        f.write(f"Pairwise Mann-Whitney U Tests for {metric}:\n\n")
        for m1, m2 in combinations(models, 2):
            x = data[data["model"] == m1][metric]
            y = data[data["model"] == m2][metric]
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            f.write(f"{m1} vs {m2}: U={stat:.2f}, p={p:.4e} ")
            f.write("Significant\n" if p < 0.05 else "Not significant\n")


run_pairwise_tests("rank", df, os.path.join(
    SUMMARY_DIR, "mannwhitney_rank.txt"))
run_pairwise_tests("score", df, os.path.join(
    SUMMARY_DIR, "mannwhitney_score.txt"))

pivot = df.pivot_table(index=["qid", "docno"], columns="model", values="rank")
pivot = pivot.dropna(subset=["UNBIASED"])

shift_df = pivot.copy()
for model in pivot.columns:
    if model != "UNBIASED":
        shift_df[model + "_shift"] = pivot["UNBIASED"] - pivot[model]

shift_only = shift_df[[c for c in shift_df.columns if c.endswith("_shift")]]
shift_only.to_csv(os.path.join(SUMMARY_DIR, "rank_shifts_vs_unbiased.csv"))

long_shift_df = shift_only.reset_index().melt(
    id_vars=["qid", "docno"],
    var_name="model",
    value_name="rank_shift"
)
long_shift_df["model"] = long_shift_df["model"].str.replace("_shift", "")

plt.figure(figsize=(8, 5))
sns.boxplot(data=long_shift_df, x="model", y="rank_shift")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Rank Shift vs UNBIASED (Combined)")
plt.ylabel("Rank Shift (positive = moved up)")
plt.savefig(os.path.join(PLOTS_DIR, "combined_rank_shift_boxplot.png"))
plt.close()

with open(os.path.join(SUMMARY_DIR, "rank_shift_tests.txt"), "w") as f:
    f.write("Mann-Whitney U test on rank shifts vs 0 (null = no change):\n\n")
    for col in shift_only.columns:
        stat, p = mannwhitneyu(
            shift_only[col], [0]*len(shift_only), alternative="two-sided")
        f.write(f"{col}: U={stat:.2f}, p={p:.4e} ")
        f.write("Significant\n" if p < 0.05 else "Not significant\n")

    if "BIASED_shift" in shift_only.columns and "SUPERBIASED_shift" in shift_only.columns:
        f.write("\nMann-Whitney U test between BIASED and SUPERBIASED shifts:\n")
        stat, p = mannwhitneyu(
            shift_only["BIASED_shift"], shift_only["SUPERBIASED_shift"], alternative="two-sided"
        )
        f.write(f"BIASED vs SUPERBIASED: U={stat:.2f}, p={p:.4e} ")
        f.write("Significant\n" if p < 0.05 else "Not significant\n")

correlation_matrix = pivot.corr(method="spearman")
plt.figure(figsize=(7, 6))
sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", fmt=".2f", square=True)
plt.title("Spearman Rank Correlation Between Models")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "rank_correlation_heatmap.png"))
plt.close()

for m1, m2 in combinations(pivot.columns, 2):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=pivot[m1], y=pivot[m2])
    plt.xlabel(f"Rank: {m1}")
    plt.ylabel(f"Rank: {m2}")
    plt.title(f"Rank Comparison: {m1} vs {m2}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"scatter_{m1}_vs_{m2}.png"))
    plt.close()

print("All plots and statistical analysis updated.")
