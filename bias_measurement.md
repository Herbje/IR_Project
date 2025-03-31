## Fairness Analysis of BM25 Ranking on TREC-FAIR 2021

### Methodology
This study investigates biases present in the rankings produced by the BM25 retrieval model over the TREC-FAIR 2021 dataset. The dataset comprises a snapshot of English Wikipedia articles, annotated with document-level metadata such as geographic origin and editorial quality class. Queries and relevance judgments are provided by the TREC Fair Ranking Track 2021.

Our methodology consists of the following steps:

1. **Indexing and Retrieval**: We index the corpus using PyTerrier's BM25 implementation. A set of evaluation queries is then executed to generate ranked document lists.

2. **Metadata Mapping**: Using the `ir_datasets` package, we extract metadata on geographic region and article quality (e.g., B, GA, Stub). These features are joined with retrieval results to enable group-wise analysis.

3. **Fairness Metrics**: We compute several fairness metrics:
   - **Disparate Exposure**: Attention received by each group across all ranks, weighted by position.
   - **Disparate Impact**: Proportion of documents from each group in the top-10 ranks.
   - **Relevance Bias**: Average human-assigned relevance scores per group.
   - **Simulated Position Bias**: User click likelihood estimated using a standard inverse logarithmic model.
   - **Intersectional Fairness**: Joint analysis across region and quality to uncover compound disparities.

All metrics are normalized to account for group sizes, allowing for direct comparisons.

---

### Disparate Exposure by Region
This metric quantifies how much visibility each geographic group receives across the entire ranking, emphasizing documents ranked higher.

| Region                          | Exposure  | Corpus Distribution | Over-/Under-representation |
|---------------------------------|-----------|----------------------|-----------------------------|
| Europe                          | 93.99%    | 95.10%               | Slight underrepresentation |
| Unknown                         | 4.38%     | 2.62%                | Overrepresented             |
| Northern America                | 1.12%     | 1.66%                | Underrepresented            |
| Asia                            | 0.27%     | 0.32%                | Aligned                     |
| Africa                          | 0.04%     | 0.06%                | Underrepresented            |
| Latin America and the Caribbean| 0.05%     | 0.11%                | Underrepresented            |
| Oceania                         | 0.14%     | 0.13%                | Aligned                     |

**Interpretation:** The exposure closely mirrors the dataset distribution for most regions. However, "Unknown" documents receive disproportionate exposure, possibly due to missing metadata or their centrality in the Wikipedia graph.

---

### Disparate Impact by Region (Top-10)
Disparate impact measures how often documents from each region appear among the top 10 results—those most likely to be viewed by users.

| Region                          | Top-10 Proportion | Corpus Distribution | Over-/Under-representation |
|---------------------------------|--------------------|----------------------|-----------------------------|
| Europe                          | 90.63%             | 95.10%               | Underrepresented            |
| Unknown                         | 8.55%              | 2.62%                | Strongly overrepresented    |
| Northern America                | 0.61%              | 1.66%                | Underrepresented            |
| Oceania                         | 0.20%              | 0.13%                | Slightly overrepresented    |

**Interpretation:** Top-ranked results are overwhelmingly from Europe and "Unknown" regions. No African, Asian, or Latin American documents appear in the top 10—a strong signal of geographic imbalance.

---

### Relevance Bias
We examine the average relevance score (from qrels) for each region. This reflects how human assessors rated the utility of documents per group.

| Region                          | Average Relevance Score |
|---------------------------------|--------------------------|
| Unknown                         | 0.92                     |
| Europe                          | 0.16                     |
| Asia                            | 0.16                     |
| Latin America and the Caribbean| 0.14                     |
| Africa                          | 0.08                     |
| Northern America                | 0.06                     |
| Oceania                         | 0.04                     |

**Interpretation:** Documents with no geographic label ("Unknown") receive much higher relevance scores, suggesting either a labeling artifact or a tendency for central, general-topic articles to lack metadata.

---

### Disparate Exposure by Article Quality
Wikipedia articles are categorized into quality classes ranging from Featured Articles (FA) and Good Articles (GA) to Start-class and Stub-class. We measure exposure per quality level.

| Quality | Exposure  | Corpus Distribution | Over-/Under-representation |
|---------|-----------|----------------------|-----------------------------|
| B       | 94.20%    | 95.02%               | Aligned                     |
| C       | 2.09%     | 1.88%                | Slightly overrepresented    |
| FA      | 0.27%     | 0.27%                | Aligned                     |
| GA      | 0.56%     | 0.38%                | Overrepresented             |
| Start   | 1.85%     | 1.52%                | Slightly overrepresented    |
| Stub    | 1.04%     | 0.94%                | Slightly overrepresented    |

**Interpretation:** High-quality articles (B and GA) dominate the exposure. Notably, GA articles receive more exposure relative to their size, suggesting that BM25 may correlate well with Wikipedia’s editorial quality assessments.

---

### Disparate Impact by Quality (Top-10)
This metric looks at which quality classes appear in the top 10 ranked results.

| Quality | Top-10 Proportion |
|---------|-------------------|
| B       | 92.26%            |
| Start   | 2.85%             |
| C       | 2.44%             |
| Stub    | 1.22%             |
| FA      | 0.61%             |
| GA      | 0.61%             |

**Interpretation:** B-class articles dominate the top-10 rankings, consistent with their corpus size. However, GA and FA articles, despite being few in number, still manage to break into the top results, reflecting the model's sensitivity to article quality.

---

### Position Bias: Simulated Click Share
Assuming users are more likely to click higher-ranked documents, we simulate click share using a position-based decay model.

**Click Share by Region:**

| Region                          | Simulated Click Share |
|---------------------------------|------------------------|
| Europe                          | 52.82%                |
| Unknown                         | 46.38%                |
| Northern America                | 0.55%                 |
| Oceania                         | 0.07%                 |
| Asia                            | 0.13%                 |
| Africa                          | 0.02%                 |
| Latin America and the Caribbean| 0.03%                 |

**Click Share by Quality:**

| Quality | Simulated Click Share |
|---------|------------------------|
| B       | 52.92%                |
| GA      | 44.52%                |
| C       | 1.02%                 |
| Start   | 0.90%                 |
| FA      | 0.13%                 |
| Stub    | 0.51%                 |

**Interpretation:** Articles of quality B and GA are the most likely to be clicked, with GA articles punching above their weight given their small presence in the corpus. This reinforces earlier evidence that article quality plays a substantial role in visibility.

---

### Intersectional Fairness: Region × Quality
Analyzing the joint distribution of region and article quality offers a more granular view of systemic bias.

**Key Findings:**
- "Europe | B" receives ~93% of total exposure and ~90% of top-10 impact—an overwhelming dominance.
- "Unknown | GA" receives nearly half of the simulated clicks, despite making up a minuscule portion of the corpus.
- Documents from Africa, Latin America, or Asia with low quality never appear in top ranks, and barely register in overall exposure.

**Interpretation:** Compounding disadvantages are evident: being from an underrepresented region and having a low-quality label results in near-complete invisibility in ranked output.

---

### Conclusion
This analysis demonstrates that while BM25 retrieval is deterministic and seemingly neutral, its outputs reflect and amplify the structural imbalances present in the corpus:

- **Geographic Bias**: Documents from Europe dominate exposure and impact, while others, particularly Africa, Latin America, and Asia, are largely invisible in top ranks.
- **Metadata Gaps**: Articles with no region label ("Unknown") are both highly visible and highly rated, indicating either systemic metadata omissions or hidden quality drivers.
- **Quality Bias**: High-quality articles, particularly Good Articles (GA), receive outsized visibility and simulated user attention despite their small footprint.
- **Intersectional Injustice**: The interaction of geographic and quality-based disadvantages leads to compounded invisibility for certain document groups.

These findings underscore the need for fairness-aware IR methods and signal caution when deploying traditional ranking models on real-world datasets that reflect social, geographic, and editorial inequalities. Future work should explore mitigation strategies such as re-ranking for exposure parity, incorporating quality-aware diversity objectives, or enriching underrepresented document groups.

