# EMATM0067 - Comparative Corpus Analysis (Task 3)

**EMATM0067 — Introduction to AI and Text Analytics**  
University of Bristol, Department of Data Science

---

## Overview

This repository contains the code and report for a comparative corpus analysis of 27,121 ArXiv abstracts in the fields of AI, machine learning, and NLP. The abstracts are spanning three periods of 2007–2011, 2012–2016, and 2017–2021.

The project addresses two research questions:
1. How did the topic focus and communication of AI/ML/NLP research fields vary between 2007 and 2021?
2. How does the choice of text analytics method impact the findings drawn from the same corpus?

---

## Team

| Name | Contribution |
|---|---|
| Nithya Dharshini Uthayasankar | SBERT embeddings, framing heatmap (Axis 1 & 2) |
| Rohan Bhardwaj | PCA temporal trajectory (Axis 1) |
| Clarence Zhen Jin Tan | TextRank keyword extraction (Axis 2) |
| Lingjuan Shu | LDA topic modelling (Axis 1) |
| Taojie Chen | TF-IDF representation, period classifier (Axis 1 & 2) |

---

## Project Management
Trello Link : 

### Timeful Links


## Repository Structure

```
├── Final_Integration_code.ipynb   # Main integrated notebook of all methods
├── Report/
│   ├── coursework_report.tex      # LaTeX source for the group report
│   ├── references.bib             # Bibliography
│   └── figures/                   # All figures used in the report
└── README.md
```
---

## Methods

### Axis 1 — Text Representation
We compare three fundamentally different methods for encoding abstracts and measure the temporal drift by the centroid distance and silhouette score.

| Method | Description |
|---|---|
| TF-IDF | Sparse weighted vocabulary vectors (bigram, V=5,000) |
| SBERT | Dense semantic embeddings — MiniLM, MPNet, SPECTER |
| LDA | Probabilistic topic model (K=12 topics, selected by perplexity) |
| PCA Trajectory | 2D projection of each representation space to visualise temporal drift |

### Axis 2 — Period-Comparison Method
We compare the contrast between inductive and discriminative methods to identifying what changed between periods.

| Method | Description |
|---|---|
| TextRank | Unsupervised graph-based keyphrase extraction per time period |
| Period Classifier | Logistic regression on TF-IDF features, trained to label abstracts with the correct period |
| SBERT Framing Heatmap | Cosine similarity between period centroids and manually created framing archetype sentences |

---

## Key Findings

- The **2012–2016 to 2017–2021 transition is the dominant shift** across all five representations, as expected from the key events of the Transformer (2017), BERT (2018), and GPT-3 (2020).
- **LDA** shows the highest relative centroid shift (0.027 to 0.273), capturing changes in themes that cannot be detected by either TF-IDF or SBERT. 
- **SBERT** (MiniLM, MPNet) achieves the highest silhouette scores (0.054–0.056). They outperform TF-IDF on separating the periods despite their lower raw centroid distances.
- The **period classifier** can highlight terms (`pre-trained` +3.30, `graph neural` +2.62, `federated learning` +2.55 for 2017–2021) that TextRank cannot detect. It confirms that contrastive methods provide sharper insights than inductive frequency aggregation.
- We can see that no single representation dominates all other metrics. TF-IDF is the most interpretable, SBERT is the most semantically reliable, and LDA can uniquely identify *what* changed rather than only *how much*.

---

## Setup and Usage

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
``` 
### Data

The dataset is the [gfissore/arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) corpus from HuggingFace. A loading script is provided in the notebook.

### Running the Notebook

Open and run `Final_Integration_code.ipynb` end-to-end. The notebook is structured in sections matching the report:

1. Data loading and preprocessing (Rohan)
2. TF-IDF representation and centroid distances (Chen)
3. SBERT embeddings and framing heatmap (Nithya)
4. LDA topic model (Shu)
5. PCA temporal trajectory (Rohan)
6. TextRank keyword extraction (Clarence)
7. Period classifier (Chen)
8. Evaluation: silhouette scores, centroid distance comparison, figure generation

All figures are saved to `Report/figures/` automatically.

---

## Report

The group report is written in LaTeX using the NeurIPS 2026 template.

### If compiling locally:
```bash
cd Report
pdflatex coursework_report.tex
bibtex coursework_report
pdflatex coursework_report.tex
pdflatex coursework_report.tex
``` 

### If using Overleaf: open the project and click Recompile.
Main file is Report/coursework_report.tex

---

## References

- Blei et al. (2003). Latent Dirichlet Allocation. *JMLR*.
- Reimers & Gurevych (2019). Sentence-BERT. *EMNLP*.
- Cohan et al. (2020). SPECTER. *ACL*.
- Mihalcea & Tarau (2004). TextRank. *EMNLP*.
- Pedregosa et al. (2011). Scikit-learn. *JMLR*.