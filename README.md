# pybibx

A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Valdecy/pybibx.git
    cd pybibx
    ```

2.  **Create the conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate pybibx-env
    ```

4.  **Install pybibx in editable mode:**
    ```bash
    pip install -e .
    ```

## Usage

Once the installation is complete, you can use the library in your Python projects.

```python
import pybibx

# Your code here
```

## Key Features

pybibx offers a comprehensive suite of tools for bibliometric and scientometric analysis, including:

*   **Data Ingestion:** Supports loading data from Scopus (.bib, .csv), Web of Science (.bib), and PubMed (.txt).
*   **Data Cleaning & Normalization:** Extensive preprocessing of bibliographic data.
*   **Bibliometric Indicators:**
    *   Productivity analysis (Lotka's Law).
    *   Citation analysis (H-index, G-index, E-index).
    *   Source impact analysis (Bradford's Law).
*   **Collaboration Analysis:**
    *   Co-authorship networks.
    *   Country collaboration networks.
    *   Institution collaboration networks.
*   **Keyword Analysis:**
    *   Co-occurrence networks.
    *   Keyword evolution plots.
*   **Reference Analysis:**
    *   Co-citation networks.
    *   Bibliographic coupling networks.
    *   Reference Publication Year Spectroscopy (RPYS).
*   **Topic Modeling:**
    *   Standard BERTopic for identifying latent topics in abstracts.
    *   **New:** Dynamic Topic Modeling to analyze how topics evolve over time.
    *   **New:** Hierarchical Topic Modeling to explore topic relationships at different granularities.
*   **Network Analysis:**
    *   Generation of various types of networks (co-authorship, co-occurrence, co-citation, bibliographic coupling).
    *   **New:** Louvain Community Detection to identify communities within these networks.
    *   **New:** Calculation of additional network metrics:
        *   Degree Centrality
        *   Betweenness Centrality
        *   Closeness Centrality
        *   Eigenvector Centrality
        *   Clustering Coefficient
        *   Graph Density
*   **Text Analysis:**
    *   Text summarization.
    *   Word embedding generation using FastText.
*   **Visualization:** Interactive plots using Plotly for most analyses.
*   **AI-Powered Insights:** Integration with OpenAI (GPT models) and Google Gemini to generate textual insights from analysis results.