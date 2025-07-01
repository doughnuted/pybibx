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

## Features

`pybibx` offers a comprehensive suite of tools for bibliometric and scientometric analysis, including:

*   **Data Loading & Processing:**
    *   Load data from Scopus (.csv), Web of Science (.bib), and PubMed (.nbib).
    *   Handle duplicate entries.
    *   Filter data by document type, year, sources, keywords, country, language, etc.
    *   Merge multiple bibliographic datasets.
    *   Save and load processed datasets.
*   **Bibliometric Analysis:**
    *   Exploratory Data Analysis (EDA) reports.
    *   Dataset health checks (completeness of entries).
    *   Calculation of H-index, G-index, M-index, E-index for authors.
    *   Bradford's Law analysis.
    *   Lotka's Law analysis.
    *   Identification of Sleeping Beauties and Princes.
*   **Network Analysis & Visualization:**
    *   Collaboration networks (authors, institutions, countries).
    *   Co-occurrence networks (author keywords, keywords plus).
    *   Co-citation networks.
    *   Bibliographic coupling networks.
    *   Citation history networks (main path analysis).
    *   Sankey diagrams for entity flows.
    *   Geographic mapping of collaborations.
*   **Text Analysis & NLP:**
    *   Text cleaning utilities.
    *   TF-IDF matrix generation.
    *   Word cloud generation.
    *   N-gram analysis.
    *   Word embeddings (FastText) with similarity and analogy operations.
*   **Topic Modeling:**
    *   Topic creation and reduction using BERTopic.
    *   Visualization of topics, topic distributions, topic projections, heatmaps, and evolution over time.
*   **AI-Powered Insights:**
    *   Abstractive and extractive text summarization using Pegasus, OpenAI GPT, Google Gemini, and BERT.
    *   Querying OpenAI GPT and Google Gemini for insights on various generated analyses (e.g., productivity plots, network data).

## Usage

Once the installation is complete, you can use the library in your Python projects.

**1. Initialize the library and load data:**

```python
from pybibx.base.pbx import pbx_probe # Corrected import

# Initialize with your bibliographic file and database type
# Supported db: "scopus", "wos", "pubmed"
# Example: file_path = "assets/bibs/scopus_m.bib"
file_path = "path/to/your/scopus_data.csv" # Or your .bib / .nbib file
probe = pbx_probe(file_bib=file_path, db="scopus")

# Access processed data
print(f"Loaded {probe.data.shape[0]} documents.")
```

**2. Perform basic analysis:**

```python
# Get an overview of the dataset
eda_report = probe.eda_bib()
print("\\nEDA Report:")
print(eda_report.to_string())

# Check data completeness
health_report = probe.health_bib()
print("\\nDataset Health:")
print(health_report.to_string())

# Get top 5 authors by document count
# This method also displays a plot by default.
# The data for the plot is stored in probe.ask_gpt_bp
print("\\nTop 5 Authors by Document Count (Plot will be displayed):")
probe.plot_bars(statistic="apd", topn=5)
print("\\nData for Top Authors Plot:")
print(probe.ask_gpt_bp.to_string())
```

**3. Generate a word cloud from abstracts:**

```python
# Ensure you have matplotlib installed if not already via environment.yml
# This function directly displays the plot.
print("\\nGenerating word cloud from abstracts (Plot will be displayed)...")
probe.word_cloud_plot(entry="abs", wordsn=100, rmv_custom_words=["research", "study", "paper"])
# Word frequencies are stored in probe.ask_gpt_wd
```

**4. Analyze author collaborations (displays a plot):**

```python
# This will plot the collaboration network for the top 3 authors
print("\\nVisualizing collaboration network for top 3 authors (Plot will be displayed)...")
probe.network_collab(entry="aut", topn=3, rows=1, cols=3)
# Collaboration data for AI query is in probe.ask_gpt_ct
```

**5. Get insights using AI (requires API keys):**

*Note: Ensure you have set your OpenAI API key or Google API key as environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`) or pass them directly via the `api_key` parameter in the respective methods.*

```python
# Example: Get insights on the EDA report
# Replace "YOUR_OPENAI_API_KEY" with your actual key if not set as an environment variable.
# insights_eda = probe.ask_chatgpt_eda(api_key="YOUR_OPENAI_API_KEY", query="Provide a brief summary of these bibliometric indicators.")
# print("\\nAI Insights on EDA Report (ChatGPT):")
# print(insights_eda)

# Example: Get insights on the Word Cloud using Gemini
# Replace "YOUR_GEMINI_API_KEY" with your actual key if not set as an environment variable.
# insights_wordcloud = probe.ask_gemini_wordcloud(api_key="YOUR_GEMINI_API_KEY", query="What are the main themes in this word cloud based on word frequency?")
# print("\\nAI Insights on Word Cloud (Gemini):")
# print(insights_wordcloud)
```

This is a brief overview. `pybibx` offers many more methods for in-depth analysis and visualization. Please refer to the docstrings within the `pbx_probe` class in `pybibx/base/pbx.py` for detailed information on each function, its parameters, and what it returns or sets.