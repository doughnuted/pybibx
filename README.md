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

## Features

pybibx provides a range of bibliometric and scientometric analysis tools, including:

*   Data loading and parsing from various bibliographic formats.
*   Descriptive statistics of your dataset (e.g., publication trends, top authors, top sources).
*   Author-level metrics:
    *   H-index, G-index, E-index.
    *   **Author Dominance Factor:** Measures the proportion of an author's multi-authored papers where they are the first author.
*   Network analysis:
    *   Co-authorship networks (authors, institutions, countries).
    *   Keyword co-occurrence networks.
    *   Citation networks (co-citation, bibliographic coupling, direct citation).
*   Visualization of networks and data trends.
*   Topic modeling.
*   Text summarization.
*   Integration with LLMs (OpenAI, Gemini) for insights.
*   And much more!