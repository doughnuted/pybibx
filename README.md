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
from pybibx.base import pbx_probe

# Initialize the probe with your .bib file
probe = pbx_probe(file_bib="path/to/library.bib", db="scopus")

# Display basic information about the bibliography
print(probe.eda_bib().head())
```
