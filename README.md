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

## Loading OpenAlex Data

`pybibx` understands the JSONL files distributed by [OpenAlex](https://openalex.org/).
To load such a file simply specify `db="openalex"` when creating the probe:

```python
from pybibx.base import pbx_probe

probe = pbx_probe(
    file_bib="assets/bibs/openalex_sample.jsonl",
    db="openalex",
    del_duplicated=False,
)
print(probe.data.head())
```
