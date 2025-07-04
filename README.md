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
from pybibx.base.pbx import pbx_probe

# Load your bibliographic file
probe = pbx_probe(file_bib="sample.bib")

# Build the citation network and retrieve edges
edges = probe.network_hist()

# Plot the chronological citation graph using the returned edges
probe.hist_plot(edges)
```