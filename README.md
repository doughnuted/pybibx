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

### Visualize Missing Data

Install the optional visualization extras to enable `plot_missing`:

```bash
pip install pybibx[viz]
```

Then plot the missing values from a loaded bibliography:

```python
from pybibx.base.pbx import pbx_probe

probe = pbx_probe("sample.bib")
probe.plot_missing()
```