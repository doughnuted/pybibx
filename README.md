# pybibx

A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools

## Overview

This project uses a modern Python tooling setup:

*   **[Pixi](https://pixi.sh/):** Manages the overall development environment, including Python versions and task running. It ensures reproducible environments across different platforms.
*   **[Poetry](https://python-poetry.org/):** Defines and manages Python package dependencies (specified in `pyproject.toml`).
*   **[UV](https://github.com/astral-sh/uv):** An extremely fast Python package installer, used via Pixi/Poetry to speed up dependency installation from `pyproject.toml`.

## Prerequisites

1.  **Install Pixi:** Follow the instructions on the [official Pixi website](https://pixi.sh/latest/installation/). Pixi will handle the installation of Python and other tools needed for the project environment.

    *Verify your installation by running:*
    ```bash
    pixi --version
    ```

2.  **Install Git:** If not already installed, download and install Git from [git-scm.com](https://git-scm.com/downloads).

## Development Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Valdecy/pybibx.git
    cd pybibx
    ```

2.  **Activate Project Environment & Install Dependencies:**
    Pixi uses the `pixi.toml` file to manage project dependencies and tasks. The first time you run a Pixi command in the project, it will set up the necessary environment based on `[dependencies]` (like Python) and `[pypi-dependencies]` (like Ruff, Pytest).

    To install all project dependencies, including those defined in `pyproject.toml` (managed by Poetry and installed via UV), run the custom `install-deps` task:
    ```bash
    pixi run install-deps
    ```
    This command will:
    *   Ensure the Python version specified in `pixi.toml` is available.
    *   Install `ruff` and `pytest` as defined in `pixi.toml`'s `[pypi-dependencies]`.
    *   Run `poetry lock` to ensure `poetry.lock` is up-to-date with `pyproject.toml`.
    *   Use `uv pip compile` to resolve all Python package dependencies from `pyproject.toml` (including development dependencies).
    *   Use `uv pip sync` to install these Python packages into the environment.

    **Note on Large Dependencies (e.g., PyTorch):** This project includes dependencies like PyTorch (`torch`), which can be very large and require significant disk space. The `install-deps` task will attempt to install the version specified. If you encounter issues due to disk space or download times, you might need to ensure your system has adequate resources. For specific hardware acceleration (like CUDA for NVIDIA GPUs), PyTorch installation might require further customization based on your system, which is outside the scope of this basic setup.

3.  **Enter the Pixi Shell (Optional but Recommended):**
    To work within the project's managed environment directly in your shell:
    ```bash
    pixi shell
    ```
    This activates the environment, making all project tools and dependencies readily available on your PATH.

## Common Development Tasks

Tasks are defined in `pixi.toml` and can be run using `pixi run <task_name>` or simply `pixi <task_name>` if the task name doesn't conflict with built-in Pixi commands.

*   **Install/Update Python Dependencies:**
    ```bash
    pixi run install-deps
    ```

*   **Lint Code:**
    (Uses Ruff)
    ```bash
    pixi run lint
    ```

*   **Format Code:**
    (Uses Ruff)
    ```bash
    pixi run format
    ```

*   **Run Tests:**
    (Uses Pytest)
    ```bash
    pixi run test
    ```

*   **Run Jupyter Notebook:**
    (Assumes Jupyter is part of the dev dependencies in `pyproject.toml`)
    ```bash
    pixi run notebook
    ```

## Building the Project (using Poetry)

While Pixi manages the development environment and tasks, Poetry is still responsible for understanding the Python package structure defined in `pyproject.toml`.

To build the distributable package (wheel and sdist):
```bash
# Ensure dependencies are installed (if not already via pixi run install-deps)
pixi run install-deps

# Then, run poetry build within the Pixi environment
pixi run poetry build
```
This command needs `poetry` to be available. The `install-deps` task uses `poetry lock`, so Poetry is implicitly part of the toolchain. If `poetry` itself needs to be explicitly installed into the Pixi environment for the `build` command, you can add `poetry` to the `[pypi-dependencies]` in `pixi.toml`.

*(The above build instruction assumes Poetry CLI is accessible. If the Poetry version issue persists or for a pure Pixi build process, Pixi might eventually have its own build commands that respect pyproject.toml)*

## Contributing

Please refer to the `CONTRIBUTING.md` file for guidelines (if available).

## License

This project is licensed under the GNU General Public License v3 (GPLv3) - see the `LICENSE` file for details.
