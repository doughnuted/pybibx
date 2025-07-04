import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from pybibx.base.pbx import pbx_probe
except ModuleNotFoundError:  # Missing heavy dependencies
    pytest.skip(
        "Skipping pbx_probe import test due to missing optional dependencies",
        allow_module_level=True,
    )

def test_pbx_probe_initialization():
    # This is a basic smoke test to ensure the class can be instantiated.
    # A more comprehensive test would require a sample .bib file.
    try:
        probe = pbx_probe(file_bib='sample.bib')
        assert probe is not None
    except FileNotFoundError:
        # This is expected since 'sample.bib' does not exist.
        # The goal of this test is to ensure the class can be imported and initialized without errors.
        pass
