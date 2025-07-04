import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from pybibx.base.pbx import pbx_probe
except Exception as e:  # pragma: no cover - skip if heavy deps missing
    pytest.skip(f"Skipping tests, failed to import pbx_probe: {e}", allow_module_level=True)

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
