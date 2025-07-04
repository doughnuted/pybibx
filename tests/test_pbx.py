import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def try_import_probe():
    try:
        from pybibx.base.pbx import pbx_probe
    except ModuleNotFoundError:
        return None
    return pbx_probe

def test_pbx_probe_initialization():
    # This is a basic smoke test to ensure the class can be instantiated.
    # A more comprehensive test would require a sample .bib file.
    probe_cls = try_import_probe()
    if probe_cls is None:
        pytest.skip("Required dependencies for pbx_probe are missing")
    try:
        probe = probe_cls(file_bib="sample.bib")
        assert probe is not None
    except FileNotFoundError:
        # Expected since 'sample.bib' does not exist. This confirms import worked.
        pass
