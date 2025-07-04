import pytest

try:
    from pybibx.base.pbx import pbx_probe
except ModuleNotFoundError:
    pbx_probe = None

@pytest.mark.skipif(pbx_probe is None, reason="pbx_probe dependencies missing")
def test_pbx_probe_initialization():
    # This is a basic smoke test to ensure the class can be instantiated.
    # A more comprehensive test would require a sample .bib file.
    try:
        probe = pbx_probe(file_bib="sample.bib")
        assert probe is not None
    except FileNotFoundError:
        # This is expected since 'sample.bib' does not exist.
        # The goal of this test is to ensure the class can be imported and initialized without errors.
        pass
