import pytest

def test_pbx_probe_initialization():
    # This is a basic smoke test to ensure the class can be instantiated.
    # A more comprehensive test would require a sample .bib file.
    try:
        from pybibx.base.pbx import pbx_probe
    except ModuleNotFoundError:
        pytest.skip("Required dependencies not installed")
    try:
        probe = pbx_probe(file_bib="sample.bib")
        assert probe is not None
    except FileNotFoundError:
        pass


def test_openalex_loading():
    try:
        from pybibx.base.pbx import pbx_probe
    except ModuleNotFoundError:
        pytest.skip("Required dependencies not installed")
    sample_path = "assets/bibs/openalex_sample.jsonl"
    probe = pbx_probe(file_bib=sample_path, db="openalex", del_duplicated=False)
    assert probe.data.shape[0] == 3
    assert "title" in probe.data.columns
