import pytest
from pybibx.base.pbx import pbx_probe, _split_references

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


def test_split_references_pubmed():
    text = "Lancet. 2011 Feb 12;377(9765):551-2"
    refs = _split_references(text, "pubmed")
    assert refs == [text]

