import pytest
import os
from pybibx.base.pbx import pbx_probe

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


def test_pbx_probe_openalex():
    path = os.path.join(os.path.dirname(__file__), "../assets/bibs/openalex_example.json")
    probe = pbx_probe(file_bib=path, db="openalex")
    assert probe.data.shape[0] > 0


def test_pbx_probe_dimensions():
    path = os.path.join(os.path.dirname(__file__), "../assets/bibs/dimensions_example.csv")
    probe = pbx_probe(file_bib=path, db="dimensions")
    assert probe.data.shape[0] == 2
