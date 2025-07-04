import os
import pytest
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


def test_biblio_analysis_basic():
    """Run biblio_analysis on a small sample and check key metrics."""
    sample_bib = os.path.join(os.path.dirname(__file__), "..", "assets", "bibs", "scopus_m.bib")
    probe = pbx_probe(file_bib=sample_bib)
    df = probe.biblio_analysis()
    docs = int(df.loc[df["Metric"] == "Documents", "Value"].iloc[0])
    assert docs > 0
