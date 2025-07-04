import pytest
import pandas as pd

try:
    from pybibx.base.pbx import pbx_probe
except Exception:  # pragma: no cover - optional dependency missing
    pbx_probe = None

def test_pbx_probe_initialization():
    if pbx_probe is None:
        pytest.skip("pbx_probe dependencies not installed")
    probe = pbx_probe(file_bib='assets/bibs/scopus_m.bib', db='scopus')
    assert probe is not None


def test_biblio_analysis():
    if pbx_probe is None:
        pytest.skip("pbx_probe dependencies not installed")
    probe = pbx_probe(file_bib='assets/bibs/scopus_m.bib', db='scopus')
    df = probe.biblio_analysis()
    assert isinstance(df, pd.DataFrame)
    assert 'Total Documents' in df['Metric'].values
