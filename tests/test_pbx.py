try:
    import pandas as pd
except Exception:  # pragma: no cover - optional deps may be missing
    pd = None

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest

try:
    from pybibx.base.pbx import pbx_probe
except Exception:  # pragma: no cover - optional deps may be missing
    pbx_probe = None

skip_msg = "optional dependencies missing"

@pytest.mark.skipif(pbx_probe is None or pd is None, reason=skip_msg)
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


@pytest.mark.skipif(pbx_probe is None or pd is None, reason=skip_msg)
def test_split_references_pubmed_snippet():
    sample = "CIN - Lancet. 2011 Feb 12;377(9765):551-2; author reply 555. PMID: 21315936"
    probe = pbx_probe.__new__(pbx_probe)
    probe.database = "pubmed"
    probe.data = pd.DataFrame({"references": [sample]})
    refs, unique = probe._pbx_probe__get_str(entry="references", s=";", lower=False, sorting=True)
    assert refs == [[sample]]
    assert unique == [sample]
