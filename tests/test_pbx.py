from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pybibx.base.pbx import pbx_probe

def test_pbx_probe_initialization():
    sample = Path(__file__).parent / "data" / "sample.bib"
    probe = pbx_probe(file_bib=str(sample))
    assert probe is not None
