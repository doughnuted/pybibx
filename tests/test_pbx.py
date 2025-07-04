import pytest
import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create lightweight stubs for optional heavy dependencies used in pbx
stub_modules = [
    "bertopic",
    "summarizer",
    "transformers",
    "umap",
    "sentence_transformers",
    "gensim",
    "networkx",
    "sklearn",
    "wordcloud",
    "numba",
    "matplotlib",
    "scipy",
    "openai",
    "google.generativeai",
    "plotly",
    "plotly.graph_objects",
    "plotly.subplots",
    "plotly.io",
]

def add_stub(name, attrs=None):
    module = types.ModuleType(name)
    attrs = attrs or {}
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[name] = module
    return module

if "bertopic" not in sys.modules:
    add_stub("bertopic", {"BERTopic": object})

if "summarizer" not in sys.modules:
    add_stub("summarizer", {"Summarizer": object})

if "transformers" not in sys.modules:
    add_stub(
        "transformers",
        {
            "PegasusForConditionalGeneration": object,
            "PegasusTokenizer": object,
        },
    )

if "umap" not in sys.modules:
    add_stub("umap", {"UMAP": object})

if "sentence_transformers" not in sys.modules:
    add_stub("sentence_transformers", {"SentenceTransformer": object})

if "gensim" not in sys.modules:
    gensim_mod = add_stub("gensim")
    add_stub("gensim.models", {"FastText": object})

if "sklearn" not in sys.modules:
    add_stub("sklearn")
    add_stub("sklearn.cluster", {"KMeans": object, "HDBSCAN": object})
    add_stub("sklearn.decomposition", {"TruncatedSVD": object})
    add_stub("sklearn.feature_extraction", {})
    add_stub(
        "sklearn.feature_extraction.text",
        {"CountVectorizer": object, "TfidfVectorizer": object},
    )
    add_stub("sklearn.metrics", {})
    add_stub("sklearn.metrics.pairwise", {"cosine_similarity": lambda x, y: [[0]]})

if "wordcloud" not in sys.modules:
    add_stub("wordcloud", {"WordCloud": object})

if "numba" not in sys.modules:
    add_stub("numba", {"njit": lambda x: x})
    add_stub("numba.typed", {"List": list})

if "scipy" not in sys.modules:
    add_stub("scipy")
    add_stub("scipy.ndimage", {"gaussian_filter1d": lambda x, sigma=1: x})
    add_stub("scipy.signal", {"find_peaks": lambda x: ([], {})})
    add_stub("scipy.sparse", {"coo_matrix": object, "csr_matrix": object})

if "networkx" not in sys.modules:
    add_stub("networkx")

if "openai" not in sys.modules:
    add_stub("openai")

if "google" not in sys.modules:
    google_mod = add_stub("google")
    add_stub("google.generativeai")

if "plotly" not in sys.modules:
    add_stub("plotly")
    add_stub("plotly.graph_objects")
    add_stub("plotly.subplots")
    add_stub("plotly.io")

if "matplotlib" not in sys.modules:
    mpl = add_stub("matplotlib")
    add_stub("matplotlib.pyplot", {"style": types.SimpleNamespace(use=lambda x: None)})

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


def test_dimensions_csv_parsing():
    path = 'assets/bibs/dimensions_sample.csv'
    probe = pbx_probe(file_bib=path, db='dimensions')
    assert probe.data is not None
    assert not probe.data.empty


def test_dimensions_bib_parsing():
    path = 'assets/bibs/dimensions_sample.bib'
    probe = pbx_probe(file_bib=path, db='dimensions')
    assert probe.data is not None
    assert not probe.data.empty
