import importlib
import os
import sys
import types

import pandas as pd  # ensure pandas is available for asserts


def load_pbx_probe():
    heavy_modules = [
        'openai', 'google', 'google.generativeai', 'gensim', 'gensim.models',
        'gensim.models.FastText', 'summarizer', 'transformers',
        'sentence_transformers', 'umap', 'wordcloud', 'sklearn',
        'sklearn.cluster', 'sklearn.decomposition', 'sklearn.feature_extraction',
        'sklearn.feature_extraction.text', 'sklearn.metrics',
        'sklearn.metrics.pairwise', 'bertopic', 'numba', 'numba.typed', 'scipy',
        'scipy.ndimage', 'scipy.signal', 'scipy.sparse', 'matplotlib',
        'matplotlib.pyplot',
    ]
    for mod in heavy_modules:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    sys.modules['gensim.models.FastText'].FastText = lambda *a, **k: None
    sys.modules['summarizer'].Summarizer = lambda *a, **k: None
    sys.modules['transformers'].PegasusForConditionalGeneration = object
    sys.modules['transformers'].PegasusTokenizer = object
    sys.modules['sentence_transformers'].SentenceTransformer = object
    sys.modules['umap'].UMAP = object
    sys.modules['wordcloud'].WordCloud = object
    sys.modules['sklearn.cluster'].KMeans = object
    sys.modules['sklearn.cluster'].HDBSCAN = object
    sys.modules['sklearn.decomposition'].TruncatedSVD = object
    sys.modules['sklearn.feature_extraction.text'].CountVectorizer = object
    sys.modules['sklearn.feature_extraction.text'].TfidfVectorizer = object
    sys.modules['sklearn.metrics.pairwise'].cosine_similarity = lambda X, Y: None
    sys.modules['bertopic'].BERTopic = object
    sys.modules['numba'].njit = lambda *a, **k: (lambda f: f)
    sys.modules['numba.typed'].List = list
    sys.modules['scipy.ndimage'].gaussian_filter1d = lambda a, sigma: a
    sys.modules['scipy.signal'].find_peaks = lambda *a, **k: ([], {})
    sys.modules['scipy.sparse'].coo_matrix = lambda *a, **k: None
    sys.modules['scipy.sparse'].csr_matrix = lambda *a, **k: None
    sys.modules['matplotlib.pyplot'].style = types.SimpleNamespace(use=lambda x: None)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    pbx_module = importlib.import_module('pybibx.base.pbx')
    return pbx_module.pbx_probe


def test_pbx_probe_parses_sample_bib():
    pbx_probe = load_pbx_probe()
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample.bib')
    probe = pbx_probe(file_bib=sample_path)
    assert probe.data.loc[0, 'title'] == 'Sample Title'
    assert probe.data.loc[0, 'author'] == 'Doe, John and Smith, Jane'
    assert probe.data.loc[0, 'abbrev_source_title'] == 'Test Journal'
    assert probe.data.shape[0] == 1
