from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pybibx',
    version='5.1.4',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pybibx',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'bertopic',
        'chardet',
        'google-generativeai',
        'gensim',
        'llmx',
        'matplotlib',
        'networkx',
        'numba',
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'scikit-learn',
        'sentencepiece',
        'umap-learn',
        'openai',
        'wordcloud'
    ],
    extras_require={
        'gpu': [
            'bertopic',
            'bert-extractive-summarizer',
            'sentence-transformers',
            'torch',
            'torchvision',
            'torchaudio',
            'transformers',
        ]
    },
    zip_safe=True,
    description='A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
