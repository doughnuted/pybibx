"""Lightweight wrapper used in tests to avoid heavy dependencies."""
import os

class pbx_probe:
    def __init__(self, file_bib, db="scopus", del_duplicated=True):
        if not os.path.exists(file_bib):
            raise FileNotFoundError(f"{file_bib} not found")
        self.file_bib = file_bib
        self.db = db
        self.del_duplicated = del_duplicated
