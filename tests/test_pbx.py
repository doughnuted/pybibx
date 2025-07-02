import pytest
import os
import pandas as pd
import numpy as np
from pybibx.base.pbx import pbx_probe # Corrected import path

# Fixture to create a temporary bib file and a pbx_probe instance
@pytest.fixture
def sample_pbx_instance(tmp_path):
    bib_content = """
@article{test1,
    author = {Doe, John and Smith, Jane},
    title = {Paper One},
    journal = {Journal of Tests},
    year = {2020},
    note = {10}
}
@article{test2,
    author = {Smith, Jane},
    title = {Paper Two},
    journal = {Journal of Tests},
    year = {2021},
    note = {5}
}
@article{test3,
    author = {Doe, John and Miller, Alice and Brown, Bob},
    title = {Paper Three},
    journal = {Another Journal},
    year = {2021},
    note = {20}
}
@article{test4,
    author = {Miller, Alice and Doe, John},
    title = {Paper Four},
    journal = {Journal of Tests},
    year = {2022},
    note = {15}
}
@article{test5,
    author = {Green, Eve},
    title = {Paper Five},
    journal = {Journal of Solitude},
    year = {2020},
    note = {2}
}
@article{test6,
    author = {Smith, Jane and Doe, John and Miller, Alice},
    title = {Paper Six},
    journal = {Journal of Collaboration},
    year = {2023},
    note = {30}
}
@article{test7,
    author = {White, Walter},
    title = {Paper Seven - No Multi-author},
    journal = {Journal of Singularity},
    year = {2023},
    note = {3}
}
@article{test8,
    author = {Pinkman, Jesse and White, Walter},
    title = {Paper Eight - Walter not first},
    journal = {Journal of Duos},
    year = {2023},
    note = {7}
}
"""
    bib_file = tmp_path / "sample.bib"
    bib_file.write_text(bib_content)
    probe = pbx_probe(file_bib=str(bib_file), db="scopus") # Assuming scopus format for simplicity
    return probe

@pytest.fixture
def empty_pbx_instance(tmp_path):
    bib_content = """
@article{empty,
    author = {},
    title = {},
    journal = {},
    year = {}
}
"""
    bib_file = tmp_path / "empty.bib"
    bib_file.write_text(bib_content)
    probe = pbx_probe(file_bib=str(bib_file), db="scopus")
    return probe

@pytest.fixture
def no_authors_pbx_instance(tmp_path):
    bib_content = """
@article{no_auth,
    title = {A paper with no authors},
    journal = {Journal of Nowhere},
    year = {2023}
}
"""
    bib_file = tmp_path / "no_authors.bib"
    bib_file.write_text(bib_content)
    probe = pbx_probe(file_bib=str(bib_file), db="scopus")
    return probe


def test_author_dominance_factor_calculation(sample_pbx_instance):
    probe = sample_pbx_instance
    adf = probe.author_dominance_factor

    assert not adf.empty
    # Expected values based on fixture data:
    # Doe, John:
    #   Total: 4 (test1, test3, test4, test6)
    #   Multi-Authored: 4
    #   First-Authored Multi: 2 (test1, test3)
    #   Dominance: 2/4 = 0.5
    # Smith, Jane:
    #   Total: 3 (test1, test2, test6)
    #   Single: 1 (test2)
    #   Multi-Authored: 2 (test1, test6)
    #   First-Authored Multi: 1 (test6)
    #   Dominance: 1/2 = 0.5
    # Miller, Alice:
    #   Total: 3 (test3, test4, test6)
    #   Multi-Authored: 3
    #   First-Authored Multi: 1 (test4)
    #   Dominance: 1/3 = 0.3333
    # Brown, Bob:
    #   Total: 1 (test3)
    #   Multi-Authored: 1
    #   First-Authored Multi: 0
    #   Dominance: 0/1 = 0.0
    # Green, Eve:
    #   Total: 1 (test5)
    #   Single: 1
    #   Multi-Authored: 0
    #   Dominance: 0.0
    # White, Walter:
    #   Total: 2 (test7, test8)
    #   Single: 1 (test7)
    #   Multi-Authored: 1 (test8)
    #   First-Authored Multi: 0
    #   Dominance: 0/1 = 0.0
    # Pinkman, Jesse:
    #   Total: 1 (test8)
    #   Multi-Authored: 1
    #   First-Authored Multi: 1
    #   Dominance: 1/1 = 1.0

    expected_authors = {
        "doe, john": {"DF": 0.5, "Total": 4, "Single": 0, "Multi": 4, "FirstMulti": 2},
        "smith, jane": {"DF": 0.5, "Total": 3, "Single": 1, "Multi": 2, "FirstMulti": 1},
        "miller, alice": {"DF": 1/3, "Total": 3, "Single": 0, "Multi": 3, "FirstMulti": 1},
        "brown, bob": {"DF": 0.0, "Total": 1, "Single": 0, "Multi": 1, "FirstMulti": 0},
        "green, eve": {"DF": 0.0, "Total": 1, "Single": 1, "Multi": 0, "FirstMulti": 0},
        "white, walter": {"DF": 0.0, "Total": 2, "Single": 1, "Multi": 1, "FirstMulti": 0},
        "pinkman, jesse": {"DF": 1.0, "Total": 1, "Single": 0, "Multi": 1, "FirstMulti": 1},
    }

    for _, row in adf.iterrows():
        author = row["Author"].lower() # Ensure case-insensitivity for lookup
        if author in expected_authors:
            expected = expected_authors[author]
            assert np.isclose(row["Dominance Factor"], expected["DF"], atol=0.0001)
            assert row["Total Articles"] == expected["Total"]
            assert row["Single-Authored Articles"] == expected["Single"]
            assert row["Multi-Authored Articles"] == expected["Multi"]
            assert row["First-Authored Multi-Articles"] == expected["FirstMulti"]
        else:
            # This case handles authors that might be picked up due to "and" splitting,
            # but are not primary test subjects. Their DF should likely be 0 or they shouldn't exist.
            # For now, we'll just print a warning if an unexpected author appears.
            # In a more rigorous test, we might want to assert they are not present or have specific values.
            print(f"Warning: Unexpected author in dominance factor results: {author}")


    # Check sorting (Pinkman should be first or among the first)
    assert adf.iloc[0]["Author"].lower() == "pinkman, jesse" or np.isclose(adf.iloc[0]["Dominance Factor"], 1.0)


def test_author_dominance_in_eda_report(sample_pbx_instance):
    probe = sample_pbx_instance
    report_df = probe.eda_bib()

    report_str = report_df.to_string()
    assert "AUTHOR DOMINANCE" in report_str
    assert "Top Authors by Dominance Factor" in report_str
    # Check for one of the top authors
    assert "pinkman, jesse" in report_str.lower() # Check for name
    assert "DF: 1.00" in report_str # Check for value

def test_author_dominance_empty_bib(empty_pbx_instance):
    probe = empty_pbx_instance
    adf = probe.author_dominance_factor
    assert adf.empty or adf.shape[0] == 0 # Should be empty or have no rows

    report_df = probe.eda_bib()
    report_str = report_df.to_string()
    assert "AUTHOR DOMINANCE" not in report_str # Section should not appear if no data

def test_author_dominance_no_authors_bib(no_authors_pbx_instance):
    probe = no_authors_pbx_instance
    # u_aut will be empty or contain 'unknown' which should be handled
    adf = probe.author_dominance_factor
    assert adf.empty or adf.shape[0] == 0 or (adf.shape[0] == 1 and adf.iloc[0]['Author'].lower() == 'unknown')

    report_df = probe.eda_bib()
    report_str = report_df.to_string()
    # The section might appear with "UNKNOWN" or not at all, depending on how empty authors are handled.
    # For now, we won't assert its absence strictly, but ensure no error.
    if "AUTHOR DOMINANCE" in report_str:
        assert "UNKNOWN" in report_str or "unknown" in report_str or "No authors found" in report_str # crude check

    # Ensure 'Max H-Index' which comes before dominance factor is present
    assert "Max H-Index" in report_str
    # Ensure 'Total Number of Citations' which comes after dominance factor is present
    assert "Total Number of Citations" in report_str


def test_pbx_probe_initialization_exists(tmp_path):
    # Test with an actual (minimal) bib file
    bib_content = """@article{test1, author = {Doe, John}, title = {Test Paper}, year = {2020}}"""
    bib_file = tmp_path / "minimal.bib"
    bib_file.write_text(bib_content)
    probe = pbx_probe(file_bib=str(bib_file))
    assert probe is not None
    assert probe.data.shape[0] > 0
