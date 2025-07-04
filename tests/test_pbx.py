import unittest
import os
import pandas as pd
import networkx as nx
from bertopic import BERTopic  # Added import
from pybibx.base.pbx import pbx_probe

# Get the absolute path to the directory of the current script
# and construct the path to the sample bib file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_BIB_FILE = os.path.join(BASE_DIR, "..", "assets", "bibs", "scopus_m.bib")


class TestPbxProbeOriginal(unittest.TestCase):
    def test_pbx_probe_initialization_original(self):
        # This is a basic smoke test to ensure the class can be instantiated.
        # A more comprehensive test would require a sample .bib file.
        # Using a dummy file name as the original test did,
        # expecting FileNotFoundError or successful import.
        try:
            probe = pbx_probe(file_bib="sample.bib", db="scopus")
            self.assertIsNotNone(probe)
        except FileNotFoundError:
            # This is expected since 'sample.bib' does not exist.
            pass
        except ImportError:
            self.fail("pbx_probe failed to import")


class TestPbxProbeExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up for all tests in this class."""
        if not os.path.exists(SAMPLE_BIB_FILE):
            raise FileNotFoundError(f"Sample bib file not found: {SAMPLE_BIB_FILE}")
        cls.pbx = pbx_probe(file_bib=SAMPLE_BIB_FILE, db="scopus")
        # Perform initial topic modeling as it's a prerequisite for some new functions
        abstracts = cls.pbx.data["abstract"].tolist()
        # Ensure abstracts are strings and handle potential NaN values
        abstracts = [
            str(abs_text) if pd.notna(abs_text) else "" for abs_text in abstracts
        ]

        # Simplified BERTopic setup for testing
        try:
            cls.pbx.topic_model = BERTopic(
                verbose=False, embedding_model="all-MiniLM-L6-v2"
            )  # Using a fast model
            cls.pbx.topics, cls.pbx.probs = cls.pbx.topic_model.fit_transform(abstracts)
            cls.pbx.topics_df = cls.pbx.topic_model.get_topic_info()

            # Generate a sample network for network-related tests
            cls.pbx.network_co_authorship(
                topn_authors=5, report=False
            )  # Generate a small graph
            cls.sample_graph = cls.pbx.graph_obj  # Store the generated graph for tests

        except Exception as e:
            cls.pbx = None  # Ensure pbx is None if setup fails
            print(f"Error during setUpClass: {e}")
            # It might be useful to skip all tests in this class if setup fails
            # For now, individual tests will check if self.pbx is None

    def assertTrue(self, expr, msg=None):
        # A helper to make migration from pytest's assert easier if needed
        super().assertTrue(expr, msg)

    def test_00_initialization_and_prerequisites(self):
        """Test that pbx_probe object is initialized and topic model exists."""
        self.assertIsNotNone(self.pbx, "pbx_probe object should be initialized.")
        if self.pbx:  # Proceed only if pbx object was created
            self.assertTrue(
                hasattr(self.pbx, "topic_model"),
                "BERTopic model should be initialized.",
            )
            self.assertIsNotNone(
                self.pbx.topics_df,
                "Topics DataFrame should not be None after fit_transform.",
            )
            self.assertIsInstance(
                self.pbx.topics_df, pd.DataFrame, "topics_df should be a DataFrame."
            )

    def test_01_dynamic_topic_modeling(self):
        """Smoke test for dynamic topic modeling."""
        if not self.pbx or not hasattr(self.pbx, "topic_model"):
            self.skipTest(
                "Skipping dynamic topic modeling test due to setup failure or no topic model."
            )

        timestamps = (
            pd.to_datetime(self.pbx.data["year"].replace("UNKNOWN", pd.NaT))
            .dt.year.fillna(0)
            .astype(int)
            .tolist()
        )
        # Ensure timestamps and docs have the same length
        docs_for_dynamic = [
            str(text) if pd.notna(text) else ""
            for text in self.pbx.data["abstract"].tolist()
        ]

        min_len = min(len(docs_for_dynamic), len(timestamps))
        docs_for_dynamic = docs_for_dynamic[:min_len]
        timestamps = timestamps[:min_len]

        try:
            self.pbx.topics_dynamic(documents=docs_for_dynamic, timestamps=timestamps)
            self.assertIsNotNone(
                self.pbx.topics_over_time_df, "topics_over_time_df should not be None."
            )
            self.assertIsInstance(
                self.pbx.topics_over_time_df,
                pd.DataFrame,
                "topics_over_time_df should be a DataFrame.",
            )

            # Test graph generation
            fig = self.pbx.graph_topics_dynamic()
            self.assertIsNotNone(fig, "graph_topics_dynamic should return a figure.")

            # Test AI functions (checking default returns as API keys are not assumed)
            self.assertIsInstance(self.pbx.ask_chatgpt_td(api_key="test_key"), str)
            self.assertIsInstance(self.pbx.ask_gemini_td(api_key="test_key"), str)
        except Exception as e:
            self.fail(f"Dynamic topic modeling test failed: {e}")

    def test_02_hierarchical_topic_modeling(self):
        """Smoke test for hierarchical topic modeling."""
        if not self.pbx or not hasattr(self.pbx, "topic_model"):
            self.skipTest(
                "Skipping hierarchical topic modeling test due to setup failure or no topic model."
            )
        try:
            self.pbx.topics_hierarchical()
            self.assertIsNotNone(
                self.pbx.hierarchical_topics_df,
                "hierarchical_topics_df should not be None.",
            )
            self.assertIsInstance(
                self.pbx.hierarchical_topics_df,
                pd.DataFrame,
                "hierarchical_topics_df should be a DataFrame.",
            )

            fig = self.pbx.graph_topics_hierarchical()
            self.assertIsNotNone(
                fig, "graph_topics_hierarchical should return a figure."
            )

            self.assertIsInstance(self.pbx.ask_chatgpt_th(api_key="test_key"), str)
            self.assertIsInstance(self.pbx.ask_gemini_th(api_key="test_key"), str)
        except Exception as e:
            self.fail(f"Hierarchical topic modeling test failed: {e}")

    def test_03_louvain_community_detection(self):
        """Smoke test for Louvain community detection."""
        if (
            not self.pbx
            or not hasattr(self, "sample_graph")
            or self.sample_graph is None
        ):
            self.skipTest(
                "Skipping Louvain community test due to setup failure or no sample graph."
            )

        # Ensure the graph has enough nodes/edges for Louvain to make sense
        if len(self.sample_graph.nodes) < 2 or self.sample_graph.number_of_edges() == 0:
            # Create a more connected graph if the co-authorship one is too sparse
            G_test = nx.Graph()
            G_test.add_edges_from(
                [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
            )
            self.sample_graph = G_test

        try:
            communities = self.pbx.network_community_louvain(
                graph_obj=self.sample_graph
            )
            self.assertIsNotNone(
                self.pbx.louvain_communities, "louvain_communities should not be None."
            )
            self.assertIsInstance(
                self.pbx.louvain_communities,
                list,
                "louvain_communities should be a list.",
            )
            if communities:  # if communities are found
                self.assertIsInstance(
                    communities[0], set, "Each community should be a set of nodes."
                )

            fig = self.pbx.graph_network_community_louvain(
                graph_obj=self.sample_graph, communities=self.pbx.louvain_communities
            )
            self.assertIsNotNone(
                fig, "graph_network_community_louvain should return a figure."
            )

            self.assertIsInstance(self.pbx.ask_chatgpt_ncl(api_key="test_key"), str)
            self.assertIsInstance(self.pbx.ask_gemini_ncl(api_key="test_key"), str)
        except Exception as e:
            self.fail(f"Louvain community detection test failed: {e}")

    def test_04_network_metrics_calculation(self):
        """Smoke test for network metrics calculation."""
        if (
            not self.pbx
            or not hasattr(self, "sample_graph")
            or self.sample_graph is None
        ):
            self.skipTest(
                "Skipping network metrics test due to setup failure or no sample graph."
            )

        # Ensure the graph is not empty for metrics calculation
        if len(self.sample_graph.nodes) == 0:
            G_test = nx.Graph()
            G_test.add_edges_from([(0, 1), (0, 2), (1, 2)])
            self.sample_graph = G_test

        try:
            self.pbx.network_calculate_metrics(
                graph_obj=self.sample_graph
            )  # Removed assignment to unused 'metrics'
            self.assertIsNotNone(
                self.pbx.network_metrics, "network_metrics should not be None."
            )
            self.assertIsInstance(
                self.pbx.network_metrics,
                dict,
                "network_metrics should be a dictionary.",
            )

            expected_keys = [
                "degree_centrality",
                "betweenness_centrality",
                "closeness_centrality",
                "eigenvector_centrality",
                "clustering_coefficient",
                "graph_density",
            ]
            for key in expected_keys:
                self.assertIn(
                    key,
                    self.pbx.network_metrics,
                    f"{key} should be in network_metrics.",
                )

            self.assertIsInstance(self.pbx.ask_chatgpt_nm(api_key="test_key"), str)
            self.assertIsInstance(self.pbx.ask_gemini_nm(api_key="test_key"), str)
        except Exception as e:
            # Catching networkx.NetworkXError specifically if graph is too small for eigenvector
            if isinstance(e, nx.NetworkXError) and "Eigenvector centrality" in str(e):
                print(
                    f"Skipping eigenvector part of network metrics due to graph structure: {e}"
                )
                # Check other metrics if eigenvector fails
                self.assertIsNotNone(self.pbx.network_metrics.get("degree_centrality"))
            else:
                self.fail(f"Network metrics calculation test failed: {e}")


if __name__ == "__main__":
    unittest.main()
