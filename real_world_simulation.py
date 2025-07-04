import os
import pandas as pd
from bertopic import BERTopic
from pybibx.base.pbx import pbx_probe
import networkx as nx  # For creating a graph if needed

# Get the absolute path to the directory of the current script
# and construct the path to the sample bib file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_BIB_FILE = os.path.join(BASE_DIR, "assets", "bibs", "scopus_m.bib")

print(f"Attempting to load bib file: {SAMPLE_BIB_FILE}")

if not os.path.exists(SAMPLE_BIB_FILE):
    print(f"Error: Sample bib file not found at {SAMPLE_BIB_FILE}")
    exit(1)

try:
    print("Initializing pbx_probe...")
    probe = pbx_probe(file_bib=SAMPLE_BIB_FILE, db="scopus")
    print("pbx_probe initialized successfully.")
except Exception as e:
    print(f"Error initializing pbx_probe: {e}")
    exit(1)

# --- Basic Topic Modeling (Prerequisite) ---
try:
    print("\n--- Running Basic Topic Modeling ---")
    abstracts = probe.data["abstract"].tolist()
    abstracts = [str(abs_text) if pd.notna(abs_text) else "" for abs_text in abstracts]

    # Check if abstracts are empty or too few for BERTopic
    if not any(abstracts):
        print("Warning: All abstracts are empty. Skipping topic modeling.")
    elif len(abstracts) < 5:  # BERTopic might need a few documents
        print(
            f"Warning: Very few documents ({len(abstracts)}). Topic modeling might be suboptimal or fail."
        )
        # For very small datasets, BERTopic might still run with all-MiniLM-L6-v2 but may not produce meaningful topics.

    print("Fitting BERTopic model...")
    probe.topic_model = BERTopic(
        verbose=False, embedding_model="all-MiniLM-L6-v2", min_topic_size=2
    )
    probe.topics, probe.probs = probe.topic_model.fit_transform(abstracts)
    probe.topics_df = probe.topic_model.get_topic_info()
    print("BERTopic model fitted.")
    print(f"Topics DataFrame head:\n{probe.topics_df.head()}")
    if probe.topics_df.empty or len(probe.topics_df) <= 1:  # Topic -1 is outliers
        print(
            "Warning: BERTopic did not produce meaningful topics (empty or only outlier topic)."
        )

except Exception as e:
    print(f"Error during basic topic modeling: {e}")
    # Continue if possible, as some network features might still work

# --- Dynamic Topic Modeling ---
if (
    hasattr(probe, "topic_model")
    and probe.topics_df is not None
    and not probe.topics_df.empty
    and len(probe.topics_df) > 1
):
    try:
        print("\n--- Testing Dynamic Topic Modeling ---")
        timestamps = (
            pd.to_datetime(probe.data["year"].replace("UNKNOWN", pd.NaT))
            .dt.year.fillna(0)
            .astype(int)
            .tolist()
        )
        docs_for_dynamic = [
            str(text) if pd.notna(text) else ""
            for text in probe.data["abstract"].tolist()
        ]
        min_len = min(len(docs_for_dynamic), len(timestamps))
        docs_for_dynamic = docs_for_dynamic[:min_len]
        timestamps = timestamps[:min_len]

        print("Calling topics_dynamic()...")
        probe.topics_dynamic(documents=docs_for_dynamic, timestamps=timestamps)
        print(
            f"topics_over_time_df created. Shape: {probe.topics_over_time_df.shape if probe.topics_over_time_df is not None else 'None'}"
        )
        if probe.topics_over_time_df is None or probe.topics_over_time_df.empty:
            print("Warning: Dynamic topics DataFrame is None or empty.")

        print(
            "Calling graph_topics_dynamic()... (Plotly output will not be visible here)"
        )
        probe.graph_topics_dynamic()
        print("graph_topics_dynamic() executed.")

        print("Calling AI functions for dynamic topics...")
        print(f"ChatGPT TD: {probe.ask_chatgpt_td(api_key='test_key_not_used')}")
        print(f"Gemini TD: {probe.ask_gemini_td(api_key='test_key_not_used')}")
    except Exception as e:
        print(f"Error in Dynamic Topic Modeling section: {e}")
else:
    print(
        "\nSkipping Dynamic Topic Modeling as basic BERTopic setup failed or produced no topics."
    )

# --- Hierarchical Topic Modeling ---
if (
    hasattr(probe, "topic_model")
    and probe.topics_df is not None
    and not probe.topics_df.empty
    and len(probe.topics_df) > 1
):
    try:
        print("\n--- Testing Hierarchical Topic Modeling ---")
        print("Calling topics_hierarchical()...")
        probe.topics_hierarchical()
        print(
            f"hierarchical_topics_df created. Shape: {probe.hierarchical_topics_df.shape if probe.hierarchical_topics_df is not None else 'None'}"
        )
        if probe.hierarchical_topics_df is None or probe.hierarchical_topics_df.empty:
            print("Warning: Hierarchical topics DataFrame is None or empty.")

        print(
            "Calling graph_topics_hierarchical()... (Plotly output will not be visible here)"
        )
        probe.graph_topics_hierarchical()
        print("graph_topics_hierarchical() executed.")

        print("Calling AI functions for hierarchical topics...")
        print(f"ChatGPT TH: {probe.ask_chatgpt_th(api_key='test_key_not_used')}")
        print(f"Gemini TH: {probe.ask_gemini_th(api_key='test_key_not_used')}")
    except Exception as e:
        print(f"Error in Hierarchical Topic Modeling section: {e}")
else:
    print(
        "\nSkipping Hierarchical Topic Modeling as basic BERTopic setup failed or produced no topics."
    )

# --- Network Generation (Prerequisite for Louvain & Metrics) ---
sample_graph = None
try:
    print("\n--- Generating Co-authorship Network ---")
    probe.network_co_authorship(
        topn_authors=10, report=False
    )  # Using a small number for faster execution
    if probe.graph_obj and probe.graph_obj.number_of_nodes() > 0:
        sample_graph = probe.graph_obj
        print(
            f"Co-authorship network generated. Nodes: {sample_graph.number_of_nodes()}, Edges: {sample_graph.number_of_edges()}"
        )
    else:
        print(
            "Warning: Co-authorship network is empty or not generated. Creating a dummy graph."
        )
        sample_graph = nx.Graph()
        sample_graph.add_edges_from(
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
        )
        probe.graph_obj = (
            sample_graph  # So that graph_network_community_louvain can use it
        )
        print(
            f"Dummy network generated. Nodes: {sample_graph.number_of_nodes()}, Edges: {sample_graph.number_of_edges()}"
        )

except Exception as e:
    print(f"Error generating co-authorship network: {e}")
    print("Creating a dummy graph for subsequent network tests.")
    sample_graph = nx.Graph()
    sample_graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
    )
    probe.graph_obj = (
        sample_graph  # Ensure it's set for graph_network_community_louvain
    )

# --- Louvain Community Detection ---
if sample_graph:
    try:
        print("\n--- Testing Louvain Community Detection ---")
        print("Calling network_community_louvain()...")
        communities = probe.network_community_louvain(graph_obj=sample_graph)
        print(f"Louvain communities detected: {probe.louvain_communities}")
        if not probe.louvain_communities:
            print(
                "Warning: No Louvain communities detected or communities list is empty."
            )

        print(
            "Calling graph_network_community_louvain()... (Plotly output will not be visible here)"
        )
        probe.graph_network_community_louvain(
            graph_obj=sample_graph, communities=probe.louvain_communities
        )
        print("graph_network_community_louvain() executed.")

        print("Calling AI functions for Louvain communities...")
        print(f"ChatGPT NCL: {probe.ask_chatgpt_ncl(api_key='test_key_not_used')}")
        print(f"Gemini NCL: {probe.ask_gemini_ncl(api_key='test_key_not_used')}")
    except Exception as e:
        print(f"Error in Louvain Community Detection section: {e}")
else:
    print("\nSkipping Louvain Community Detection as sample graph was not generated.")

# --- Network Metrics Calculation ---
if sample_graph:
    try:
        print("\n--- Testing Network Metrics Calculation ---")
        print("Calling network_calculate_metrics()...")
        metrics = probe.network_calculate_metrics(graph_obj=sample_graph)
        print(f"Network metrics calculated: {probe.network_metrics is not None}")
        if probe.network_metrics:
            print(f"Metrics keys: {list(probe.network_metrics.keys())}")
            print(f"Graph Density: {probe.network_metrics.get('graph_density')}")
        else:
            print("Warning: Network metrics dictionary is None.")

        print("Calling AI functions for network metrics...")
        print(f"ChatGPT NM: {probe.ask_chatgpt_nm(api_key='test_key_not_used')}")
        print(f"Gemini NM: {probe.ask_gemini_nm(api_key='test_key_not_used')}")
    except Exception as e:
        print(f"Error in Network Metrics Calculation section: {e}")
else:
    print("\nSkipping Network Metrics Calculation as sample graph was not generated.")

print("\n--- Real-world usage simulation script finished ---")
