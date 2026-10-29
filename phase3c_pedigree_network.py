# phase3c_pedigree_network.py

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data.csv"
START_YEAR = 2013
END_YEAR = 2025

OUTPUT_DIR = "phase3c_output"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def classify_parent(value):
    """
    Classifies a parent entry.

    Returns:
        named
        self
        open
        unknown
        sibling
        missing
    """

    x = clean_text(value)

    if not x:
        return "missing"

    s = x.lower().strip()

    # Remove punctuation for easier matching
    normalized = re.sub(r"[^a-z0-9\s]", " ", s)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Missing / unknown
    unknown_terms = {
        "unknown",
        "unknow",
        "unk",
        "n a",
        "na",
        "not known",
        "not available",
        "none",
        "no information",
        "no info",
        "?"
    }

    if normalized in unknown_terms:
        return "unknown"

    # Self pollination
    self_terms = {
        "self",
        "selfed",
        "self pollinated",
        "self pollination",
        "selfpollinated",
        "self pollination",
        "selfed pumpkin"
    }

    if normalized in self_terms:
        return "self"

    # Open pollination
    open_terms = {
        "open",
        "open pollinated",
        "open pollination",
        "open poll",
        "op"
    }

    if normalized in open_terms:
        return "open"

    # Sibling / sib descriptors
    sibling_terms = {
        "sib",
        "sibb",
        "sibling",
        "sibs",
        "sibling cross"
    }

    if normalized in sibling_terms:
        return "sibling"

    return "named"


def normalize_parent(value):
    """
    Conservative normalization.

    Does NOT aggressively rewrite names because we don't
    want to accidentally merge genuinely different cultivars/
    seed identifiers.
    """

    x = clean_text(value)

    if not x:
        return None

    # Normalize whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x


def parent_is_named(value):
    return classify_parent(value) == "named"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("PHASE 3C: CLEAN PEDIGREE NETWORK + LINEAGE ANALYSIS")
print("=" * 75)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} records.")

required_columns = [
    "Year",
    "Weight_lbs",
    "Grower_Name",
    "Seed_Mother",
    "Pollinator_Father"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Weight_lbs"] = pd.to_numeric(
    df["Weight_lbs"], errors="coerce"
)

df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

df["Year"] = df["Year"].astype(int)

print(
    f"Analysis period: {START_YEAR}-{END_YEAR}"
)
print(
    f"Records: {len(df):,}"
)


# ============================================================
# CLEAN PARENT FIELDS
# ============================================================

df["mother_type"] = df["Seed_Mother"].apply(
    classify_parent
)

df["pollinator_type"] = df["Pollinator_Father"].apply(
    classify_parent
)

df["mother_clean"] = df["Seed_Mother"].apply(
    normalize_parent
)

df["pollinator_clean"] = df["Pollinator_Father"].apply(
    normalize_parent
)

print("\nParent classification:")
print(
    pd.crosstab(
        df["mother_type"],
        columns="count"
    )
)

print("\nPollinator classification:")
print(
    pd.crosstab(
        df["pollinator_type"],
        columns="count"
    )
)


# ============================================================
# PART 1
# PARENT → OFFSPRING NETWORK
# ============================================================

print("\n" + "=" * 75)
print("PART 1: PARENT → OFFSPRING NETWORK")
print("=" * 75)

parent_rows = []

for idx, row in df.iterrows():

    # Mother
    if row["mother_type"] == "named":

        parent_rows.append({
            "parent": row["mother_clean"],
            "role": "mother",
            "offspring_record": idx,
            "year": row["Year"],
            "offspring_weight_lbs": row["Weight_lbs"],
            "grower": row["Grower_Name"]
        })

    # Pollinator
    if row["pollinator_type"] == "named":

        parent_rows.append({
            "parent": row["pollinator_clean"],
            "role": "pollinator",
            "offspring_record": idx,
            "year": row["Year"],
            "offspring_weight_lbs": row["Weight_lbs"],
            "grower": row["Grower_Name"]
        })


edges_df = pd.DataFrame(parent_rows)

edges_df.to_csv(
    f"{OUTPUT_DIR}/parent_offspring_edges_clean.csv",
    index=False
)

print(
    f"Documented named parent-offspring links: "
    f"{len(edges_df):,}"
)


# ============================================================
# PARENT STATISTICS
# ============================================================

parent_stats = (
    edges_df
    .groupby("parent")
    .agg(
        total_offspring_records=(
            "offspring_record",
            "count"
        ),
        years_documented=(
            "year",
            "nunique"
        ),
        first_year=(
            "year",
            "min"
        ),
        last_year=(
            "year",
            "max"
        ),
        max_offspring_weight_lbs=(
            "offspring_weight_lbs",
            "max"
        ),
        mean_offspring_weight_lbs=(
            "offspring_weight_lbs",
            "mean"
        )
    )
    .reset_index()
    .sort_values(
        "total_offspring_records",
        ascending=False
    )
)

parent_stats.to_csv(
    f"{OUTPUT_DIR}/parent_statistics_clean.csv",
    index=False
)

print("\nTOP DOCUMENTED PARENTS")
print("-" * 75)

print(
    parent_stats.head(25).to_string(
        index=False
    )
)


# ============================================================
# PART 2
# PARENT-PAIR NETWORK
# ============================================================

print("\n" + "=" * 75)
print("PART 2: DOCUMENTED MOTHER × POLLINATOR PAIRS")
print("=" * 75)

pairs = df[
    (df["mother_type"] == "named") &
    (df["pollinator_type"] == "named")
].copy()

pairs["mother"] = pairs["mother_clean"]
pairs["pollinator"] = pairs["pollinator_clean"]

pair_summary = (
    pairs
    .groupby(
        ["mother", "pollinator"]
    )
    .agg(
        records=("Year", "size"),
        first_year=("Year", "min"),
        last_year=("Year", "max"),
        years_documented=("Year", "nunique"),
        mean_weight_lbs=("Weight_lbs", "mean"),
        max_weight_lbs=("Weight_lbs", "max")
    )
    .reset_index()
    .sort_values(
        ["records", "years_documented"],
        ascending=False
    )
)

pair_summary.to_csv(
    f"{OUTPUT_DIR}/parent_pair_summary_clean.csv",
    index=False
)

print(
    f"Documented parent pairs: "
    f"{len(pair_summary):,}"
)

print("\nTOP RECURRING PAIRS")
print("-" * 75)

print(
    pair_summary.head(25).to_string(
        index=False
    )
)


# ============================================================
# PART 3
# BIPARTITE PARENT → OFFSPRING NETWORK
# ============================================================

print("\n" + "=" * 75)
print("PART 3: NETWORK STATISTICS")
print("=" * 75)

G = nx.Graph()

for _, row in edges_df.iterrows():

    parent = "PARENT:" + row["parent"]
    offspring = "OFFSPRING:" + str(row["offspring_record"])

    G.add_node(
        parent,
        node_type="parent",
        label=row["parent"]
    )

    G.add_node(
        offspring,
        node_type="offspring",
        label=str(row["offspring_record"])
    )

    if G.has_edge(parent, offspring):
        G[parent][offspring]["weight"] += 1
    else:
        G.add_edge(
            parent,
            offspring,
            weight=1
        )


parent_nodes = [
    n for n, d in G.nodes(data=True)
    if d.get("node_type") == "parent"
]

offspring_nodes = [
    n for n, d in G.nodes(data=True)
    if d.get("node_type") == "offspring"
]

print(f"Parent nodes: {len(parent_nodes):,}")
print(f"Offspring nodes: {len(offspring_nodes):,}")
print(f"Edges: {G.number_of_edges():,}")


# Parent degree statistics
degree_stats = []

for node in parent_nodes:

    label = G.nodes[node]["label"]

    degree = G.degree(node)

    weighted_degree = sum(
        data.get("weight", 1)
        for _, _, data
        in G.edges(node, data=True)
    )

    degree_stats.append({
        "parent": label,
        "degree": degree,
        "weighted_degree": weighted_degree
    })

degree_df = (
    pd.DataFrame(degree_stats)
    .sort_values(
        "weighted_degree",
        ascending=False
    )
)

degree_df.to_csv(
    f"{OUTPUT_DIR}/parent_network_degree_clean.csv",
    index=False
)


# ============================================================
# PART 4
# PARENT PERSISTENCE
# ============================================================

print("\n" + "=" * 75)
print("PART 4: PARENT PERSISTENCE")
print("=" * 75)

persistence = (
    edges_df
    .groupby("parent")
    .agg(
        years_present=("year", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        total_records=("offspring_record", "count")
    )
    .reset_index()
)

persistence["span_years"] = (
    persistence["last_year"]
    - persistence["first_year"]
    + 1
)

persistence["mean_records_per_present_year"] = (
    persistence["total_records"]
    / persistence["years_present"]
)

persistence = persistence.sort_values(
    ["years_present", "total_records"],
    ascending=False
)

persistence.to_csv(
    f"{OUTPUT_DIR}/parent_persistence_clean.csv",
    index=False
)

print(
    persistence.head(30).to_string(
        index=False
    )
)


# ============================================================
# PART 5
# HIGH-WEIGHT ASSOCIATIONS
# ============================================================

print("\n" + "=" * 75)
print("PART 5: HIGH-WEIGHT OFFSPRING ASSOCIATIONS")
print("=" * 75)

threshold = df["Weight_lbs"].quantile(0.99)

elite = df[
    df["Weight_lbs"] >= threshold
].copy()

print(
    f"99th percentile weight: "
    f"{threshold:.1f} lbs"
)

print(
    f"Top 1% offspring records: "
    f"{len(elite):,}"
)

elite_rows = []

for _, row in elite.iterrows():

    if row["mother_type"] == "named":
        elite_rows.append({
            "parent": row["mother_clean"],
            "role": "mother",
            "weight_lbs": row["Weight_lbs"],
            "year": row["Year"]
        })

    if row["pollinator_type"] == "named":
        elite_rows.append({
            "parent": row["pollinator_clean"],
            "role": "pollinator",
            "weight_lbs": row["Weight_lbs"],
            "year": row["Year"]
        })

elite_df = pd.DataFrame(elite_rows)

elite_summary = (
    elite_df
    .groupby(["parent", "role"])
    .agg(
        elite_records=("weight_lbs", "size"),
        mean_elite_weight=("weight_lbs", "mean"),
        max_elite_weight=("weight_lbs", "max"),
        years=("year", "nunique")
    )
    .reset_index()
    .sort_values(
        "elite_records",
        ascending=False
    )
)

elite_summary.to_csv(
    f"{OUTPUT_DIR}/elite_parent_associations_clean.csv",
    index=False
)

print(
    elite_summary.head(30).to_string(
        index=False
    )
)


# ============================================================
# PART 6
# ATLANTIC GIANT / DILL CANDIDATE SEARCH
# ============================================================

print("\n" + "=" * 75)
print("PART 6: HISTORICAL LINEAGE CANDIDATE SEARCH")
print("=" * 75)

keywords = [
    "dill",
    "atlantic giant",
    "atlantic"
]

lineage_rows = []

for parent in parent_stats["parent"]:

    s = parent.lower()

    matches = [
        k for k in keywords
        if k in s
    ]

    if matches:

        stats = parent_stats[
            parent_stats["parent"] == parent
        ].iloc[0]

        lineage_rows.append({
            "parent": parent,
            "matched_keywords": "; ".join(matches),
            "total_offspring_records":
                stats["total_offspring_records"],
            "years_documented":
                stats["years_documented"],
            "first_year":
                stats["first_year"],
            "last_year":
                stats["last_year"],
            "max_offspring_weight_lbs":
                stats["max_offspring_weight_lbs"]
        })

lineage_df = pd.DataFrame(lineage_rows)

if len(lineage_df):

    lineage_df = lineage_df.sort_values(
        "total_offspring_records",
        ascending=False
    )

    print(
        lineage_df.to_string(index=False)
    )

else:
    print(
        "No lineage-related parent identifiers found."
    )

lineage_df.to_csv(
    f"{OUTPUT_DIR}/lineage_candidate_parents_clean.csv",
    index=False
)


# ============================================================
# PART 7
# TEMPORAL PARENT REUSE
# ============================================================

print("\n" + "=" * 75)
print("PART 7: TEMPORAL PARENT REUSE")
print("=" * 75)

year_parent = (
    edges_df
    .groupby("year")
    .agg(
        named_parent_records=(
            "offspring_record",
            "count"
        ),
        unique_named_parents=(
            "parent",
            "nunique"
        )
    )
    .reset_index()
)

year_parent["unique_parent_ratio"] = (
    year_parent["unique_named_parents"]
    / year_parent["named_parent_records"]
)

year_parent.to_csv(
    f"{OUTPUT_DIR}/network_yearly_parent_reuse_clean.csv",
    index=False
)

print(
    year_parent.to_string(index=False)
)


# ============================================================
# PART 8
# CLEAN TOP-PARENT NETWORK
# ============================================================

print("\n" + "=" * 75)
print("PART 8: GENERATING CLEAN NETWORK GRAPH")
print("=" * 75)

# Top 20 parents by number of offspring records
top_parents = set(
    parent_stats
    .head(20)["parent"]
)

top_edges = edges_df[
    edges_df["parent"].isin(top_parents)
].copy()

TG = nx.Graph()

for _, row in top_edges.iterrows():

    parent = row["parent"]

    offspring = (
        f'{row["offspring_record"]}'
    )

    TG.add_node(
        parent,
        node_type="parent"
    )

    TG.add_node(
        offspring,
        node_type="offspring"
    )

    TG.add_edge(
        parent,
        offspring
    )


plt.figure(figsize=(16, 12))

pos = nx.spring_layout(
    TG,
    seed=42,
    k=0.7,
    iterations=100
)

parent_graph_nodes = [
    n for n, d in TG.nodes(data=True)
    if d["node_type"] == "parent"
]

offspring_graph_nodes = [
    n for n, d in TG.nodes(data=True)
    if d["node_type"] == "offspring"
]

nx.draw_networkx_edges(
    TG,
    pos,
    alpha=0.18,
    width=0.7
)

nx.draw_networkx_nodes(
    TG,
    pos,
    nodelist=offspring_graph_nodes,
    node_size=20,
    alpha=0.35
)

nx.draw_networkx_nodes(
    TG,
    pos,
    nodelist=parent_graph_nodes,
    node_size=500,
    alpha=0.9
)

nx.draw_networkx_labels(
    TG,
    pos,
    labels={
        n: n
        for n in parent_graph_nodes
    },
    font_size=8
)

plt.title(
    "Network of the 20 Most Frequently Documented Named Parents"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/top_20_parent_network_clean.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PART 9
# PARENT-PAIR NETWORK
# ============================================================

PG = nx.Graph()

for _, row in pair_summary.head(50).iterrows():

    mother = row["mother"]
    father = row["pollinator"]

    PG.add_node(mother)
    PG.add_node(father)

    PG.add_edge(
        mother,
        father,
        weight=row["records"]
    )


plt.figure(figsize=(16, 12))

pos = nx.spring_layout(
    PG,
    seed=42,
    k=1.2,
    iterations=150
)

weights = [
    0.5 + 0.4 * np.log1p(
        PG[u][v]["weight"]
    )
    for u, v in PG.edges()
]

nx.draw_networkx_edges(
    PG,
    pos,
    width=weights,
    alpha=0.4
)

node_sizes = [
    150 + 20 * PG.degree(n)
    for n in PG.nodes()
]

nx.draw_networkx_nodes(
    PG,
    pos,
    node_size=node_sizes,
    alpha=0.8
)

nx.draw_networkx_labels(
    PG,
    pos,
    font_size=7
)

plt.title(
    "Most Frequently Documented Giant Pumpkin Parent Pairs"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/top_parent_pair_network_clean.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PART 10
# SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "analysis_start_year": START_YEAR,
    "analysis_end_year": END_YEAR,
    "total_records": len(df),
    "named_parent_offspring_links": len(edges_df),
    "unique_named_parents": edges_df["parent"].nunique(),
    "documented_parent_pairs": len(pair_summary),
    "network_parent_nodes": len(parent_nodes),
    "network_offspring_nodes": len(offspring_nodes),
    "network_edges": G.number_of_edges(),
    "top_1_percent_weight_threshold_lbs": threshold
}])

summary.to_csv(
    f"{OUTPUT_DIR}/network_summary_clean.csv",
    index=False
)


print("\n" + "=" * 75)
print("PHASE 3C COMPLETE")
print("=" * 75)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nImportant files:")

for f in [
    "parent_statistics_clean.csv",
    "parent_persistence_clean.csv",
    "parent_pair_summary_clean.csv",
    "elite_parent_associations_clean.csv",
    "lineage_candidate_parents_clean.csv",
    "network_yearly_parent_reuse_clean.csv",
    "top_20_parent_network_clean.png",
    "top_parent_pair_network_clean.png",
    "network_summary_clean.csv"
]:
    print(" ", f)

print("\nIMPORTANT:")
print(
    "Frequency of a documented parent does NOT prove "
    "genetic dominance or ancestral importance."
)

print(
    "Lineage claims require documented pedigree chains "
    "or independent historical/genomic evidence."
)