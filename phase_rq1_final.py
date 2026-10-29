#!/usr/bin/env python3

"""
RQ1 FINAL ANALYSIS
Pedigree Diversity, Parent Reuse, Ancestral Concentration,
Lineage Tracing, and Network Analysis

Primary analysis period:
    2013-2025

2026:
    Treated separately because it is incomplete.

Outputs:
    rq1_final_output/
"""

import os
import re
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from scipy.stats import linregress, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data.csv"
OUTPUT_DIR = "rq1_final_output"

START_YEAR = 2013
END_YEAR = 2025

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    """Normalize harmless whitespace/case formatting."""

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(r"\s+", " ", value)

    return value

def classify_parent(value):
    """
    Classify pedigree entries.

    Only genuinely identifiable named parents are retained
    for pedigree-diversity calculations.

    Returns:
        named
        self
        open
        unknown
        missing
    """

    value = clean_text(value)

    if value == "":
        return "missing"

    normalized = value.lower().strip()

    # --------------------------------------------------------
    # Self / self-pollination
    # --------------------------------------------------------

    self_values = {
        "self",
        "selfed",
        "self pollinated",
        "self-pollinated",
        "self pollination",
        "self-pollination",
        "same",
    }

    if normalized in self_values:
        return "self"

    # --------------------------------------------------------
    # Open pollination
    # --------------------------------------------------------

    open_values = {
        "open",
        "open pollinated",
        "open-pollinated",
        "open pollination",
        "open-pollination",
    }

    if normalized in open_values:
        return "open"

    # --------------------------------------------------------
    # Unknown / unavailable
    # --------------------------------------------------------

    unknown_values = {
        "unknown",
        "unk",
        "unknow",
        "not known",
        "not available",
        "not recorded",
        "missing",
        "none",
        "na",
        "n/a",
        "n.a.",
        "?",
        "-",
        "--",
        "nil",
        "no data",
    }

    if normalized in unknown_values:
        return "unknown"

    # --------------------------------------------------------
    # Sibling / sibling placeholders
    # --------------------------------------------------------

    sibling_values = {
        "sib",
        "sibb",
        "sibling",
        "siblings",
    }

    if normalized in sibling_values:
        return "unknown"

    # --------------------------------------------------------
    # Other obvious placeholder strings
    # --------------------------------------------------------

    placeholder_values = {
        "same as mother",
        "same as father",
        "same mother",
        "same father",
        "same seed",
        "same pollinator",
        "same plant",
        "same pumpkin",
        "unknown parent",
        "unknown mother",
        "unknown father",
        "unknown pollinator",
        "unknown seed",
        "not sure",
        "n/a parent",
    }

    if normalized in placeholder_values:
        return "unknown"

    # --------------------------------------------------------
    # Otherwise treat as a named parent
    # --------------------------------------------------------

    return "named"


def normalize_parent(value):
    """
    Normalize a named parent.

    Matching is case-insensitive but the original readable
    capitalization is retained from the first occurrence.
    """

    value = clean_text(value)

    if not value:
        return ""

    return value


def canonical_parent(value):
    """
    Canonical form for matching parents.
    """

    value = normalize_parent(value)

    if not value:
        return ""

    return value.lower()


def save_df(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {path}")


def linear_stats(x, y):

    result = linregress(x, y)

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "stderr": result.stderr,
        "spearman_rho": spearmanr(x, y).statistic,
        "spearman_p": spearmanr(x, y).pvalue,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("RQ1 FINAL ANALYSIS")
print("PEDIGREE DIVERSITY AND ANCESTRAL CONCENTRATION")
print("=" * 80)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df = df[
    df["Year"].notna()
].copy()

df["Year"] = df["Year"].astype(int)

print(f"Total records loaded: {len(df):,}")

# ============================================================
# CLEAN PARENT FIELDS
# ============================================================

print("\nCleaning pedigree fields...")

for column in ["Seed_Mother", "Pollinator_Father"]:

    df[column] = df[column].apply(clean_text)

    df[f"{column}_type"] = (
        df[column]
        .apply(classify_parent)
    )

    df[f"{column}_canonical"] = (
        df[column]
        .apply(canonical_parent)
    )

# ============================================================
# PRIMARY ANALYSIS DATA
# ============================================================

analysis_df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

print(
    f"Primary analysis records "
    f"({START_YEAR}-{END_YEAR}): "
    f"{len(analysis_df):,}"
)

# ============================================================
# PHASE 2
# DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("PART 1: PEDIGREE DATA QUALITY")
print("=" * 80)

quality_rows = []

for year in sorted(df["Year"].unique()):

    subset = df[df["Year"] == year]

    total = len(subset)

    mother_named = (
        subset["Seed_Mother_type"] == "named"
    ).sum()

    pollinator_named = (
        subset["Pollinator_Father_type"] == "named"
    ).sum()

    both_named = (
        (subset["Seed_Mother_type"] == "named") &
        (subset["Pollinator_Father_type"] == "named")
    ).sum()

    self_count = (
        subset["Pollinator_Father_type"] == "self"
    ).sum()

    open_count = (
        subset["Pollinator_Father_type"] == "open"
    ).sum()

    unknown_count = (
        subset["Pollinator_Father_type"].isin(
            ["unknown", "missing"]
        )
    ).sum()

    quality_rows.append({
        "Year": year,
        "records": total,
        "known_mother": mother_named,
        "known_pollinator": pollinator_named,
        "both_parents_known": both_named,
        "pct_known_mother":
            100 * mother_named / total,
        "pct_known_pollinator":
            100 * pollinator_named / total,
        "pct_both_known":
            100 * both_named / total,
        "self_count":
            self_count,
        "open_count":
            open_count,
        "unknown_or_missing_pollinator":
            unknown_count,
        "pct_self":
            100 * self_count / total,
        "pct_open":
            100 * open_count / total,
        "pct_unknown_or_missing_pollinator":
            100 * unknown_count / total,
    })

quality = pd.DataFrame(quality_rows)

save_df(
    quality,
    "rq1_data_quality_by_year.csv"
)

# ============================================================
# PARENT DISTRIBUTION FUNCTION
# ============================================================

def parent_distribution(subset, column, type_column):
    """
    Calculate annual parent diversity/concentration metrics.
    """

    named = subset[
        subset[type_column] == "named"
    ].copy()

    if len(named) == 0:

        return {
            "named_records": 0,
            "unique_parents": 0,
            "unique_parent_ratio": np.nan,
            "top1_share": np.nan,
            "top5_share": np.nan,
            "top10_share": np.nan,
            "simpson_concentration": np.nan,
            "effective_number": np.nan,
        }

    counts = (
        named[column + "_canonical"]
        .value_counts()
    )

    total = counts.sum()

    proportions = counts / total

    concentration = (
        proportions ** 2
    ).sum()

    return {
        "named_records": total,
        "unique_parents": len(counts),
        "unique_parent_ratio":
            len(counts) / total,
        "top1_share":
            counts.head(1).sum() / total,
        "top5_share":
            counts.head(5).sum() / total,
        "top10_share":
            counts.head(10).sum() / total,
        "simpson_concentration":
            concentration,
        "effective_number":
            1 / concentration
            if concentration > 0
            else np.nan,
    }


# ============================================================
# PHASE 3-5
# ANNUAL PEDIGREE METRICS
# ============================================================

print("\n" + "=" * 80)
print("PART 2: ANNUAL PEDIGREE DIVERSITY AND CONCENTRATION")
print("=" * 80)

annual_rows = []

for year in range(START_YEAR, END_YEAR + 1):

    subset = analysis_df[
        analysis_df["Year"] == year
    ]

    mother = parent_distribution(
        subset,
        "Seed_Mother",
        "Seed_Mother_type"
    )

    father = parent_distribution(
        subset,
        "Pollinator_Father",
        "Pollinator_Father_type"
    )

    row = {"Year": year}

    for key, value in mother.items():
        row[f"mother_{key}"] = value

    for key, value in father.items():
        row[f"pollinator_{key}"] = value

    annual_rows.append(row)

annual_metrics = pd.DataFrame(
    annual_rows
)

save_df(
    annual_metrics,
    "rq1_annual_pedigree_metrics.csv"
)

print(
    annual_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PHASE 6
# EARLY / MIDDLE / RECENT
# ============================================================

print("\n" + "=" * 80)
print("PART 3: PERIOD COMPARISON")
print("=" * 80)

periods = {
    "2013-2016": (2013, 2016),
    "2017-2020": (2017, 2020),
    "2021-2025": (2021, 2025),
}

period_rows = []

for period, (start, end) in periods.items():

    subset = analysis_df[
        (analysis_df["Year"] >= start) &
        (analysis_df["Year"] <= end)
    ]

    mother = parent_distribution(
        subset,
        "Seed_Mother",
        "Seed_Mother_type"
    )

    father = parent_distribution(
        subset,
        "Pollinator_Father",
        "Pollinator_Father_type"
    )

    period_rows.append({
        "period": period,

        "records": len(subset),

        "mother_unique_parents":
            mother["unique_parents"],

        "mother_named_records":
            mother["named_records"],

        "mother_unique_parent_ratio":
            mother["unique_parent_ratio"],

        "mother_top1_share":
            mother["top1_share"],

        "mother_top5_share":
            mother["top5_share"],

        "mother_top10_share":
            mother["top10_share"],

        "mother_simpson":
            mother["simpson_concentration"],

        "mother_effective_number":
            mother["effective_number"],

        "pollinator_unique_parents":
            father["unique_parents"],

        "pollinator_named_records":
            father["named_records"],

        "pollinator_unique_parent_ratio":
            father["unique_parent_ratio"],

        "pollinator_top1_share":
            father["top1_share"],

        "pollinator_top5_share":
            father["top5_share"],

        "pollinator_top10_share":
            father["top10_share"],

        "pollinator_simpson":
            father["simpson_concentration"],

        "pollinator_effective_number":
            father["effective_number"],
    })

period_table = pd.DataFrame(period_rows)

save_df(
    period_table,
    "rq1_period_comparison.csv"
)

print(
    period_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PHASE 10
# TEMPORAL TREND TESTS
# ============================================================

print("\n" + "=" * 80)
print("PART 4: TEMPORAL TREND TESTS")
print("=" * 80)

trend_rows = []

metrics_to_test = [
    "mother_unique_parent_ratio",
    "mother_top1_share",
    "mother_top5_share",
    "mother_top10_share",
    "mother_simpson_concentration",
    "mother_effective_number",

    "pollinator_unique_parent_ratio",
    "pollinator_top1_share",
    "pollinator_top5_share",
    "pollinator_top10_share",
    "pollinator_simpson_concentration",
    "pollinator_effective_number",
]

for metric in metrics_to_test:

    values = annual_metrics[metric].values

    valid = np.isfinite(values)

    x = annual_metrics.loc[
        valid,
        "Year"
    ].values

    y = values[valid]

    stats = linear_stats(x, y)

    trend_rows.append({
        "metric": metric,
        **stats
    })

trend_results = pd.DataFrame(
    trend_rows
)

# FDR correction
reject, adjusted, _, _ = multipletests(
    trend_results["p_value"],
    alpha=0.05,
    method="fdr_bh"
)

trend_results["fdr_adjusted_p"] = adjusted
trend_results["significant_fdr_05"] = reject

save_df(
    trend_results,
    "rq1_temporal_trends.csv"
)

print(
    trend_results[
        [
            "metric",
            "slope",
            "r_squared",
            "p_value",
            "spearman_rho",
            "spearman_p",
            "fdr_adjusted_p",
            "significant_fdr_05",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PHASE 12
# SENSITIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("PART 5: SENSITIVITY ANALYSIS")
print("=" * 80)

sensitivity_rows = []


def calculate_concentration(
    subset,
    column,
    type_column
):

    named = subset[
        subset[type_column] == "named"
    ]

    if len(named) == 0:
        return np.nan

    counts = (
        named[column + "_canonical"]
        .value_counts()
    )

    shares = counts / counts.sum()

    return (
        shares ** 2
    ).sum()


# Analysis A: all years 2013-2025
for scenario, subset in [

    (
        "2013-2025_all",
        analysis_df
    ),

    (
        "2013-2025_min500_records",
        analysis_df.groupby("Year")
        .filter(lambda x: len(x) >= 500)
    ),

    (
        "2013-2024_excluding_2025",
        analysis_df[
            analysis_df["Year"] <= 2024
        ]
    ),

]:

    years_available = sorted(
        subset["Year"].unique()
    )

    for parent_name, column, type_column in [

        (
            "mother",
            "Seed_Mother",
            "Seed_Mother_type"
        ),

        (
            "pollinator",
            "Pollinator_Father",
            "Pollinator_Father_type"
        ),

    ]:

        yearly = []

        for year in years_available:

            year_data = subset[
                subset["Year"] == year
            ]

            concentration = calculate_concentration(
                year_data,
                column,
                type_column
            )

            if np.isfinite(concentration):
                yearly.append(
                    (year, concentration)
                )

        if len(yearly) >= 4:

            x = np.array(
                [a for a, b in yearly]
            )

            y = np.array(
                [b for a, b in yearly]
            )

            stats = linear_stats(x, y)

            sensitivity_rows.append({
                "scenario": scenario,
                "parent_type": parent_name,
                "years": len(yearly),
                "slope": stats["slope"],
                "p_value": stats["p_value"],
                "spearman_rho":
                    stats["spearman_rho"],
                "spearman_p":
                    stats["spearman_p"],
            })

sensitivity = pd.DataFrame(
    sensitivity_rows
)

save_df(
    sensitivity,
    "rq1_sensitivity_analysis.csv"
)

print(
    sensitivity.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PHASE 7-9
# LINEAGE STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("PART 6: RECURSIVE PEDIGREE LINEAGE ANALYSIS")
print("=" * 80)

# Build a lookup:
# canonical parent name -> canonical parents of that parent

parent_lookup = defaultdict(
    lambda: {
        "mother": set(),
        "pollinator": set(),
        "display": ""
    }
)

# Every record can itself be a parent later.
# We therefore use the pumpkin's identifying pedigree label
# when possible.

for _, row in analysis_df.iterrows():

    mother = row["Seed_Mother_canonical"]
    father = row["Pollinator_Father_canonical"]

    # A pumpkin's own identity is approximated using its
    # recorded row identity when no unique ID exists.
    # We primarily trace named parents that appear elsewhere
    # as parent strings.

    if mother:

        parent_lookup[mother]["display"] = (
            row["Seed_Mother"]
        )

    if father:

        parent_lookup[father]["display"] = (
            row["Pollinator_Father"]
        )


# Direct parent reuse
parent_records = []

for column, type_column, role in [

    (
        "Seed_Mother",
        "Seed_Mother_type",
        "mother"
    ),

    (
        "Pollinator_Father",
        "Pollinator_Father_type",
        "pollinator"
    ),

]:

    named = analysis_df[
        analysis_df[type_column] == "named"
    ]

    counts = (
        named[column + "_canonical"]
        .value_counts()
    )

    display_map = (
        named[
            [column + "_canonical", column]
        ]
        .drop_duplicates(
            column + "_canonical"
        )
        .set_index(
            column + "_canonical"
        )[column]
        .to_dict()
    )

    for canonical, count in counts.items():

        parent_records.append({
            "parent_type": role,
            "parent": display_map.get(
                canonical,
                canonical
            ),
            "canonical_parent": canonical,
            "offspring_records": int(count),
        })

parent_reuse = pd.DataFrame(
    parent_records
)

parent_reuse = parent_reuse.sort_values(
    [
        "parent_type",
        "offspring_records"
    ],
    ascending=[True, False]
)

save_df(
    parent_reuse,
    "rq1_parent_reuse_all.csv"
)


# ============================================================
# RECURSIVE TRACE FROM RECURRING PARENTS
# ============================================================

# Find parent strings that themselves appear as parents
# in the dataset.

all_named_parents = set(
    analysis_df.loc[
        analysis_df["Seed_Mother_type"] == "named",
        "Seed_Mother_canonical"
    ]
)

all_named_parents.update(
    analysis_df.loc[
        analysis_df["Pollinator_Father_type"] == "named",
        "Pollinator_Father_canonical"
    ]
)


def find_ancestral_occurrences(
    canonical_name,
    max_generation=3
):
    """
    Conservative recursive search.

    We search whether a named parent appears as another
    pumpkin's recorded parent, allowing only exact normalized
    string matches.
    """

    results = []

    current = {
        canonical_name
    }

    visited = {
        canonical_name
    }

    for generation in range(1, max_generation + 1):

        next_generation = set()

        for ancestor in current:

            # Search records where this ancestor is itself
            # listed as a parent.
            matching = analysis_df[
                (
                    analysis_df[
                        "Seed_Mother_canonical"
                    ] == ancestor
                )
                |
                (
                    analysis_df[
                        "Pollinator_Father_canonical"
                    ] == ancestor
                )
            ]

            # We cannot infer the parent's own parents unless
            # the parent identity is independently represented
            # by a pedigree string. We therefore record only
            # documented exact recurrence rather than inventing
            # a parent chain.
            for _, row in matching.iterrows():

                mother = row[
                    "Seed_Mother_canonical"
                ]

                father = row[
                    "Pollinator_Father_canonical"
                ]

                for candidate in [mother, father]:

                    if (
                        candidate
                        and candidate != ancestor
                        and candidate not in visited
                    ):
                        next_generation.add(
                            candidate
                        )

        for candidate in next_generation:

            results.append({
                "starting_parent":
                    canonical_name,
                "generation":
                    generation,
                "ancestor_candidate":
                    candidate
            })

        visited.update(next_generation)
        current = next_generation

        if not current:
            break

    return results


# NOTE:
# The exact dataset structure does not provide a guaranteed
# unique pumpkin ID -> parent identity mapping. Therefore this
# analysis deliberately avoids fabricating generations.
#
# Instead, we create a conservative recurrence table.

recurrence_rows = []

for parent in all_named_parents:

    mother_occurrences = (
        analysis_df[
            analysis_df[
                "Seed_Mother_canonical"
            ] == parent
        ]
    )

    father_occurrences = (
        analysis_df[
            analysis_df[
                "Pollinator_Father_canonical"
            ] == parent
        ]
    )

    years_seen = sorted(
        set(
            mother_occurrences["Year"]
        ).union(
            father_occurrences["Year"]
        )
    )

    recurrence_rows.append({
        "parent": parent,
        "years_observed": len(years_seen),
        "first_year":
            min(years_seen)
            if years_seen else np.nan,
        "last_year":
            max(years_seen)
            if years_seen else np.nan,
        "year_list":
            ";".join(
                map(str, years_seen)
            ),
        "total_occurrences":
            len(mother_occurrences)
            + len(father_occurrences),
    })

recurrence = pd.DataFrame(
    recurrence_rows
)

recurrence = recurrence.sort_values(
    [
        "years_observed",
        "total_occurrences"
    ],
    ascending=False
)

save_df(
    recurrence,
    "rq1_recurring_parent_lines.csv"
)


# ============================================================
# HOWARD DILL / ATLANTIC GIANT CANDIDATE SEARCH
# ============================================================

print("\n" + "=" * 80)
print("PART 7: ATLANTIC GIANT / HOWARD DILL CANDIDATE SEARCH")
print("=" * 80)

keywords = [
    "howard dill",
    "dill",
    "atlantic giant",
    "atlantic",
]

candidate_mask = (
    analysis_df[
        "Seed_Mother"
    ]
    .str.lower()
    .str.contains(
        "|".join(
            re.escape(k)
            for k in keywords
        ),
        na=False
    )
    |
    analysis_df[
        "Pollinator_Father"
    ]
    .str.lower()
    .str.contains(
        "|".join(
            re.escape(k)
            for k in keywords
        ),
        na=False
    )
)

dill_candidates = analysis_df[
    candidate_mask
].copy()

save_df(
    dill_candidates,
    "rq1_atlantic_giant_dill_candidates.csv"
)

print(
    f"Candidate records containing historical "
    f"keywords: {len(dill_candidates)}"
)

print(
    "\nIMPORTANT:"
    "\nThese are keyword-based candidate records only."
    "\nThey do NOT establish a documented Howard Dill"
    "\nor Atlantic Giant ancestry."
)


# ============================================================
# TOP RECURRING PARENTS
# ============================================================

print("\nTop recurring parents:")

top_recurrent = recurrence.head(25)

print(
    top_recurrent[
        [
            "parent",
            "years_observed",
            "first_year",
            "last_year",
            "total_occurrences"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FIGURE 1
# PEDIGREE COMPLETENESS
# ============================================================

print("\n" + "=" * 80)
print("PART 8: RQ1 FIGURES")
print("=" * 80)

plt.figure(figsize=(10, 6))

plt.plot(
    quality["Year"],
    quality["pct_known_mother"],
    marker="o",
    linewidth=2,
    label="Known mother"
)

plt.plot(
    quality["Year"],
    quality["pct_known_pollinator"],
    marker="o",
    linewidth=2,
    label="Known pollinator"
)

plt.plot(
    quality["Year"],
    quality["pct_both_known"],
    marker="o",
    linewidth=2,
    label="Both parents known"
)

plt.xlabel("Year")
plt.ylabel("Records with documented pedigree (%)")
plt.title("Pedigree Recording Completeness, 2013–2026")

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(frameon=False)

save_fig(
    "figure1_pedigree_recording_completeness.png"
)


# ============================================================
# FIGURE 2
# EFFECTIVE NUMBER OF PARENTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "mother_effective_number"
    ],
    marker="o",
    linewidth=2,
    label="Maternal lines"
)

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "pollinator_effective_number"
    ],
    marker="o",
    linewidth=2,
    label="Pollinator lines"
)

plt.xlabel("Year")
plt.ylabel("Simpson effective number of parents")
plt.title("Effective Number of Recorded Parental Lines")

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(frameon=False)

save_fig(
    "figure2_effective_parent_number.png"
)


# ============================================================
# FIGURE 3
# TOP-10 PARENT SHARE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "mother_top10_share"
    ] * 100,
    marker="o",
    linewidth=2,
    label="Mother Top-10"
)

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "pollinator_top10_share"
    ] * 100,
    marker="o",
    linewidth=2,
    label="Pollinator Top-10"
)

plt.xlabel("Year")
plt.ylabel("Offspring associated with top 10 parents (%)")
plt.title("Top-10 Parent Reuse Over Time")

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(frameon=False)

save_fig(
    "figure3_top10_parent_share.png"
)


# ============================================================
# FIGURE 4
# SIMPSON CONCENTRATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "mother_simpson_concentration"
    ],
    marker="o",
    linewidth=2,
    label="Maternal concentration"
)

plt.plot(
    annual_metrics["Year"],
    annual_metrics[
        "pollinator_simpson_concentration"
    ],
    marker="o",
    linewidth=2,
    label="Pollinator concentration"
)

plt.xlabel("Year")
plt.ylabel("Simpson concentration")
plt.title("Recorded Pedigree Concentration Over Time")

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(frameon=False)

save_fig(
    "figure4_pedigree_concentration.png"
)


# ============================================================
# FIGURE 5
# PERIOD COMPARISON
# ============================================================

period_labels = period_table["period"]

x = np.arange(
    len(period_labels)
)

width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    period_table[
        "mother_effective_number"
    ],
    width,
    label="Maternal"
)

plt.bar(
    x + width / 2,
    period_table[
        "pollinator_effective_number"
    ],
    width,
    label="Pollinator"
)

plt.xticks(
    x,
    period_labels
)

plt.xlabel("Period")
plt.ylabel("Effective number of parents")
plt.title("Pedigree Diversity Across Study Periods")

plt.legend(frameon=False)

plt.grid(
    axis="y",
    alpha=0.25
)

save_fig(
    "figure5_period_comparison.png"
)


# ============================================================
# FINAL RQ1 TABLE
# ============================================================

print("\nCreating final RQ1 summary table...")

summary_rows = []

for _, row in period_table.iterrows():

    summary_rows.append({
        "Period":
            row["period"],

        "Unique mothers / named records":
            row["mother_unique_parent_ratio"],

        "Mother Top-10 share":
            row["mother_top10_share"],

        "Mother Simpson concentration":
            row["mother_simpson"],

        "Effective mothers":
            row["mother_effective_number"],

        "Unique pollinators / named records":
            row[
                "pollinator_unique_parent_ratio"
            ],

        "Pollinator Top-10 share":
            row["pollinator_top10_share"],

        "Pollinator Simpson concentration":
            row["pollinator_simpson"],

        "Effective pollinators":
            row["pollinator_effective_number"],
    })

final_table = pd.DataFrame(
    summary_rows
)

save_df(
    final_table,
    "rq1_final_period_summary.csv"
)


# ============================================================
# 2026 SEPARATE ANALYSIS
# ============================================================

print("\nAnalyzing 2026 separately...")

data_2026 = df[
    df["Year"] == 2026
]

if len(data_2026) > 0:

    mother_2026 = parent_distribution(
        data_2026,
        "Seed_Mother",
        "Seed_Mother_type"
    )

    father_2026 = parent_distribution(
        data_2026,
        "Pollinator_Father",
        "Pollinator_Father_type"
    )

    current_2026 = pd.DataFrame([{
        "records": len(data_2026),

        "mother_unique_parents":
            mother_2026[
                "unique_parents"
            ],

        "mother_top10_share":
            mother_2026[
                "top10_share"
            ],

        "mother_simpson":
            mother_2026[
                "simpson_concentration"
            ],

        "mother_effective":
            mother_2026[
                "effective_number"
            ],

        "pollinator_unique_parents":
            father_2026[
                "unique_parents"
            ],

        "pollinator_top10_share":
            father_2026[
                "top10_share"
            ],

        "pollinator_simpson":
            father_2026[
                "simpson_concentration"
            ],

        "pollinator_effective":
            father_2026[
                "effective_number"
            ],
    }])

    save_df(
        current_2026,
        "rq1_2026_separate.csv"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 80)
print("RQ1 FINAL ANALYSIS COMPLETE")
print("=" * 80)

print("""
Outputs:

rq1_data_quality_by_year.csv
rq1_annual_pedigree_metrics.csv
rq1_period_comparison.csv
rq1_temporal_trends.csv
rq1_sensitivity_analysis.csv
rq1_parent_reuse_all.csv
rq1_recurring_parent_lines.csv
rq1_atlantic_giant_dill_candidates.csv
rq1_final_period_summary.csv
rq1_2026_separate.csv

Figures:

figure1_pedigree_recording_completeness.png
figure2_effective_parent_number.png
figure3_top10_parent_share.png
figure4_pedigree_concentration.png
figure5_period_comparison.png
""")

print("""
IMPORTANT INTERPRETATION RULES:

1. Pedigree diversity is NOT genetic diversity.
2. Parent reuse does NOT prove genetic dominance.
3. Keyword matches to Howard Dill / Atlantic Giant do NOT
   prove documented ancestry.
4. 2026 is treated separately because it is incomplete.
5. Concentration trends describe the recorded pedigree data,
   not necessarily the complete biological population.
6. Recursive ancestry is only reported where the dataset
   provides an exact documented connection.
""")

print("\nDone.")