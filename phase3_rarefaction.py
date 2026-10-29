import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

# ============================================================
# PHASE 3B
# STANDARDIZED PEDIGREE DIVERSITY / RAREFACTION ANALYSIS
#
# Input:
#   data.csv
#
# Output:
#   phase3_rarefaction/
#       rarefaction_results.csv
#       rarefaction_summary.csv
#       rarefaction_trends.csv
#       graph_unique_parent_ratio.png
#       graph_effective_parents.png
#       graph_simpson_concentration.png
#       graph_top10_share.png
#       graph_rarefaction_mother.png
#       graph_rarefaction_pollinator.png
#
# Primary years:
#   2013-2025
#
# 2026 is excluded because it is a partial/current year.
# ============================================================


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

INPUT_FILE = "data.csv"
OUTPUT_DIR = Path("phase3_rarefaction")

START_YEAR = 2013
END_YEAR = 2025

# Number of random subsamples per year
N_ITERATIONS = 1000

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of records to sample per year.
#
# The script automatically determines the largest common
# sample size available across all years.
#
# You can manually override this if desired.
MANUAL_MOTHER_SAMPLE_SIZE = None
MANUAL_POLLINATOR_SAMPLE_SIZE = None


# ------------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 70)
print("PHASE 3B: STANDARDIZED PEDIGREE ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} records.")

required_columns = [
    "Year",
    "Seed_Mother",
    "Pollinator_Father"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ------------------------------------------------------------
# CLEAN YEAR
# ------------------------------------------------------------

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df = df.dropna(subset=["Year"])

df["Year"] = df["Year"].astype(int)


# ------------------------------------------------------------
# PARENT CLASSIFICATION
# ------------------------------------------------------------

def classify_parent(value):

    if pd.isna(value):
        return "missing"

    value = str(value).strip()

    if value == "":
        return "missing"

    normalized = " ".join(value.split()).lower()

    if normalized in {
        "self",
        "self pollinated",
        "self-pollination",
        "self pollination"
    }:
        return "self"

    if normalized in {
        "open",
        "open pollinated",
        "open-pollinated",
        "open pollination",
        "open pollinated?"
    }:
        return "open"

    if normalized in {
        "unknown",
        "unk",
        "?"
    }:
        return "unknown"

    return "named"


def clean_parent_name(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # Normalize repeated whitespace
    value = " ".join(value.split())

    return value


df["mother_type"] = df["Seed_Mother"].apply(
    classify_parent
)

df["pollinator_type"] = df["Pollinator_Father"].apply(
    classify_parent
)

df["mother_clean"] = df["Seed_Mother"].apply(
    clean_parent_name
)

df["pollinator_clean"] = df["Pollinator_Father"].apply(
    clean_parent_name
)


# ------------------------------------------------------------
# FILTER ANALYSIS PERIOD
# ------------------------------------------------------------

analysis_df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

print(
    f"Analysis period: {START_YEAR}-{END_YEAR}"
)

print(
    f"Records in analysis period: "
    f"{len(analysis_df):,}"
)


# ------------------------------------------------------------
# KEEP ONLY NAMED PARENT RECORDS
# ------------------------------------------------------------

mother_named = analysis_df[
    analysis_df["mother_type"] == "named"
].copy()

pollinator_named = analysis_df[
    analysis_df["pollinator_type"] == "named"
].copy()


# ------------------------------------------------------------
# PRINT YEARLY SAMPLE SIZES
# ------------------------------------------------------------

print("\nNamed parent-linked records by year:")
print("-" * 70)

mother_counts = (
    mother_named
    .groupby("Year")
    .size()
)

pollinator_counts = (
    pollinator_named
    .groupby("Year")
    .size()
)

for year in range(START_YEAR, END_YEAR + 1):

    m = mother_counts.get(year, 0)
    p = pollinator_counts.get(year, 0)

    print(
        f"{year}: "
        f"Mother = {m:,} | "
        f"Pollinator = {p:,}"
    )


# ------------------------------------------------------------
# DETERMINE COMMON SAMPLE SIZE
# ------------------------------------------------------------

mother_min = mother_counts.min()
pollinator_min = pollinator_counts.min()

if MANUAL_MOTHER_SAMPLE_SIZE is None:
    mother_sample_size = int(mother_min)
else:
    mother_sample_size = MANUAL_MOTHER_SAMPLE_SIZE

if MANUAL_POLLINATOR_SAMPLE_SIZE is None:
    pollinator_sample_size = int(pollinator_min)
else:
    pollinator_sample_size = MANUAL_POLLINATOR_SAMPLE_SIZE


if mother_sample_size <= 0:
    raise ValueError(
        "Mother sample size is zero."
    )

if pollinator_sample_size <= 0:
    raise ValueError(
        "Pollinator sample size is zero."
    )


print("\nStandardized sample sizes:")
print("-" * 70)

print(
    f"Mother:      {mother_sample_size:,} "
    f"records/year"
)

print(
    f"Pollinator:  {pollinator_sample_size:,} "
    f"records/year"
)

print(
    f"Iterations:  {N_ITERATIONS:,} per year"
)


# ------------------------------------------------------------
# METRIC FUNCTION
# ------------------------------------------------------------

def calculate_metrics(parent_series):

    parent_series = pd.Series(parent_series).dropna()

    n = len(parent_series)

    if n == 0:
        return {
            "n": 0,
            "unique_parents": 0,
            "unique_parent_ratio": np.nan,
            "simpson_concentration": np.nan,
            "effective_number": np.nan,
            "top1_share": np.nan,
            "top5_share": np.nan,
            "top10_share": np.nan
        }

    counts = parent_series.value_counts()

    unique_parents = len(counts)

    unique_ratio = unique_parents / n

    proportions = counts / n

    # Simpson concentration:
    # sum(p_i^2)
    simpson = np.sum(proportions ** 2)

    # Effective number of parents
    if simpson > 0:
        effective_number = 1 / simpson
    else:
        effective_number = np.nan

    # Parent reuse
    sorted_counts = counts.sort_values(
        ascending=False
    )

    top1_share = (
        sorted_counts.head(1).sum() / n
    )

    top5_share = (
        sorted_counts.head(5).sum() / n
    )

    top10_share = (
        sorted_counts.head(10).sum() / n
    )

    return {
        "n": n,
        "unique_parents": unique_parents,
        "unique_parent_ratio": unique_ratio,
        "simpson_concentration": simpson,
        "effective_number": effective_number,
        "top1_share": top1_share,
        "top5_share": top5_share,
        "top10_share": top10_share
    }


# ------------------------------------------------------------
# RANDOM NUMBER GENERATOR
# ------------------------------------------------------------

rng = np.random.default_rng(
    RANDOM_SEED
)


# ------------------------------------------------------------
# RAREFACTION FUNCTION
# ------------------------------------------------------------

def run_rarefaction(
    data,
    parent_column,
    sample_size,
    parent_type
):

    results = []

    print("\n" + "=" * 70)
    print(
        f"Running standardized analysis: "
        f"{parent_type}"
    )
    print("=" * 70)

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        yearly = data[
            data["Year"] == year
        ][parent_column].dropna()

        yearly = yearly.reset_index(
            drop=True
        )

        if len(yearly) < sample_size:

            raise ValueError(
                f"{parent_type} year {year} "
                f"has only {len(yearly)} records, "
                f"but {sample_size} are required."
            )

        print(
            f"Year {year}: "
            f"{len(yearly):,} available -> "
            f"{sample_size:,} sampled"
        )

        for iteration in range(
            N_ITERATIONS
        ):

            sampled = yearly.sample(
                n=sample_size,
                replace=False,
                random_state=int(
                    rng.integers(
                        0,
                        2**32 - 1
                    )
                )
            )

            metrics = calculate_metrics(
                sampled
            )

            metrics["year"] = year
            metrics["iteration"] = iteration + 1
            metrics["parent_type"] = parent_type

            results.append(metrics)

    return pd.DataFrame(results)


# ------------------------------------------------------------
# RUN MOTHER ANALYSIS
# ------------------------------------------------------------

mother_results = run_rarefaction(
    mother_named,
    "mother_clean",
    mother_sample_size,
    "Mother"
)


# ------------------------------------------------------------
# RUN POLLINATOR ANALYSIS
# ------------------------------------------------------------

pollinator_results = run_rarefaction(
    pollinator_named,
    "pollinator_clean",
    pollinator_sample_size,
    "Pollinator"
)


# ------------------------------------------------------------
# COMBINE RESULTS
# ------------------------------------------------------------

results = pd.concat(
    [
        mother_results,
        pollinator_results
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# SAVE RAW ITERATION RESULTS
# ------------------------------------------------------------

results_file = (
    OUTPUT_DIR /
    "rarefaction_results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print(
    f"\nSaved: {results_file}"
)


# ------------------------------------------------------------
# CALCULATE SUMMARY
# ------------------------------------------------------------

metric_columns = [
    "unique_parents",
    "unique_parent_ratio",
    "simpson_concentration",
    "effective_number",
    "top1_share",
    "top5_share",
    "top10_share"
]

summary_rows = []

for (
    parent_type,
    year
), group in results.groupby(
    ["parent_type", "year"]
):

    row = {
        "parent_type": parent_type,
        "year": year,
        "sample_size": group["n"].iloc[0]
    }

    for metric in metric_columns:

        values = group[metric].values

        row[f"{metric}_mean"] = np.mean(
            values
        )

        row[f"{metric}_sd"] = np.std(
            values,
            ddof=1
        )

        row[f"{metric}_lower95"] = np.percentile(
            values,
            2.5
        )

        row[f"{metric}_upper95"] = np.percentile(
            values,
            97.5
        )

    summary_rows.append(row)


summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR /
    "rarefaction_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"Saved: {summary_file}"
)


# ------------------------------------------------------------
# TREND ANALYSIS
# ------------------------------------------------------------

trend_rows = []

for parent_type in [
    "Mother",
    "Pollinator"
]:

    subset = summary[
        summary["parent_type"] ==
        parent_type
    ].sort_values("year")

    years = subset["year"].values

    for metric in [
        "unique_parent_ratio_mean",
        "simpson_concentration_mean",
        "effective_number_mean",
        "top10_share_mean"
    ]:

        values = subset[metric].values

        rho, p = spearmanr(
            years,
            values
        )

        slope = np.polyfit(
            years,
            values,
            1
        )[0]

        trend_rows.append({
            "parent_type": parent_type,
            "metric": metric,
            "spearman_rho": rho,
            "spearman_p": p,
            "linear_slope_per_year": slope
        })


trends = pd.DataFrame(
    trend_rows
)

trend_file = (
    OUTPUT_DIR /
    "rarefaction_trends.csv"
)

trends.to_csv(
    trend_file,
    index=False
)

print(
    f"Saved: {trend_file}"
)


# ------------------------------------------------------------
# PRINT TREND RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STANDARDIZED TREND RESULTS")
print("=" * 70)

for _, row in trends.iterrows():

    print(
        f"\n{row['parent_type']} | "
        f"{row['metric']}"
    )

    print(
        f"  Spearman rho = "
        f"{row['spearman_rho']:.4f}"
    )

    print(
        f"  p-value      = "
        f"{row['spearman_p']:.6f}"
    )

    print(
        f"  Linear slope = "
        f"{row['linear_slope_per_year']:.6f}/year"
    )


# ============================================================
# GRAPHING
# ============================================================


def plot_metric(
    metric,
    ylabel,
    filename,
    title,
    percentage=False
):

    plt.figure(
        figsize=(10, 6)
    )

    for parent_type in [
        "Mother",
        "Pollinator"
    ]:

        subset = summary[
            summary["parent_type"] ==
            parent_type
        ].sort_values("year")

        x = subset["year"].values

        y = subset[
            f"{metric}_mean"
        ].values

        lower = subset[
            f"{metric}_lower95"
        ].values

        upper = subset[
            f"{metric}_upper95"
        ].values

        if percentage:

            y_plot = y * 100
            lower_plot = lower * 100
            upper_plot = upper * 100

        else:

            y_plot = y
            lower_plot = lower
            upper_plot = upper

        plt.plot(
            x,
            y_plot,
            marker="o",
            linewidth=2,
            label=parent_type
        )

        plt.fill_between(
            x,
            lower_plot,
            upper_plot,
            alpha=0.15
        )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ------------------------------------------------------------
# GRAPH 1
# UNIQUE PARENT RATIO
# ------------------------------------------------------------

plot_metric(
    metric="unique_parent_ratio",
    ylabel="Unique named parents / named parent-linked records (%)",
    filename="graph_unique_parent_ratio.png",
    title="Standardized Pedigree Uniqueness, 2013–2025",
    percentage=True
)


# ------------------------------------------------------------
# GRAPH 2
# EFFECTIVE NUMBER OF PARENTS
# ------------------------------------------------------------

plot_metric(
    metric="effective_number",
    ylabel="Effective number of parents",
    filename="graph_effective_parents.png",
    title="Standardized Effective Number of Parents, 2013–2025"
)


# ------------------------------------------------------------
# GRAPH 3
# SIMPSON CONCENTRATION
# ------------------------------------------------------------

plot_metric(
    metric="simpson_concentration",
    ylabel="Simpson concentration",
    filename="graph_simpson_concentration.png",
    title="Standardized Pedigree Concentration, 2013–2025"
)


# ------------------------------------------------------------
# GRAPH 4
# TOP 10 PARENT SHARE
# ------------------------------------------------------------

plot_metric(
    metric="top10_share",
    ylabel="Records attributed to top 10 parents (%)",
    filename="graph_top10_share.png",
    title="Standardized Top-10 Parent Share, 2013–2025",
    percentage=True
)


# ============================================================
# RAREFACTION CURVES
# ============================================================

def create_rarefaction_curve(
    data,
    parent_column,
    parent_type,
    max_sample_size,
    filename
):

    sample_sizes = np.linspace(
        50,
        max_sample_size,
        15,
        dtype=int
    )

    years_to_plot = [
        START_YEAR,
        2016,
        2019,
        2022,
        2025
    ]

    plt.figure(
        figsize=(10, 6)
    )

    for year in years_to_plot:

        yearly = data[
            data["Year"] == year
        ][parent_column].dropna()

        if len(yearly) < 50:
            continue

        means = []

        for n in sample_sizes:

            if n > len(yearly):
                means.append(np.nan)
                continue

            iteration_values = []

            for _ in range(200):

                sampled = yearly.sample(
                    n=n,
                    replace=False,
                    random_state=int(
                        rng.integers(
                            0,
                            2**32 - 1
                        )
                    )
                )

                unique_count = (
                    sampled.nunique()
                )

                iteration_values.append(
                    unique_count
                )

            means.append(
                np.mean(
                    iteration_values
                )
            )

        plt.plot(
            sample_sizes,
            means,
            marker="o",
            linewidth=1.8,
            label=str(year)
        )

    plt.xlabel(
        "Number of sampled named parent records"
    )

    plt.ylabel(
        "Expected number of unique named parents"
    )

    plt.title(
        f"Pedigree Rarefaction Curve: "
        f"{parent_type}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        title="Year"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        filename
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ------------------------------------------------------------
# MOTHER RAREFACTION CURVE
# ------------------------------------------------------------

create_rarefaction_curve(
    mother_named,
    "mother_clean",
    "Mother",
    mother_sample_size,
    "graph_rarefaction_mother.png"
)


# ------------------------------------------------------------
# POLLINATOR RAREFACTION CURVE
# ------------------------------------------------------------

create_rarefaction_curve(
    pollinator_named,
    "pollinator_clean",
    "Pollinator",
    pollinator_sample_size,
    "graph_rarefaction_pollinator.png"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3B COMPLETE")
print("=" * 70)

print(
    "\nGenerated files:"
)

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  {file.name}"
    )

print(
    "\nThe main files to inspect are:"
)

print(
    "  rarefaction_summary.csv"
)

print(
    "  rarefaction_trends.csv"
)

print(
    "  graph_unique_parent_ratio.png"
)

print(
    "  graph_effective_parents.png"
)

print(
    "  graph_simpson_concentration.png"
)

print(
    "  graph_top10_share.png"
)

print(
    "  graph_rarefaction_mother.png"
)

print(
    "  graph_rarefaction_pollinator.png"
)

print(
    "\nIMPORTANT:"
)

print(
    "These metrics describe pedigree-record diversity/"
)

print(
    "concentration, NOT genomic diversity."
)

print(
    "Do not claim that pedigree data directly measures"
)

print(
    "DNA-level genetic diversity."
)

print(
    "\nDone."
)