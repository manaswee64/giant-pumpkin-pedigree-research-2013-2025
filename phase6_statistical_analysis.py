#!/usr/bin/env python3

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data.csv"
OUTPUT_DIR = "phase6_output"

START_YEAR = 2013
END_YEAR = 2025

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("PHASE 6: STATISTICAL SYNTHESIS")
print("=" * 75)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def linear_regression(x, y):
    """
    Ordinary least-squares linear regression.

    Returns:
        slope
        intercept
        r_squared
        p_value
        ci_low
        ci_high
        rss
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept = model.params[0]
    slope = model.params[1]

    ci = model.conf_int(alpha=0.05)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": model.rsquared,
        "p_value": model.pvalues[1],
        "ci_low": ci[1, 0],
        "ci_high": ci[1, 1],
        "rss": np.sum(model.resid ** 2),
    }


def spearman_test(x, y):
    """
    Spearman rank correlation.
    """

    rho, p = spearmanr(x, y)

    return {
        "rho": rho,
        "p_value": p,
    }


def add_test_result(results, analysis, test, estimate, p_value):
    """
    Add a statistical test to the master testing table.
    """

    results.append({
        "analysis": analysis,
        "test": test,
        "estimate": estimate,
        "p_value": p_value,
    })


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Weight_lbs"] = pd.to_numeric(df["Weight_lbs"], errors="coerce")

df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

df = df.dropna(subset=["Year", "Weight_lbs"])

df["Year"] = df["Year"].astype(int)

print(f"Records included: {len(df):,}")


# ============================================================
# PART 1: BUILD ANNUAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PART 1: BUILDING ANNUAL SUMMARY")
print("=" * 75)

annual = (
    df.groupby("Year")
    .agg(
        records=("Weight_lbs", "size"),
        maximum_weight_lbs=("Weight_lbs", "max"),
        mean_weight_lbs=("Weight_lbs", "mean"),
        median_weight_lbs=("Weight_lbs", "median"),
    )
    .reset_index()
)

annual.to_csv(
    os.path.join(OUTPUT_DIR, "annual_summary.csv"),
    index=False
)

print(annual.to_string(index=False))


# ============================================================
# PART 2: RECORD WEIGHT TREND
# ============================================================

print("\n" + "=" * 75)
print("PART 2: RECORD WEIGHT TREND")
print("=" * 75)

years = annual["Year"].values
record_weights = annual["maximum_weight_lbs"].values

record_linear = linear_regression(
    years,
    record_weights
)

record_spearman = spearman_test(
    years,
    record_weights
)

print(
    f"\nLinear slope: "
    f"{record_linear['slope']:.4f} lb/year"
)

print(
    f"95% CI: "
    f"[{record_linear['ci_low']:.4f}, "
    f"{record_linear['ci_high']:.4f}]"
)

print(
    f"R²: "
    f"{record_linear['r_squared']:.4f}"
)

print(
    f"p-value: "
    f"{record_linear['p_value']:.8f}"
)

print(
    f"\nSpearman rho: "
    f"{record_spearman['rho']:.4f}"
)

print(
    f"Spearman p-value: "
    f"{record_spearman['p_value']:.8f}"
)


# ============================================================
# PART 3: MEAN WEIGHT TREND
# ============================================================

print("\n" + "=" * 75)
print("PART 3: MEAN WEIGHT TREND")
print("=" * 75)

mean_weights = annual["mean_weight_lbs"].values

mean_linear = linear_regression(
    years,
    mean_weights
)

mean_spearman = spearman_test(
    years,
    mean_weights
)

print(
    f"\nLinear slope: "
    f"{mean_linear['slope']:.4f} lb/year"
)

print(
    f"95% CI: "
    f"[{mean_linear['ci_low']:.4f}, "
    f"{mean_linear['ci_high']:.4f}]"
)

print(
    f"R²: "
    f"{mean_linear['r_squared']:.4f}"
)

print(
    f"p-value: "
    f"{mean_linear['p_value']:.8f}"
)

print(
    f"\nSpearman rho: "
    f"{mean_spearman['rho']:.4f}"
)

print(
    f"Spearman p-value: "
    f"{mean_spearman['p_value']:.8f}"
)


# ============================================================
# PART 4: YEAR-TO-YEAR RECORD IMPROVEMENT
# ============================================================

print("\n" + "=" * 75)
print("PART 4: YEAR-TO-YEAR RECORD IMPROVEMENT")
print("=" * 75)

record_improvement = annual.copy()

record_improvement["previous_record_lbs"] = (
    record_improvement["maximum_weight_lbs"].shift(1)
)

record_improvement["record_change_lbs"] = (
    record_improvement["maximum_weight_lbs"]
    - record_improvement["previous_record_lbs"]
)

record_improvement["percent_change"] = (
    record_improvement["record_change_lbs"]
    / record_improvement["previous_record_lbs"]
    * 100
)

record_improvement = record_improvement.dropna(
    subset=["record_change_lbs"]
).copy()

record_improvement.to_csv(
    os.path.join(OUTPUT_DIR, "record_improvement.csv"),
    index=False
)

improvement_years = record_improvement["Year"].values
improvements = record_improvement["record_change_lbs"].values

improvement_linear = linear_regression(
    improvement_years,
    improvements
)

improvement_spearman = spearman_test(
    improvement_years,
    improvements
)

print(
    f"\nSlope of yearly improvement: "
    f"{improvement_linear['slope']:.4f} lb/year"
)

print(
    f"95% CI: "
    f"[{improvement_linear['ci_low']:.4f}, "
    f"{improvement_linear['ci_high']:.4f}]"
)

print(
    f"R²: "
    f"{improvement_linear['r_squared']:.4f}"
)

print(
    f"p-value: "
    f"{improvement_linear['p_value']:.8f}"
)

print(
    f"\nSpearman rho: "
    f"{improvement_spearman['rho']:.4f}"
)

print(
    f"Spearman p-value: "
    f"{improvement_spearman['p_value']:.8f}"
)


# ============================================================
# PART 5: GEOGRAPHIC EXPANSION
# ============================================================

print("\n" + "=" * 75)
print("PART 5: GEOGRAPHIC EXPANSION")
print("=" * 75)


def geographic_counts(data):
    """
    Calculate geographic representation by year.
    """

    output = []

    for year, group in data.groupby("Year"):

        countries = group["Country"].replace(
            ["", "Unknown", "unknown", "nan", "NaN"],
            np.nan
        ).dropna().nunique()

        states = group["State_Province"].replace(
            ["", "Unknown", "unknown", "nan", "NaN"],
            np.nan
        ).dropna().nunique()

        cities = group["City"].replace(
            ["", "Unknown", "unknown", "nan", "NaN"],
            np.nan
        ).dropna().nunique()

        sites = group["GPC_Site"].replace(
            ["", "Unknown", "unknown", "nan", "NaN"],
            np.nan
        ).dropna().nunique()

        output.append({
            "Year": year,
            "countries": countries,
            "states_provinces": states,
            "cities": cities,
            "gpc_sites": sites
        })

    return pd.DataFrame(output)


geo = geographic_counts(df)

geo.to_csv(
    os.path.join(OUTPUT_DIR, "geographic_expansion.csv"),
    index=False
)

geo_results = []

for metric in [
    "countries",
    "states_provinces",
    "cities",
    "gpc_sites"
]:

    x = geo["Year"].values
    y = geo[metric].values

    linear = linear_regression(x, y)
    spear = spearman_test(x, y)

    geo_results.append({
        "metric": metric,
        "slope": linear["slope"],
        "ci_low": linear["ci_low"],
        "ci_high": linear["ci_high"],
        "r_squared": linear["r_squared"],
        "p_linear": linear["p_value"],
        "spearman_rho": spear["rho"],
        "spearman_p": spear["p_value"]
    })

geo_statistics = pd.DataFrame(geo_results)

geo_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "geographic_expansion_statistics.csv"
    ),
    index=False
)

print(geo_statistics.to_string(index=False))


# ============================================================
# PART 6: COUNTRY CONCENTRATION
# ============================================================

print("\n" + "=" * 75)
print("PART 6: COUNTRY CONCENTRATION")
print("=" * 75)


def country_concentration(data):
    """
    Calculate yearly country concentration.

    HHI:
        sum of squared country shares.

    Effective number:
        1 / HHI
    """

    rows = []

    for year, group in data.groupby("Year"):

        countries = (
            group["Country"]
            .astype(str)
            .str.strip()
        )

        countries = countries[
            ~countries.isin([
                "",
                "nan",
                "NaN",
                "Unknown",
                "unknown"
            ])
        ]

        counts = countries.value_counts()

        shares = counts / counts.sum()

        hhi = np.sum(shares ** 2)

        effective = (
            1 / hhi
            if hhi > 0
            else np.nan
        )

        top1 = (
            shares.head(1).sum() * 100
        )

        top3 = (
            shares.head(3).sum() * 100
        )

        top5 = (
            shares.head(5).sum() * 100
        )

        rows.append({
            "Year": year,
            "country_hhi": hhi,
            "effective_countries": effective,
            "top_country_share_pct": top1,
            "top_3_country_share_pct": top3,
            "top_5_country_share_pct": top5,
            "countries": len(counts)
        })

    return pd.DataFrame(rows)


country_conc = country_concentration(df)

country_conc.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "country_concentration_reanalysis.csv"
    ),
    index=False
)

country_results = []

for metric in [
    "country_hhi",
    "effective_countries",
    "top_country_share_pct",
    "top_3_country_share_pct",
    "top_5_country_share_pct"
]:

    x = country_conc["Year"].values
    y = country_conc[metric].values

    linear = linear_regression(x, y)
    spear = spearman_test(x, y)

    country_results.append({
        "metric": metric,
        "slope": linear["slope"],
        "ci_low": linear["ci_low"],
        "ci_high": linear["ci_high"],
        "r_squared": linear["r_squared"],
        "p_linear": linear["p_value"],
        "spearman_rho": spear["rho"],
        "spearman_p": spear["p_value"]
    })

country_statistics = pd.DataFrame(country_results)

country_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "country_concentration_statistics.csv"
    ),
    index=False
)

print(country_statistics.to_string(index=False))


# ============================================================
# PART 7: EARLY VS LATE COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("PART 7: EARLY VS LATE COMPARISON")
print("=" * 75)

EARLY_START = 2013
EARLY_END = 2017

LATE_START = 2021
LATE_END = 2025

early = df[
    (df["Year"] >= EARLY_START) &
    (df["Year"] <= EARLY_END)
]

late = df[
    (df["Year"] >= LATE_START) &
    (df["Year"] <= LATE_END)
]

early_counts = early["Country"].value_counts()
late_counts = late["Country"].value_counts()

early_total = early_counts.sum()
late_total = late_counts.sum()

all_countries = sorted(
    set(early_counts.index) |
    set(late_counts.index)
)

comparison = []

for country in all_countries:

    early_share = (
        early_counts.get(country, 0)
        / early_total
        * 100
    )

    late_share = (
        late_counts.get(country, 0)
        / late_total
        * 100
    )

    comparison.append({
        "country": country,
        "early_share_pct": early_share,
        "late_share_pct": late_share,
        "change_percentage_points":
            late_share - early_share
    })

early_late = pd.DataFrame(comparison)

early_late = early_late.sort_values(
    "change_percentage_points",
    ascending=False
)

early_late.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "early_vs_late_country_comparison.csv"
    ),
    index=False
)

print(
    "\nLargest increases in share:"
)

print(
    early_late.head(15).to_string(index=False)
)

print(
    "\nLargest decreases in share:"
)

print(
    early_late.tail(15)
    .sort_values(
        "change_percentage_points"
    )
    .to_string(index=False)
)


# ============================================================
# PART 8: MULTIPLE TESTING SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PART 8: MULTIPLE TESTING SUMMARY")
print("=" * 75)

tests = []

# ------------------------------------------------------------
# Record maximum
# ------------------------------------------------------------

add_test_result(
    tests,
    "Annual record maximum",
    "Linear regression",
    record_linear["slope"],
    record_linear["p_value"]
)

add_test_result(
    tests,
    "Annual record maximum",
    "Spearman correlation",
    record_spearman["rho"],
    record_spearman["p_value"]
)

# ------------------------------------------------------------
# Mean weight
# ------------------------------------------------------------

add_test_result(
    tests,
    "Annual mean weight",
    "Linear regression",
    mean_linear["slope"],
    mean_linear["p_value"]
)

add_test_result(
    tests,
    "Annual mean weight",
    "Spearman correlation",
    mean_spearman["rho"],
    mean_spearman["p_value"]
)

# ------------------------------------------------------------
# Year-to-year improvement
# ------------------------------------------------------------

add_test_result(
    tests,
    "Year-to-year record change",
    "Linear regression",
    improvement_linear["slope"],
    improvement_linear["p_value"]
)

add_test_result(
    tests,
    "Year-to-year record change",
    "Spearman correlation",
    improvement_spearman["rho"],
    improvement_spearman["p_value"]
)

# ------------------------------------------------------------
# Geographic expansion
# ------------------------------------------------------------

for _, row in geo_statistics.iterrows():

    add_test_result(
        tests,
        f"Geographic {row['metric']}",
        "Linear regression",
        row["slope"],
        row["p_linear"]
    )

    add_test_result(
        tests,
        f"Geographic {row['metric']}",
        "Spearman correlation",
        row["spearman_rho"],
        row["spearman_p"]
    )

# ------------------------------------------------------------
# Country concentration
# ------------------------------------------------------------

for _, row in country_statistics.iterrows():

    add_test_result(
        tests,
        f"Country {row['metric']}",
        "Linear regression",
        row["slope"],
        row["p_linear"]
    )

    add_test_result(
        tests,
        f"Country {row['metric']}",
        "Spearman correlation",
        row["spearman_rho"],
        row["spearman_p"]
    )


tests_df = pd.DataFrame(tests)

# ------------------------------------------------------------
# CORRECT BENJAMINI-HOCHBERG FDR CORRECTION
# ------------------------------------------------------------

reject, fdr_adjusted, _, _ = multipletests(
    tests_df["p_value"].values,
    alpha=0.05,
    method="fdr_bh"
)

tests_df["fdr_adjusted_p"] = fdr_adjusted

tests_df["significant_p05"] = (
    tests_df["p_value"] < 0.05
)

tests_df["significant_fdr05"] = reject

# Sort by raw p-value for readability
tests_df = tests_df.sort_values(
    "p_value"
).reset_index(drop=True)

tests_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "statistical_tests_summary.csv"
    ),
    index=False
)

print(
    tests_df.to_string(index=False)
)


# ============================================================
# PART 9: FINAL STATISTICAL SYNTHESIS
# ============================================================

print("\n" + "=" * 75)
print("PART 9: FINAL SYNTHESIS")
print("=" * 75)

synthesis = pd.DataFrame([
    {
        "research_question":
            "RQ2: Record trajectory",

        "metric":
            "Annual maximum weight",

        "estimate":
            record_linear["slope"],

        "ci_low":
            record_linear["ci_low"],

        "ci_high":
            record_linear["ci_high"],

        "p_value":
            record_linear["p_value"],

        "interpretation":
            "Positive temporal trend"
    },

    {
        "research_question":
            "RQ2: Record trajectory",

        "metric":
            "Year-to-year improvement",

        "estimate":
            improvement_linear["slope"],

        "ci_low":
            improvement_linear["ci_low"],

        "ci_high":
            improvement_linear["ci_high"],

        "p_value":
            improvement_linear["p_value"],

        "interpretation":
            "No statistically significant trend in annual improvement"
    },

    {
        "research_question":
            "RQ3: Geographic expansion",

        "metric":
            "Number of countries",

        "estimate":
            geo_statistics.loc[
                geo_statistics["metric"] == "countries",
                "slope"
            ].iloc[0],

        "ci_low":
            geo_statistics.loc[
                geo_statistics["metric"] == "countries",
                "ci_low"
            ].iloc[0],

        "ci_high":
            geo_statistics.loc[
                geo_statistics["metric"] == "countries",
                "ci_high"
            ].iloc[0],

        "p_value":
            geo_statistics.loc[
                geo_statistics["metric"] == "countries",
                "p_linear"
            ].iloc[0],

        "interpretation":
            "Geographic representation increased"
    },

    {
        "research_question":
            "RQ3: Geographic concentration",

        "metric":
            "Country HHI",

        "estimate":
            country_statistics.loc[
                country_statistics["metric"] == "country_hhi",
                "slope"
            ].iloc[0],

        "ci_low":
            country_statistics.loc[
                country_statistics["metric"] == "country_hhi",
                "ci_low"
            ].iloc[0],

        "ci_high":
            country_statistics.loc[
                country_statistics["metric"] == "country_hhi",
                "ci_high"
            ].iloc[0],

        "p_value":
            country_statistics.loc[
                country_statistics["metric"] == "country_hhi",
                "p_linear"
            ].iloc[0],

        "interpretation":
            "Country concentration decreased"
    }
])

synthesis.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_statistical_synthesis.csv"
    ),
    index=False
)

print(
    synthesis.to_string(index=False)
)


# ============================================================
# PART 10: FIGURES
# ============================================================

print("\n" + "=" * 75)
print("PART 10: GENERATING FIGURES")
print("=" * 75)


# ------------------------------------------------------------
# Figure 1: Record trajectory
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    years,
    record_weights,
    label="Annual maximum"
)

x_line = np.linspace(
    years.min(),
    years.max(),
    200
)

y_line = (
    record_linear["intercept"]
    + record_linear["slope"] * x_line
)

plt.plot(
    x_line,
    y_line,
    label="Linear trend"
)

plt.xlabel("Year")
plt.ylabel("Maximum weight (lb)")
plt.title("Competitive Giant Pumpkin Record Trajectory")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "record_weight_trajectory.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Figure 2: Annual mean weight
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    years,
    mean_weights,
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Mean weight (lb)")
plt.title("Mean Recorded Pumpkin Weight by Year")
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "mean_weight_trajectory.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Figure 3: Year-to-year improvement
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    improvement_years,
    improvements
)

plt.axhline(
    0,
    linewidth=1
)

plt.xlabel("Year")
plt.ylabel("Change from previous year's maximum (lb)")
plt.title("Year-to-Year Record Weight Change")
plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "record_improvement.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Figure 4: Geographic expansion
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for metric in [
    "countries",
    "states_provinces",
    "gpc_sites"
]:

    plt.plot(
        geo["Year"],
        geo[metric],
        marker="o",
        label=metric.replace("_", " ").title()
    )

plt.xlabel("Year")
plt.ylabel("Number of geographic units")
plt.title("Geographic Representation Over Time")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "geographic_expansion_synthesis.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Figure 5: Country concentration
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    country_conc["Year"],
    country_conc["country_hhi"],
    marker="o",
    label="Country HHI"
)

plt.xlabel("Year")
plt.ylabel("HHI")
plt.title("Country Concentration Over Time")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "country_concentration_synthesis.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("PHASE 6 COMPLETE")
print("=" * 75)

print("\nMain outputs:")
print("  annual_summary.csv")
print("  record_improvement.csv")
print("  geographic_expansion.csv")
print("  geographic_expansion_statistics.csv")
print("  country_concentration_reanalysis.csv")
print("  country_concentration_statistics.csv")
print("  early_vs_late_country_comparison.csv")
print("  statistical_tests_summary.csv")
print("  final_statistical_synthesis.csv")

print("\nFigures:")
print("  record_weight_trajectory.png")
print("  mean_weight_trajectory.png")
print("  record_improvement.png")
print("  geographic_expansion_synthesis.png")
print("  country_concentration_synthesis.png")

print("\nKey results:")
print(
    f"  Record slope: "
    f"{record_linear['slope']:.4f} lb/year"
)

print(
    f"  Record slope p-value: "
    f"{record_linear['p_value']:.8f}"
)

print(
    f"  Record Spearman rho: "
    f"{record_spearman['rho']:.4f}"
)

print(
    f"  Record Spearman p-value: "
    f"{record_spearman['p_value']:.8f}"
)

print(
    f"  Year-to-year improvement slope: "
    f"{improvement_linear['slope']:.4f} lb/year"
)

print(
    f"  Year-to-year improvement p-value: "
    f"{improvement_linear['p_value']:.8f}"
)

print(
    f"  Countries/year: "
    f"{geo_statistics.loc[geo_statistics['metric'] == 'countries', 'slope'].iloc[0]:.4f}"
)

print(
    f"  Country HHI/year: "
    f"{country_statistics.loc[country_statistics['metric'] == 'country_hhi', 'slope'].iloc[0]:.6f}"
)

print("\nImportant:")
print(
    "Annual maximum weight represents a competitive record trajectory, "
    "not the biological growth curve of an individual pumpkin."
)

print(
    "Geographic results measure representation in the recorded dataset "
    "and do not establish climate causation."
)

print(
    "Statistical significance does not establish a biological mechanism."
)

print("\nDone.")