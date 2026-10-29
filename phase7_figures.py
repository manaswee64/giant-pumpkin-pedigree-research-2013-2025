#!/usr/bin/env python3

"""
PHASE 7: PUBLICATION-QUALITY FIGURES
Giant Pumpkin Computational Analysis

Generates publication-quality figures for the IRIS/ISEF research paper.

Input:
    data.csv
    phase6_output/

Output:
    phase7_figures/
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data.csv"
PHASE6_DIR = "phase6_output"
OUTPUT_DIR = "phase7_figures"

START_YEAR = 2013
END_YEAR = 2025

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication-style defaults
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
# LOAD DATA
# ============================================================

print("=" * 75)
print("PHASE 7: PUBLICATION-QUALITY FIGURES")
print("=" * 75)

print("\nLoading data...")

df = pd.read_csv(INPUT_FILE)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Weight_lbs"] = pd.to_numeric(df["Weight_lbs"], errors="coerce")

df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR) &
    (df["Weight_lbs"] > 0)
].copy()

df["Year"] = df["Year"].astype(int)

print(f"Records included: {len(df):,}")

# ============================================================
# ANNUAL SUMMARY
# ============================================================

annual = (
    df.groupby("Year")["Weight_lbs"]
    .agg(
        records="count",
        maximum_weight_lbs="max",
        mean_weight_lbs="mean",
        median_weight_lbs="median"
    )
    .reset_index()
)

annual["record_improvement"] = annual["maximum_weight_lbs"].diff()

years = annual["Year"].values

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_figure(filename):
    path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {path}")


def regression_line(x, y):
    result = linregress(x, y)

    predicted = result.intercept + result.slope * x

    return result, predicted


# ============================================================
# FIGURE 1
# RECORD WEIGHT TRAJECTORY
# ============================================================

print("\n[1/7] Record-weight trajectory...")

x = years
y = annual["maximum_weight_lbs"].values

result, predicted = regression_line(x, y)

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    s=55,
    zorder=3,
    label="Annual maximum"
)

plt.plot(
    x,
    predicted,
    linewidth=2,
    label=f"Linear trend ({result.slope:.1f} lb/year)"
)

plt.xlabel("Year")
plt.ylabel("Annual maximum pumpkin weight (lb)")
plt.title("Competitive Record-Weight Trajectory, 2013–2025")

plt.xticks(years[::2])

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    frameon=False
)

save_figure("figure1_record_weight_trajectory.png")


# ============================================================
# FIGURE 2
# MEAN WEIGHT TRAJECTORY
# ============================================================

print("\n[2/7] Mean-weight trajectory...")

y = annual["mean_weight_lbs"].values

result, predicted = regression_line(x, y)

plt.figure(figsize=(10, 6))

plt.scatter(
    x,
    y,
    s=50,
    zorder=3,
    label="Annual mean"
)

plt.plot(
    x,
    predicted,
    linewidth=2,
    label=f"Linear trend ({result.slope:.1f} lb/year)"
)

plt.xlabel("Year")
plt.ylabel("Mean pumpkin weight (lb)")
plt.title("Mean Recorded Pumpkin Weight, 2013–2025")

plt.xticks(years[::2])

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    frameon=False
)

save_figure("figure2_mean_weight_trajectory.png")


# ============================================================
# FIGURE 3
# YEAR-TO-YEAR RECORD IMPROVEMENT
# ============================================================

print("\n[3/7] Year-to-year record improvement...")

improvement = annual.dropna(subset=["record_improvement"])

x_imp = improvement["Year"].values
y_imp = improvement["record_improvement"].values

result, predicted = regression_line(x_imp, y_imp)

plt.figure(figsize=(10, 6))

plt.axhline(
    0,
    linewidth=1
)

plt.scatter(
    x_imp,
    y_imp,
    s=55,
    zorder=3
)

plt.plot(
    x_imp,
    predicted,
    linewidth=2,
    label=f"Trend ({result.slope:.1f} lb/year)"
)

plt.xlabel("Year")
plt.ylabel("Change in annual record (lb)")
plt.title("Year-to-Year Change in Competitive Record Weight")

plt.xticks(years[::2])

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    frameon=False
)

save_figure("figure3_record_improvement.png")


# ============================================================
# FIGURE 4
# GEOGRAPHIC EXPANSION
# ============================================================

print("\n[4/7] Geographic expansion...")

geo = pd.DataFrame({
    "Year": years,

    "Countries": [
        df[df["Year"] == year]["Country"].dropna().nunique()
        for year in years
    ],

    "States / Provinces": [
        df[df["Year"] == year]["State_Province"].dropna().nunique()
        for year in years
    ],

    "GPC Sites": [
        df[df["Year"] == year]["GPC_Site"].dropna().nunique()
        for year in years
    ],
})

plt.figure(figsize=(10, 6))

plt.plot(
    geo["Year"],
    geo["Countries"],
    marker="o",
    linewidth=2,
    label="Countries"
)

plt.plot(
    geo["Year"],
    geo["States / Provinces"],
    marker="o",
    linewidth=2,
    label="States / provinces"
)

plt.plot(
    geo["Year"],
    geo["GPC Sites"],
    marker="o",
    linewidth=2,
    label="GPC sites"
)

plt.xlabel("Year")
plt.ylabel("Number represented")
plt.title("Geographic Representation in the Recorded Dataset")

plt.xticks(years[::2])

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    frameon=False
)

save_figure("figure4_geographic_expansion.png")


# ============================================================
# FIGURE 5
# COUNTRY CONCENTRATION
# ============================================================

print("\n[5/7] Country concentration...")

hhi_values = []
effective_values = []
top1_values = []

for year in years:

    subset = df[df["Year"] == year]

    counts = (
        subset["Country"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    counts = counts[counts != ""]

    shares = counts.value_counts(normalize=True)

    hhi = (shares ** 2).sum()

    hhi_values.append(hhi)
    effective_values.append(1 / hhi if hhi > 0 else np.nan)
    top1_values.append(shares.iloc[0] * 100 if len(shares) else np.nan)

fig, ax1 = plt.subplots(figsize=(10, 6))

line1 = ax1.plot(
    years,
    hhi_values,
    marker="o",
    linewidth=2,
    label="Country HHI"
)

ax1.set_xlabel("Year")
ax1.set_ylabel("Country concentration (HHI)")

ax1.set_xticks(years[::2])

ax1.grid(
    axis="y",
    alpha=0.25
)

ax2 = ax1.twinx()

line2 = ax2.plot(
    years,
    effective_values,
    marker="s",
    linewidth=2,
    label="Effective number of countries"
)

ax2.set_ylabel("Effective number of countries")

lines = line1 + line2
labels = [line.get_label() for line in lines]

ax1.legend(
    lines,
    labels,
    frameon=False,
    loc="center right"
)

plt.title("Changing Country Concentration, 2013–2025")

save_figure("figure5_country_concentration.png")


# ============================================================
# FIGURE 6
# TOP COUNTRIES OVER TIME
# ============================================================

print("\n[6/7] Country-share trajectories...")

country_counts = (
    df["Country"]
    .dropna()
    .astype(str)
    .str.strip()
)

country_counts = country_counts[country_counts != ""]

top_countries = (
    country_counts
    .value_counts()
    .head(5)
    .index
    .tolist()
)

plt.figure(figsize=(10, 6))

for country in top_countries:

    shares = []

    for year in years:

        subset = df[df["Year"] == year]

        total = len(subset)

        if total == 0:
            shares.append(np.nan)
            continue

        count = (
            subset["Country"]
            .astype(str)
            .str.strip()
            .eq(country)
            .sum()
        )

        shares.append(
            100 * count / total
        )

    plt.plot(
        years,
        shares,
        marker="o",
        linewidth=2,
        label=country
    )

plt.xlabel("Year")
plt.ylabel("Share of recorded pumpkins (%)")
plt.title("Shares of the Five Most Represented Countries")

plt.xticks(years[::2])

plt.grid(
    axis="y",
    alpha=0.25
)

plt.legend(
    frameon=False
)

save_figure("figure6_top_country_shares.png")


# ============================================================
# FIGURE 7
# RECORD VS MEAN WEIGHT
# ============================================================

print("\n[7/7] Record versus mean weight...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    years,
    annual["maximum_weight_lbs"],
    marker="o",
    linewidth=2,
    label="Annual maximum"
)

ax.plot(
    years,
    annual["mean_weight_lbs"],
    marker="o",
    linewidth=2,
    label="Annual mean"
)

ax.set_xlabel("Year")
ax.set_ylabel("Weight (lb)")
ax.set_title("Annual Maximum and Mean Pumpkin Weight")

ax.set_xticks(years[::2])

ax.grid(
    axis="y",
    alpha=0.25
)

ax.legend(
    frameon=False
)

save_figure("figure7_record_vs_mean.png")


# ============================================================
# SAVE GEO DATA
# ============================================================

geo.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "geographic_annual_summary.csv"
    ),
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PHASE 7 COMPLETE")
print("=" * 75)

print("\nGenerated figures:")

for filename in sorted(os.listdir(OUTPUT_DIR)):

    if filename.endswith(".png"):
        print(f"  {filename}")

print("\nAll figures saved at 300 DPI.")

print("\nRecommended figures for the final paper:")

print("""
FIGURE 1
Record-weight trajectory

FIGURE 2
Year-to-year record improvement

FIGURE 3
Geographic expansion

FIGURE 4
Country concentration

FIGURE 5
Top-country representation

FIGURE 6
Record vs mean weight
""")

print("\nDone.")