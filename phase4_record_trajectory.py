#!/usr/bin/env python3

"""
PHASE 4: RECORD-WEIGHT TRAJECTORY ANALYSIS

Research Question:
Has the competitive giant-pumpkin record-weight trajectory
shown statistically significant deceleration or a plateau?

Uses annual maximum recorded weight, 2013-2025.

Models:
1. Linear regression
2. Quadratic regression
3. Year-to-year record improvement
4. Grid-search piecewise linear regression
5. Spearman trend tests
6. Model comparison using RSS, AIC and BIC
7. Bootstrap confidence intervals for linear and quadratic slopes

Outputs:
phase4_output/
    annual_record_weights.csv
    trajectory_statistics.csv
    year_to_year_improvement.csv
    piecewise_models.csv
    figure_record_trajectory.png
    figure_record_improvement.png
    figure_piecewise_trajectory.png
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = "data.csv"
OUTPUT_DIR = "phase4_output"

START_YEAR = 2013
END_YEAR = 2025

os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def aic(rss, n, k):
    """
    AIC for Gaussian least-squares regression.
    k = number of estimated parameters.
    """
    return n * np.log(rss / n) + 2 * k


def bic(rss, n, k):
    """
    BIC for Gaussian least-squares regression.
    """
    return n * np.log(rss / n) + k * np.log(n)


def fit_linear(x, y):
    """
    Fit y = a + b*x
    """
    result = stats.linregress(x, y)

    intercept = result.intercept
    slope = result.slope

    predicted = intercept + slope * x
    residuals = y - predicted
    rss = np.sum(residuals ** 2)

    return {
        "intercept": intercept,
        "slope": slope,
        "r_squared": result.rvalue ** 2,
        "p_value": result.pvalue,
        "stderr": result.stderr,
        "rss": rss,
        "predicted": predicted,
    }


def bootstrap_linear_slope(x, y, iterations=10000, seed=42):
    """
    Bootstrap the slope of a linear regression by resampling
    year-weight pairs.
    """
    rng = np.random.default_rng(seed)

    slopes = []

    n = len(x)

    for _ in range(iterations):
        indices = rng.integers(0, n, n)

        xb = x[indices]
        yb = y[indices]

        if len(np.unique(xb)) < 2:
            continue

        slope, _, _, _, _ = stats.linregress(xb, yb)
        slopes.append(slope)

    slopes = np.array(slopes)

    return {
        "median": np.median(slopes),
        "lower_95": np.percentile(slopes, 2.5),
        "upper_95": np.percentile(slopes, 97.5),
        "n_bootstrap": len(slopes),
    }


def fit_quadratic(x, y):
    """
    Fit:

        y = a + b*x + c*x^2
    """

    coefficients = np.polyfit(x, y, 2)

    c = coefficients[0]
    b = coefficients[1]
    a = coefficients[2]

    predicted = a + b * x + c * x ** 2

    residuals = y - predicted
    rss = np.sum(residuals ** 2)

    ss_total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - rss / ss_total

    return {
        "intercept": a,
        "linear_coefficient": b,
        "quadratic_coefficient": c,
        "r_squared": r_squared,
        "rss": rss,
        "predicted": predicted,
    }


def bootstrap_quadratic(x, y, iterations=10000, seed=42):
    """
    Bootstrap quadratic coefficient.
    """

    rng = np.random.default_rng(seed)

    coefficients = []

    n = len(x)

    for _ in range(iterations):

        indices = rng.integers(0, n, n)

        xb = x[indices]
        yb = y[indices]

        if len(np.unique(xb)) < 3:
            continue

        coeffs = np.polyfit(xb, yb, 2)

        coefficients.append(coeffs[0])

    coefficients = np.array(coefficients)

    return {
        "median": np.median(coefficients),
        "lower_95": np.percentile(coefficients, 2.5),
        "upper_95": np.percentile(coefficients, 97.5),
        "n_bootstrap": len(coefficients),
    }


def fit_piecewise(x, y, breakpoint):

    """
    Continuous piecewise linear regression.

    y = a + b1*x + b2*max(0, x-breakpoint)

    Before breakpoint:
        slope = b1

    After breakpoint:
        slope = b1 + b2
    """

    hinge = np.maximum(0, x - breakpoint)

    X = np.column_stack([
        np.ones(len(x)),
        x,
        hinge
    ])

    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    intercept = coefficients[0]
    slope_before = coefficients[1]
    slope_change = coefficients[2]

    slope_after = slope_before + slope_change

    predicted = X @ coefficients

    residuals = y - predicted
    rss = np.sum(residuals ** 2)

    return {
        "breakpoint": breakpoint,
        "intercept": intercept,
        "slope_before": slope_before,
        "slope_after": slope_after,
        "slope_change": slope_change,
        "rss": rss,
        "aic": aic(rss, len(x), 3),
        "bic": bic(rss, len(x), 3),
        "predicted": predicted,
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PHASE 4: RECORD-WEIGHT TRAJECTORY ANALYSIS")
print("=" * 75)

print()
print("Research question:")
print("Has the competitive record-weight trajectory")
print("shown statistically significant deceleration or plateau?")
print()


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df):,} records.")

required_columns = [
    "Year",
    "Weight_lbs"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Weight_lbs"] = pd.to_numeric(df["Weight_lbs"], errors="coerce")

df = df.dropna(subset=["Year", "Weight_lbs"])

df["Year"] = df["Year"].astype(int)

df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

print(
    f"Analysis period: {START_YEAR}-{END_YEAR}"
)

print(
    f"Records included: {len(df):,}"
)

print()


# ============================================================
# ANNUAL RECORD WEIGHTS
# ============================================================

print("=" * 75)
print("PART 1: ANNUAL RECORD WEIGHTS")
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

annual = annual.sort_values("Year")

print()
print(annual.to_string(index=False))

annual.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "annual_record_weights.csv"
    ),
    index=False
)

print()


# ============================================================
# PREPARE ARRAYS
# ============================================================

years = annual["Year"].to_numpy(dtype=float)

weights = annual["maximum_weight_lbs"].to_numpy(dtype=float)

n = len(years)

print(f"Number of annual observations: {n}")

if n < 8:
    raise ValueError(
        "Too few annual observations for trajectory analysis."
    )

print()


# ============================================================
# PART 2: LINEAR REGRESSION
# ============================================================

print("=" * 75)
print("PART 2: LINEAR TREND")
print("=" * 75)

linear = fit_linear(years, weights)

print()
print(f"Linear slope: {linear['slope']:.4f} lb/year")
print(f"Intercept: {linear['intercept']:.4f}")
print(f"R²: {linear['r_squared']:.4f}")
print(f"p-value: {linear['p_value']:.6f}")
print(f"Standard error: {linear['stderr']:.4f}")
print(f"RSS: {linear['rss']:.4f}")

print()

if linear["p_value"] < 0.05:
    print("Linear trend: statistically significant.")
else:
    print("Linear trend: not statistically significant.")

print()


# ============================================================
# PART 3: SPEARMAN TREND
# ============================================================

print("=" * 75)
print("PART 3: SPEARMAN MONOTONIC TREND")
print("=" * 75)

spearman_rho, spearman_p = stats.spearmanr(
    years,
    weights
)

print()
print(f"Spearman rho: {spearman_rho:.4f}")
print(f"p-value: {spearman_p:.6f}")

if spearman_p < 0.05:
    print("Monotonic trend: statistically significant.")
else:
    print("Monotonic trend: not statistically significant.")

print()


# ============================================================
# PART 4: QUADRATIC MODEL
# ============================================================

print("=" * 75)
print("PART 4: QUADRATIC TREND")
print("=" * 75)

quadratic = fit_quadratic(years, weights)

print()
print(
    f"Linear coefficient: "
    f"{quadratic['linear_coefficient']:.6f}"
)

print(
    f"Quadratic coefficient: "
    f"{quadratic['quadratic_coefficient']:.6f}"
)

print(
    f"R²: "
    f"{quadratic['r_squared']:.4f}"
)

print(
    f"RSS: "
    f"{quadratic['rss']:.4f}"
)

print()

quadratic_coefficient = quadratic[
    "quadratic_coefficient"
]

if quadratic_coefficient < 0:
    print(
        "Quadratic curvature is negative, "
        "consistent with a downward-curving trajectory."
    )
elif quadratic_coefficient > 0:
    print(
        "Quadratic curvature is positive, "
        "indicating upward acceleration in the fitted curve."
    )
else:
    print("No quadratic curvature detected.")

print()


# ============================================================
# PART 5: MODEL COMPARISON
# ============================================================

print("=" * 75)
print("PART 5: MODEL COMPARISON")
print("=" * 75)

linear_aic = aic(
    linear["rss"],
    n,
    2
)

linear_bic = bic(
    linear["rss"],
    n,
    2
)

quadratic_aic = aic(
    quadratic["rss"],
    n,
    3
)

quadratic_bic = bic(
    quadratic["rss"],
    n,
    3
)

model_comparison = pd.DataFrame([
    {
        "model": "linear",
        "parameters": 2,
        "RSS": linear["rss"],
        "AIC": linear_aic,
        "BIC": linear_bic,
        "R_squared": linear["r_squared"],
    },
    {
        "model": "quadratic",
        "parameters": 3,
        "RSS": quadratic["rss"],
        "AIC": quadratic_aic,
        "BIC": quadratic_bic,
        "R_squared": quadratic["r_squared"],
    }
])

print()
print(
    model_comparison.to_string(index=False)
)

model_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trajectory_model_comparison.csv"
    ),
    index=False
)

print()


# ============================================================
# PART 6: BOOTSTRAP LINEAR SLOPE
# ============================================================

print("=" * 75)
print("PART 6: BOOTSTRAP LINEAR SLOPE")
print("=" * 75)

linear_bootstrap = bootstrap_linear_slope(
    years,
    weights,
    iterations=10000
)

print()
print(
    f"Bootstrap median slope: "
    f"{linear_bootstrap['median']:.4f} lb/year"
)

print(
    f"95% CI: "
    f"[{linear_bootstrap['lower_95']:.4f}, "
    f"{linear_bootstrap['upper_95']:.4f}]"
)

print()


# ============================================================
# PART 7: BOOTSTRAP QUADRATIC CURVATURE
# ============================================================

print("=" * 75)
print("PART 7: BOOTSTRAP QUADRATIC CURVATURE")
print("=" * 75)

quadratic_bootstrap = bootstrap_quadratic(
    years,
    weights,
    iterations=10000
)

print()
print(
    f"Bootstrap median quadratic coefficient: "
    f"{quadratic_bootstrap['median']:.6f}"
)

print(
    f"95% CI: "
    f"[{quadratic_bootstrap['lower_95']:.6f}, "
    f"{quadratic_bootstrap['upper_95']:.6f}]"
)

print()


# ============================================================
# PART 8: YEAR-TO-YEAR RECORD IMPROVEMENT
# ============================================================

print("=" * 75)
print("PART 8: YEAR-TO-YEAR RECORD IMPROVEMENT")
print("=" * 75)

improvement = annual[
    ["Year", "maximum_weight_lbs"]
].copy()

improvement["previous_record_lbs"] = (
    improvement["maximum_weight_lbs"].shift(1)
)

improvement["record_change_lbs"] = (
    improvement["maximum_weight_lbs"]
    - improvement["previous_record_lbs"]
)

improvement["percent_change"] = (
    improvement["record_change_lbs"]
    / improvement["previous_record_lbs"]
) * 100

improvement = improvement.dropna()

print()
print(
    improvement.to_string(index=False)
)

improvement.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_to_year_improvement.csv"
    ),
    index=False
)

print()


# ============================================================
# PART 9: TREND IN YEAR-TO-YEAR IMPROVEMENT
# ============================================================

print("=" * 75)
print("PART 9: TREND IN YEAR-TO-YEAR IMPROVEMENT")
print("=" * 75)

improvement_years = improvement[
    "Year"
].to_numpy(dtype=float)

changes = improvement[
    "record_change_lbs"
].to_numpy(dtype=float)

improvement_linear = fit_linear(
    improvement_years,
    changes
)

print()
print(
    f"Change slope: "
    f"{improvement_linear['slope']:.4f} lb/year"
)

print(
    f"R²: "
    f"{improvement_linear['r_squared']:.4f}"
)

print(
    f"p-value: "
    f"{improvement_linear['p_value']:.6f}"
)

print()

if improvement_linear["slope"] < 0:
    print(
        "Year-to-year record improvements show a "
        "negative fitted trend."
    )
else:
    print(
        "Year-to-year record improvements do not show "
        "a negative fitted trend."
    )

print()


# ============================================================
# PART 10: PIECEWISE LINEAR MODELS
# ============================================================

print("=" * 75)
print("PART 10: PIECEWISE LINEAR CHANGEPOINT SEARCH")
print("=" * 75)

piecewise_results = []

# Avoid endpoints.
candidate_breakpoints = years[2:-2]

for breakpoint in candidate_breakpoints:

    result = fit_piecewise(
        years,
        weights,
        breakpoint
    )

    piecewise_results.append({
        "breakpoint": breakpoint,
        "slope_before": result["slope_before"],
        "slope_after": result["slope_after"],
        "slope_change": result["slope_change"],
        "RSS": result["rss"],
        "AIC": result["aic"],
        "BIC": result["bic"],
    })

piecewise_df = pd.DataFrame(
    piecewise_results
)

piecewise_df = piecewise_df.sort_values(
    "BIC"
)

print()
print(
    piecewise_df.to_string(index=False)
)

piecewise_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "piecewise_models.csv"
    ),
    index=False
)

best_piecewise = piecewise_df.iloc[0]

print()
print("BEST PIECEWISE MODEL")
print("-" * 40)

print(
    f"Breakpoint: "
    f"{best_piecewise['breakpoint']:.0f}"
)

print(
    f"Slope before breakpoint: "
    f"{best_piecewise['slope_before']:.4f} lb/year"
)

print(
    f"Slope after breakpoint: "
    f"{best_piecewise['slope_after']:.4f} lb/year"
)

print(
    f"Slope change: "
    f"{best_piecewise['slope_change']:.4f} lb/year"
)

print(
    f"BIC: "
    f"{best_piecewise['BIC']:.4f}"
)

print()


# ============================================================
# PART 11: SAVE SUMMARY STATISTICS
# ============================================================

trajectory_statistics = pd.DataFrame([
    {
        "metric": "linear_slope_lb_per_year",
        "value": linear["slope"]
    },
    {
        "metric": "linear_p_value",
        "value": linear["p_value"]
    },
    {
        "metric": "linear_R_squared",
        "value": linear["r_squared"]
    },
    {
        "metric": "linear_95CI_lower",
        "value": linear_bootstrap["lower_95"]
    },
    {
        "metric": "linear_95CI_upper",
        "value": linear_bootstrap["upper_95"]
    },
    {
        "metric": "spearman_rho",
        "value": spearman_rho
    },
    {
        "metric": "spearman_p_value",
        "value": spearman_p
    },
    {
        "metric": "quadratic_coefficient",
        "value": quadratic["quadratic_coefficient"]
    },
    {
        "metric": "quadratic_R_squared",
        "value": quadratic["r_squared"]
    },
    {
        "metric": "quadratic_95CI_lower",
        "value": quadratic_bootstrap["lower_95"]
    },
    {
        "metric": "quadratic_95CI_upper",
        "value": quadratic_bootstrap["upper_95"]
    },
    {
        "metric": "best_piecewise_breakpoint",
        "value": best_piecewise["breakpoint"]
    },
    {
        "metric": "piecewise_slope_before",
        "value": best_piecewise["slope_before"]
    },
    {
        "metric": "piecewise_slope_after",
        "value": best_piecewise["slope_after"]
    },
    {
        "metric": "piecewise_slope_change",
        "value": best_piecewise["slope_change"]
    },
])

trajectory_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "trajectory_statistics.csv"
    ),
    index=False
)


# ============================================================
# FIGURE 1: RECORD TRAJECTORY
# ============================================================

print("=" * 75)
print("PART 11: GENERATING FIGURES")
print("=" * 75)

x_plot = np.linspace(
    years.min(),
    years.max(),
    300
)

linear_y = (
    linear["intercept"]
    + linear["slope"] * x_plot
)

quadratic_y = (
    quadratic["intercept"]
    + quadratic["linear_coefficient"] * x_plot
    + quadratic["quadratic_coefficient"] * x_plot ** 2
)

plt.figure(figsize=(12, 7))

plt.scatter(
    years,
    weights,
    s=70,
    label="Annual maximum"
)

plt.plot(
    x_plot,
    linear_y,
    linewidth=2,
    label="Linear regression"
)

plt.plot(
    x_plot,
    quadratic_y,
    linewidth=2,
    label="Quadratic regression"
)

plt.xlabel("Year")
plt.ylabel("Maximum recorded weight (lb)")
plt.title(
    "Competitive Giant Pumpkin Record-Weight Trajectory, 2013–2025"
)

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_record_trajectory.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2: YEAR-TO-YEAR IMPROVEMENT
# ============================================================

plt.figure(figsize=(12, 7))

plt.axhline(
    0,
    linewidth=1
)

plt.bar(
    improvement_years,
    changes,
    width=0.65
)

plt.xlabel("Year")
plt.ylabel("Change from previous annual record (lb)")
plt.title(
    "Year-to-Year Change in Competitive Giant Pumpkin Record"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_record_improvement.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3: BEST PIECEWISE MODEL
# ============================================================

best_breakpoint = best_piecewise["breakpoint"]

best_piecewise_result = fit_piecewise(
    years,
    weights,
    best_breakpoint
)

piecewise_y = (
    best_piecewise_result["predicted"]
)

plt.figure(figsize=(12, 7))

plt.scatter(
    years,
    weights,
    s=70,
    label="Annual maximum"
)

plt.plot(
    years,
    piecewise_y,
    linewidth=2,
    label=f"Piecewise model, breakpoint = {int(best_breakpoint)}"
)

plt.axvline(
    best_breakpoint,
    linestyle="--",
    linewidth=1.5,
    label=f"Candidate breakpoint: {int(best_breakpoint)}"
)

plt.xlabel("Year")
plt.ylabel("Maximum recorded weight (lb)")
plt.title(
    "Piecewise Model of Giant Pumpkin Record-Weight Trajectory"
)

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "figure_piecewise_trajectory.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("PHASE 4 COMPLETE")
print("=" * 75)

print()

print("KEY RESULTS")
print("-" * 40)

print(
    f"Linear slope: "
    f"{linear['slope']:.4f} lb/year"
)

print(
    f"Linear p-value: "
    f"{linear['p_value']:.6f}"
)

print(
    f"Spearman rho: "
    f"{spearman_rho:.4f}"
)

print(
    f"Spearman p-value: "
    f"{spearman_p:.6f}"
)

print(
    f"Quadratic coefficient: "
    f"{quadratic['quadratic_coefficient']:.6f}"
)

print(
    f"Quadratic 95% CI: "
    f"[{quadratic_bootstrap['lower_95']:.6f}, "
    f"{quadratic_bootstrap['upper_95']:.6f}]"
)

print(
    f"Best candidate breakpoint: "
    f"{int(best_breakpoint)}"
)

print(
    f"Slope before breakpoint: "
    f"{best_piecewise['slope_before']:.4f} lb/year"
)

print(
    f"Slope after breakpoint: "
    f"{best_piecewise['slope_after']:.4f} lb/year"
)

print()

print("Files written to:")
print(f"  {OUTPUT_DIR}/")

print()
print("IMPORTANT:")
print(
    "The annual maximum represents a competitive record trajectory, "
    "not the biological growth curve of an individual pumpkin."
)

print(
    "A detected changepoint is a statistical association, "
    "not proof of a biological mechanism."
)

print()
print("Done.")