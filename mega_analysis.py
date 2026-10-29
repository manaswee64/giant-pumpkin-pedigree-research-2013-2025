#!/usr/bin/env python3

"""
GIANT PUMPKIN MEGA ANALYSIS
===========================

Run:
    python mega_analysis.py

Or:
    python mega_analysis.py data.csv

Outputs:
    mega_analysis_output/
        tables/
        figures/
        analysis_summary.txt
"""

import sys
import re
import math
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from scipy import stats

warnings.filterwarnings("ignore")

# Optional statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (
        acorr_ljungbox,
        het_breuschpagan,
    )
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

OUTPUT_DIR = Path("mega_analysis_output")
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

PRIMARY_START = 2013
PRIMARY_END = 2025
SEPARATE_YEAR = 2026

BOOTSTRAP_N = 3000
CHANGEPOINT_BOOTSTRAP_N = 500
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# Aesthetic palette
BG = "#0d1117"
PANEL = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"

BLUE = "#58a6ff"
CYAN = "#39d0d8"
GREEN = "#3fb950"
PURPLE = "#bc8cff"
PINK = "#f778ba"
ORANGE = "#ffa657"
YELLOW = "#d29922"
RED = "#ff7b72"

sns.set_theme(style="darkgrid")


# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def save_table(df, filename):
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_fig(fig, filename):
    path = FIG_DIR / filename

    fig.patch.set_facecolor(BG)

    for ax in fig.axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

        for spine in ax.spines.values():
            spine.set_color(GRID)

        ax.grid(
            True,
            color=GRID,
            alpha=0.45,
            linewidth=0.8
        )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        facecolor=BG
    )
    plt.close(fig)

    return path


def norm_text(value):
    if pd.isna(value):
        return ""

    s = str(value)

    # Handles the literal * characters in the pasted CSV.
    s = s.strip().strip("*").strip()

    s = re.sub(r"\s+", " ", s)

    return s


def norm_key(value):
    s = norm_text(value).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def numeric_clean(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("*", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def as_array(x):
    """
    Compatibility helper.

    Works for:
        pandas Series
        pandas DataFrame
        numpy arrays
        statsmodels prediction arrays
    """
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())
    return np.asarray(x)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_jarque_bera(resid):
    """
    Compatible with multiple SciPy versions.

    Some versions return:
        SignificanceResult(statistic=..., pvalue=...)

    Others may behave like:
        (statistic, pvalue)
    """
    result = stats.jarque_bera(np.asarray(resid, dtype=float))

    if hasattr(result, "statistic"):
        return float(result.statistic), float(result.pvalue)

    return float(result[0]), float(result[1])


def safe_skew(x):
    try:
        return float(stats.skew(x, bias=False))
    except Exception:
        return np.nan


def safe_kurtosis(x):
    try:
        return float(stats.kurtosis(x, fisher=True, bias=False))
    except Exception:
        return np.nan


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

COLUMN_ALIASES = {
    "year": [
        "year"
    ],
    "place": [
        "place"
    ],
    "weight_lbs": [
        "weight_lbs",
        "weight lbs",
        "weight"
    ],
    "weight_kg": [
        "weight_kg",
        "weight kg"
    ],
    "grower_name": [
        "grower_name",
        "grower name",
        "grower"
    ],
    "city": [
        "city"
    ],
    "state_province": [
        "state_province",
        "state province",
        "state",
        "province"
    ],
    "country": [
        "country"
    ],
    "gpc_site": [
        "gpc_site",
        "gpc site"
    ],
    "seed_mother": [
        "seed_mother",
        "seed mother",
        "mother"
    ],
    "pollinator_father": [
        "pollinator_father",
        "pollinator father",
        "father",
        "pollinator"
    ],
    "ott": [
        "ott"
    ],
    "estimated_weight": [
        "estimated_weight",
        "estimated weight"
    ],
    "pct_chart": [
        "pct_chart",
        "pct chart",
        "percent chart",
        "% chart"
    ],
}


def normalize_columns(df):
    original = list(df.columns)

    lookup = {
        norm_key(c): c
        for c in original
    }

    rename = {}

    for canonical, aliases in COLUMN_ALIASES.items():

        found = None

        for alias in aliases:
            k = norm_key(alias)

            if k in lookup:
                found = lookup[k]
                break

        if found is not None:
            rename[found] = canonical

    df = df.rename(columns=rename)

    required = [
        "year",
        "weight_lbs",
        "grower_name",
        "city",
        "state_province",
        "country",
        "gpc_site",
        "seed_mother",
        "pollinator_father",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        print("\nERROR: Missing columns:")
        for c in missing:
            print("  ", c)

        print("\nColumns found:")
        for c in df.columns:
            print("  ", repr(c))

        raise SystemExit(1)

    return df


# ============================================================
# COUNTRY CLEANING
# ============================================================

COUNTRY_ALIASES = {
    "usa": "United States",
    "u s a": "United States",
    "us": "United States",
    "u s": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "america": "United States",

    "canada": "Canada",

    "uk": "United Kingdom",
    "u k": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",

    "germany": "Germany",
    "deutschland": "Germany",

    "italy": "Italy",
    "italia": "Italy",

    "austria": "Austria",

    "belgium": "Belgium",
    "belgique": "Belgium",

    "japan": "Japan",

    "spain": "Spain",
    "espana": "Spain",

    "portugal": "Portugal",

    "poland": "Poland",
    "polska": "Poland",

    "south africa": "South Africa",

    "australia": "Australia",

    "new zealand": "New Zealand",

    "france": "France",

    "netherlands": "Netherlands",
    "the netherlands": "Netherlands",
    "holland": "Netherlands",

    "switzerland": "Switzerland",

    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",

    "slovakia": "Slovakia",

    "slovenia": "Slovenia",

    "ireland": "Ireland",

    "denmark": "Denmark",

    "sweden": "Sweden",

    "norway": "Norway",

    "finland": "Finland",

    "mexico": "Mexico",

    "brazil": "Brazil",

    "argentina": "Argentina",

    "chile": "Chile",

    "china": "China",

    "south korea": "South Korea",
    "korea": "South Korea",

    "taiwan": "Taiwan",

    "india": "India",

    "croatia": "Croatia",

    "hungary": "Hungary",

    "romania": "Romania",

    "bulgaria": "Bulgaria",

    "serbia": "Serbia",

    "estonia": "Estonia",

    "latvia": "Latvia",

    "lithuania": "Lithuania",

    "ukraine": "Ukraine",

    "russia": "Russia",
    "russian federation": "Russia",

    "turkey": "Turkey",

    "israel": "Israel",

    "iceland": "Iceland",

    "luxembourg": "Luxembourg",

    "malta": "Malta",

    "cyprus": "Cyprus",

    "greece": "Greece",
}


US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california",
    "colorado", "connecticut", "delaware", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas",
    "kentucky", "louisiana", "maine", "maryland", "massachusetts",
    "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
    "district of columbia",
}

US_ABBREVIATIONS = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"
}

CANADA_PROVINCES = {
    "ontario",
    "quebec",
    "nova scotia",
    "new brunswick",
    "manitoba",
    "british columbia",
    "prince edward island",
    "saskatchewan",
    "alberta",
    "newfoundland and labrador",
    "newfoundland",
    "yukon",
    "northwest territories",
    "nunavut",
}

CANADA_ABBREVIATIONS = {
    "on", "qc", "ns", "nb", "mb", "bc",
    "pe", "sk", "ab", "nl", "yt", "nt", "nu"
}

BAD_COUNTRIES = {
    "",
    "unknown",
    "unknown country",
    "n a",
    "na",
    "n/a",
    "none",
    "null",
    "nan",
    "-",
    "--",
    "?",
}


def clean_country(raw, state_province=""):
    raw_text = norm_text(raw)
    k = norm_key(raw_text)

    if k in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[k]

    if k in CANADA_PROVINCES:
        return "Canada"

    if k in CANADA_ABBREVIATIONS:
        return "Canada"

    if k in US_STATES:
        return "United States"

    if k in US_ABBREVIATIONS:
        return "United States"

    if k in BAD_COUNTRIES:

        state_key = norm_key(state_province)

        if state_key in CANADA_PROVINCES:
            return "Canada"

        if state_key in CANADA_ABBREVIATIONS:
            return "Canada"

        if state_key in US_STATES:
            return "United States"

        if state_key in US_ABBREVIATIONS:
            return "United States"

        return "Unknown"

    return raw_text


# ============================================================
# PARENT CLASSIFICATION
# ============================================================

def classify_parent(value):
    k = norm_key(value)

    if k in BAD_COUNTRIES:
        return "missing"

    if k in {
        "open",
        "open pollination",
        "open pollinated",
        "open pollinated seed",
        "op",
    }:
        return "open"

    if "open poll" in k:
        return "open"

    if k in {
        "same",
        "self",
        "self pollinated",
        "self pollination",
    }:
        return "self"

    if k in {
        "sib",
        "sibs",
        "sibling",
        "siblings",
        "sibb",
    }:
        return "sibling"

    if k.startswith("sib "):
        return "sibling"

    return "named"


# ============================================================
# BASIC DATA CLEANING
# ============================================================

def load_data():
    print("=" * 70)
    print("GIANT PUMPKIN MEGA ANALYSIS")
    print("=" * 70)
    print()
    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False
    )

    print(f"Rows loaded: {len(df):,}")

    df = normalize_columns(df)

    # Clean text fields
    text_cols = [
        "grower_name",
        "city",
        "state_province",
        "country",
        "gpc_site",
        "seed_mother",
        "pollinator_father",
        "place",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(norm_text)

    # Numeric fields
    for col in [
        "year",
        "weight_lbs",
        "weight_kg",
        "ott",
        "estimated_weight",
        "pct_chart",
    ]:
        if col in df.columns:
            df[col] = numeric_clean(df[col])

    df["year"] = df["year"].astype("Int64")

    df["country_clean"] = [
        clean_country(c, s)
        for c, s in zip(
            df["country"],
            df["state_province"]
        )
    ]

    df["mother_class"] = df["seed_mother"].map(
        classify_parent
    )

    df["pollinator_class"] = df["pollinator_father"].map(
        classify_parent
    )

    # Remove records with no usable year
    df = df[df["year"].notna()].copy()

    df["year"] = df["year"].astype(int)

    primary = df[
        (df["year"] >= PRIMARY_START) &
        (df["year"] <= PRIMARY_END)
    ].copy()

    separate = df[
        df["year"] == SEPARATE_YEAR
    ].copy()

    print()
    print(
        f"Primary records ({PRIMARY_START}-{PRIMARY_END}): "
        f"{len(primary):,}"
    )

    print(
        f"{SEPARATE_YEAR} records kept separately: "
        f"{len(separate):,}"
    )

    return df, primary, separate


# ============================================================
# DATA QUALITY
# ============================================================

def data_quality(primary, separate):
    rows = []

    all_data = pd.concat(
        [primary, separate],
        ignore_index=True
    )

    for year, g in all_data.groupby("year"):

        total = len(g)

        rows.append({
            "year": int(year),
            "records": total,
            "missing_weight": int(
                g["weight_lbs"].isna().sum()
            ),
            "duplicate_rows": int(
                g.duplicated().sum()
            ),
            "missing_grower": int(
                (g["grower_name"] == "").sum()
            ),
            "missing_city": int(
                (g["city"] == "").sum()
            ),
            "missing_state_province": int(
                (g["state_province"] == "").sum()
            ),
            "missing_country_raw": int(
                (g["country"] == "").sum()
            ),
            "unknown_country_after_cleaning": int(
                (g["country_clean"] == "Unknown").sum()
            ),
            "mother_named": int(
                (g["mother_class"] == "named").sum()
            ),
            "pollinator_named": int(
                (g["pollinator_class"] == "named").sum()
            ),
        })

    quality = pd.DataFrame(rows)

    save_table(
        quality,
        "data_quality_by_year.csv"
    )

    return quality


# ============================================================
# ANNUAL WEIGHT SUMMARY
# ============================================================

def annual_weight_summary(primary, separate):
    rows = []

    all_data = pd.concat(
        [primary, separate],
        ignore_index=True
    )

    for year, g in all_data.groupby("year"):

        w = g["weight_lbs"].dropna().astype(float)

        if len(w) == 0:
            continue

        rows.append({
            "year": int(year),
            "records": len(w),
            "mean_weight": w.mean(),
            "median_weight": w.median(),
            "std_weight": w.std(ddof=1),
            "min_weight": w.min(),
            "q10_weight": w.quantile(.10),
            "q25_weight": w.quantile(.25),
            "q75_weight": w.quantile(.75),
            "q90_weight": w.quantile(.90),
            "q95_weight": w.quantile(.95),
            "q99_weight": w.quantile(.99),
            "record_max": w.max(),
        })

    annual = pd.DataFrame(rows).sort_values("year")

    annual["record_improvement"] = (
        annual["record_max"].diff()
    )

    annual["mean_improvement"] = (
        annual["mean_weight"].diff()
    )

    annual["record_pct_change"] = (
        annual["record_max"].pct_change() * 100
    )

    save_table(
        annual,
        "annual_weight_summary.csv"
    )

    return annual


# ============================================================
# LINEAR REGRESSION
# ============================================================

def linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return None

    result = stats.linregress(x, y)

    slope = float(result.slope)
    intercept = float(result.intercept)
    r2 = float(result.rvalue ** 2)
    p = float(result.pvalue)
    se = float(result.stderr)

    n = len(x)

    tcrit = stats.t.ppf(
        0.975,
        n - 2
    )

    ci_low = slope - tcrit * se
    ci_high = slope + tcrit * se

    fitted = intercept + slope * x
    residuals = y - fitted

    rss = float(np.sum(residuals ** 2))

    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_value": p,
        "slope_se": se,
        "slope_ci_low": ci_low,
        "slope_ci_high": ci_high,
        "rss": rss,
        "x": x,
        "y": y,
        "fitted": fitted,
        "residuals": residuals,
    }


# ============================================================
# THEIL-SEN + SPEARMAN + MANN-KENDALL
# ============================================================

def mann_kendall(x, y):
    """
    Two-sided Mann-Kendall trend test.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)

    y = y[mask]

    n = len(y)

    if n < 4:
        return np.nan, np.nan, np.nan

    S = 0

    for i in range(n - 1):
        differences = y[i + 1:] - y[i]

        S += np.sum(
            np.sign(differences)
        )

    # Tie correction
    _, counts = np.unique(
        y,
        return_counts=True
    )

    tie_term = np.sum(
        counts * (counts - 1) * (2 * counts + 5)
    )

    var_s = (
        n * (n - 1) * (2 * n + 5)
        - tie_term
    ) / 18

    if var_s == 0:
        return float(S), 0.0, 1.0

    if S > 0:
        z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * stats.norm.sf(abs(z))

    return float(S), float(z), float(p)


def trend_analysis(annual, metrics):
    rows = []

    for metric in metrics:

        if metric not in annual.columns:
            continue

        temp = annual[
            ["year", metric]
        ].dropna()

        if len(temp) < 4:
            continue

        x = temp["year"].to_numpy(float)
        y = temp[metric].to_numpy(float)

        # Linear
        fit = linear_regression(x, y)

        if fit is not None:
            rows.append({
                "metric": metric,
                "test": "linear regression",
                "n": fit["n"],
                "effect": fit["slope"],
                "effect_name": "slope_per_year",
                "r2": fit["r2"],
                "p_value": fit["p_value"],
                "ci_low": fit["slope_ci_low"],
                "ci_high": fit["slope_ci_high"],
            })

        # Spearman
        rho, p = stats.spearmanr(x, y)

        rows.append({
            "metric": metric,
            "test": "Spearman",
            "n": len(x),
            "effect": float(rho),
            "effect_name": "rho",
            "r2": np.nan,
            "p_value": float(p),
            "ci_low": np.nan,
            "ci_high": np.nan,
        })

        # Kendall
        tau, p = stats.kendalltau(x, y)

        rows.append({
            "metric": metric,
            "test": "Kendall",
            "n": len(x),
            "effect": float(tau),
            "effect_name": "tau",
            "r2": np.nan,
            "p_value": float(p),
            "ci_low": np.nan,
            "ci_high": np.nan,
        })

        # Mann-Kendall
        S, z, p = mann_kendall(x, y)

        rows.append({
            "metric": metric,
            "test": "Mann-Kendall",
            "n": len(x),
            "effect": S,
            "effect_name": "S",
            "r2": np.nan,
            "p_value": p,
            "ci_low": np.nan,
            "ci_high": np.nan,
        })

        # Theil-Sen
        try:
            ts = stats.theilslopes(
                y,
                x,
                alpha=0.95
            )

            rows.append({
                "metric": metric,
                "test": "Theil-Sen",
                "n": len(x),
                "effect": float(ts.slope),
                "effect_name": "slope_per_year",
                "r2": np.nan,
                "p_value": np.nan,
                "ci_low": float(ts.low_slope),
                "ci_high": float(ts.high_slope),
            })

        except Exception:
            pass

    out = pd.DataFrame(rows)

    if len(out):
        out["family"] = "trend"

    return out


# ============================================================
# FDR
# ============================================================

def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)

    q = np.full_like(
        p,
        np.nan,
        dtype=float
    )

    valid = np.isfinite(p)

    if not valid.any():
        return q

    pv = p[valid]

    order = np.argsort(pv)

    ranked = pv[order]

    m = len(ranked)

    adjusted = (
        ranked * m /
        np.arange(1, m + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    temp = np.empty_like(adjusted)

    temp[order] = adjusted

    q[valid] = temp

    return q


def add_fdr(df):
    df = df.copy()

    if "p_value" not in df.columns:
        return df

    if "family" not in df.columns:
        df["family"] = "all"

    df["fdr_q"] = np.nan

    for family, idx in df.groupby("family").groups.items():

        vals = df.loc[
            idx,
            "p_value"
        ].to_numpy(float)

        df.loc[
            idx,
            "fdr_q"
        ] = benjamini_hochberg(vals)

    return df


# ============================================================
# BOOTSTRAP SLOPE
# ============================================================

def bootstrap_linear_slope(
    x,
    y,
    n_boot=BOOTSTRAP_N
):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    slopes = []

    n = len(x)

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n
        )

        xb = x[idx]
        yb = y[idx]

        if len(np.unique(xb)) < 3:
            continue

        try:
            fit = stats.linregress(
                xb,
                yb
            )

            if np.isfinite(fit.slope):
                slopes.append(
                    fit.slope
                )

        except Exception:
            continue

    if not slopes:
        return {
            "median": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    return {
        "median": float(np.median(slopes)),
        "ci_low": float(np.percentile(slopes, 2.5)),
        "ci_high": float(np.percentile(slopes, 97.5)),
    }


# ============================================================
# QUADRATIC MODEL
# ============================================================

def fit_polynomial(x, y, degree):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    X = np.column_stack([
        x ** p
        for p in range(degree + 1)
    ])

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    fitted = X @ beta

    residuals = y - fitted

    rss = float(
        np.sum(residuals ** 2)
    )

    n = len(y)
    k = degree + 1

    if rss <= 0:
        aic = -np.inf
        bic = -np.inf
    else:
        aic = (
            n * np.log(rss / n)
            + 2 * k
        )

        bic = (
            n * np.log(rss / n)
            + k * np.log(n)
        )

    tss = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1 - rss / tss
        if tss > 0
        else np.nan
    )

    return {
        "degree": degree,
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "r2": r2,
        "x": x,
        "y": y,
    }


# ============================================================
# PIECEWISE / CHANGEPOINT
# ============================================================

def fit_piecewise(x, y, breakpoint):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    hinge = np.maximum(
        0,
        x - breakpoint
    )

    X = np.column_stack([
        np.ones(len(x)),
        x,
        hinge
    ])

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    fitted = X @ beta

    residuals = y - fitted

    rss = float(
        np.sum(residuals ** 2)
    )

    n = len(y)
    k = 3

    if rss <= 0:
        aic = -np.inf
        bic = -np.inf
    else:
        aic = (
            n * np.log(rss / n)
            + 2 * k
        )

        bic = (
            n * np.log(rss / n)
            + k * np.log(n)
        )

    slope_before = beta[1]
    slope_change = beta[2]
    slope_after = beta[1] + beta[2]

    return {
        "breakpoint": breakpoint,
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "rss": rss,
        "aic": aic,
        "bic": bic,
        "slope_before": slope_before,
        "slope_change": slope_change,
        "slope_after": slope_after,
        "x": x,
        "y": y,
    }


def search_changepoints(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    candidates = sorted(
        np.unique(x)
    )[2:-2]

    rows = []

    models = []

    for cp in candidates:

        model = fit_piecewise(
            x,
            y,
            cp
        )

        models.append(model)

        rows.append({
            "breakpoint": cp,
            "rss": model["rss"],
            "aic": model["aic"],
            "bic": model["bic"],
            "slope_before": model["slope_before"],
            "slope_change": model["slope_change"],
            "slope_after": model["slope_after"],
        })

    table = pd.DataFrame(rows)

    if len(table) == 0:
        return table, None

    best_idx = table["bic"].idxmin()

    best = models[
        table.index.get_loc(best_idx)
    ]

    return table, best


def bootstrap_changepoint(
    x,
    y,
    best_model,
    n_boot=CHANGEPOINT_BOOTSTRAP_N
):

    if best_model is None:
        return pd.DataFrame()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fitted = best_model["fitted"]
    residuals = y - fitted

    results = []

    for _ in range(n_boot):

        resampled_resid = rng.choice(
            residuals,
            size=len(residuals),
            replace=True
        )

        y_boot = fitted + resampled_resid

        try:
            _, model = search_changepoints(
                x,
                y_boot
            )

            if model is not None:
                results.append(
                    model["breakpoint"]
                )

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    arr = np.asarray(results)

    return pd.DataFrame({
        "breakpoint": arr
    })


# ============================================================
# MODEL DIAGNOSTICS
# ============================================================

def residual_diagnostics(model, label):
    resid = np.asarray(
        model["residuals"],
        dtype=float
    )

    fitted = np.asarray(
        model["fitted"],
        dtype=float
    )

    n = len(resid)

    jb_stat, jb_p = safe_jarque_bera(
        resid
    )

    dw = np.sum(
        np.diff(resid) ** 2
    ) / np.sum(
        resid ** 2
    )

    row = {
        "model": label,
        "n": n,
        "residual_mean": np.mean(resid),
        "residual_sd": np.std(
            resid,
            ddof=1
        ),
        "residual_skew": safe_skew(
            resid
        ),
        "residual_kurtosis": safe_kurtosis(
            resid
        ),
        "jarque_bera": jb_stat,
        "jarque_bera_p": jb_p,
        "durbin_watson": dw,
    }

    if STATSMODELS_AVAILABLE:

        try:
            lb = acorr_ljungbox(
                resid,
                lags=[min(3, max(1, n // 3))],
                return_df=True
            )

            row["ljung_box_p"] = float(
                lb["lb_pvalue"].iloc[0]
            )

        except Exception:
            row["ljung_box_p"] = np.nan

        try:
            X = sm.add_constant(
                fitted
            )

            bp = het_breuschpagan(
                resid,
                X
            )

            row["breusch_pagan_lm"] = float(bp[0])
            row["breusch_pagan_p"] = float(bp[1])

        except Exception:
            row["breusch_pagan_lm"] = np.nan
            row["breusch_pagan_p"] = np.nan

    else:
        row["ljung_box_p"] = np.nan
        row["breusch_pagan_lm"] = np.nan
        row["breusch_pagan_p"] = np.nan

    return row


# ============================================================
# LEAVE-ONE-OUT SENSITIVITY
# ============================================================

def leave_one_out_slopes(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    rows = []

    for i in range(len(x)):

        mask = np.ones(
            len(x),
            dtype=bool
        )

        mask[i] = False

        fit = linear_regression(
            x[mask],
            y[mask]
        )

        rows.append({
            "omitted_year": int(x[i]),
            "slope": fit["slope"],
            "r2": fit["r2"],
            "p_value": fit["p_value"],
        })

    return pd.DataFrame(rows)


# ============================================================
# PARENT METRICS
# ============================================================

def parent_metrics(primary, column, prefix):

    rows = []

    seen = set()

    for year, g in primary.groupby("year"):

        classes = (
            g[column.replace("_clean", "")]
            if column in g.columns
            else g[column]
        )

        # column is already the original parent field here
        values = g[column].map(
            norm_text
        )

        classifications = values.map(
            classify_parent
        )

        named = values[
            classifications == "named"
        ]

        counts = (
            named.value_counts()
            if len(named)
            else pd.Series(dtype=int)
        )

        n_records = len(g)
        n_named = len(named)
        n_unique = counts.size

        if n_named > 0:

            shares = (
                counts /
                n_named
            )

            top1 = shares.iloc[:1].sum()
            top5 = shares.iloc[:5].sum()
            top10 = shares.iloc[:10].sum()

            simpson = float(
                np.sum(
                    shares.to_numpy() ** 2
                )
            )

            effective = (
                1 / simpson
                if simpson > 0
                else np.nan
            )

            singleton_records = (
                counts[counts == 1].sum()
            )

            singleton_parent_fraction = (
                (counts == 1).mean()
            )

            singleton_record_fraction = (
                singleton_records /
                n_named
            )

            gini_value = gini(
                counts.to_numpy()
            )

            current_parents = set(
                counts.index
            )

            new_parents = (
                current_parents - seen
            )

            new_parent_fraction = (
                len(new_parents) /
                n_unique
                if n_unique
                else np.nan
            )

            jaccard = (
                len(current_parents & seen) /
                len(current_parents | seen)
                if seen and current_parents
                else np.nan
            )

            seen.update(
                current_parents
            )

        else:

            top1 = np.nan
            top5 = np.nan
            top10 = np.nan
            simpson = np.nan
            effective = np.nan
            singleton_parent_fraction = np.nan
            singleton_record_fraction = np.nan
            gini_value = np.nan
            new_parent_fraction = np.nan
            jaccard = np.nan

        rows.append({
            "year": int(year),
            "records": n_records,
            f"{prefix}_named_records": n_named,
            f"{prefix}_unique": n_unique,
            f"{prefix}_unique_ratio": (
                n_unique / n_named
                if n_named
                else np.nan
            ),
            f"{prefix}_top1_share": top1,
            f"{prefix}_top5_share": top5,
            f"{prefix}_top10_share": top10,
            f"{prefix}_simpson": simpson,
            f"{prefix}_effective": effective,
            f"{prefix}_gini": gini_value,
            f"{prefix}_singleton_parent_fraction":
                singleton_parent_fraction,
            f"{prefix}_singleton_record_fraction":
                singleton_record_fraction,
            f"{prefix}_new_parent_fraction":
                new_parent_fraction,
            f"{prefix}_jaccard_previous":
                jaccard,
        })

    return pd.DataFrame(rows)


def gini(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    if np.any(values < 0):
        return np.nan

    if np.sum(values) == 0:
        return 0.0

    values = np.sort(values)

    n = len(values)

    index = np.arange(
        1,
        n + 1
    )

    return float(
        (
            np.sum(
                (2 * index - n - 1)
                * values
            )
        )
        /
        (
            n * np.sum(values)
        )
    )


def parent_frequency_table(
    primary,
    column
):

    rows = []

    for parent, g in (
        primary.groupby(
            primary[column].map(norm_text)
        )
    ):

        classification = classify_parent(
            parent
        )

        if classification != "named":
            continue

        years = sorted(
            g["year"].unique()
        )

        rows.append({
            "parent": parent,
            "records": len(g),
            "years_present": len(years),
            "first_year": min(years),
            "last_year": max(years),
            "year_span": max(years) - min(years) + 1,
            "mean_weight": g["weight_lbs"].mean(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["years_present", "records"],
            ascending=False
        )
        if rows
        else pd.DataFrame()
    )


def parent_period_metrics(
    primary,
    column,
    prefix
):

    periods = {
        "2013-2016": (2013, 2016),
        "2017-2020": (2017, 2020),
        "2021-2025": (2021, 2025),
    }

    rows = []

    for label, (start, end) in periods.items():

        g = primary[
            primary["year"].between(
                start,
                end
            )
        ]

        values = g[column].map(norm_text)

        classes = values.map(
            classify_parent
        )

        named = values[
            classes == "named"
        ]

        counts = named.value_counts()

        n = len(named)

        if n == 0:
            continue

        shares = counts / n

        simpson = np.sum(
            shares.to_numpy() ** 2
        )

        rows.append({
            "period": label,
            "records": len(g),
            f"{prefix}_named": n,
            f"{prefix}_unique": len(counts),
            f"{prefix}_unique_ratio":
                len(counts) / n,
            f"{prefix}_top1":
                shares.iloc[:1].sum(),
            f"{prefix}_top5":
                shares.iloc[:5].sum(),
            f"{prefix}_top10":
                shares.iloc[:10].sum(),
            f"{prefix}_simpson":
                simpson,
            f"{prefix}_effective":
                1 / simpson,
        })

    return pd.DataFrame(rows)


# ============================================================
# PARENT JACCARD MATRIX
# ============================================================

def parent_jaccard_matrix(
    primary,
    column
):

    year_sets = {}

    for year, g in primary.groupby("year"):

        values = g[column].map(norm_text)

        named = values[
            values.map(classify_parent) == "named"
        ]

        year_sets[int(year)] = set(
            named.unique()
        )

    years = sorted(
        year_sets
    )

    matrix = np.zeros(
        (len(years), len(years))
    )

    for i, y1 in enumerate(years):

        for j, y2 in enumerate(years):

            a = year_sets[y1]
            b = year_sets[y2]

            union = a | b

            matrix[i, j] = (
                len(a & b) / len(union)
                if union
                else np.nan
            )

    return (
        pd.DataFrame(
            matrix,
            index=years,
            columns=years
        ),
        year_sets
    )


# ============================================================
# GEOGRAPHY
# ============================================================

def country_gini(counts):
    return gini(
        np.asarray(
            counts,
            dtype=float
        )
    )


def country_metrics(primary):

    annual_rows = []
    share_rows = []

    previous_countries = set()

    for year, g in primary.groupby("year"):

        valid = g[
            g["country_clean"] != "Unknown"
        ]

        counts = (
            valid["country_clean"]
            .value_counts()
        )

        n = counts.sum()

        if n == 0:
            continue

        shares = counts / n

        hhi = float(
            np.sum(
                shares.to_numpy() ** 2
            )
        )

        shannon = float(
            -np.sum(
                shares.to_numpy()
                * np.log(
                    shares.to_numpy()
                )
            )
        )

        current = set(
            counts.index
        )

        union = (
            current |
            previous_countries
        )

        jaccard = (
            len(
                current &
                previous_countries
            )
            /
            len(union)
            if previous_countries
            else np.nan
        )

        top_country = (
            counts.index[0]
            if len(counts)
            else ""
        )

        top1 = shares.iloc[:1].sum()
        top3 = shares.iloc[:3].sum()
        top5 = shares.iloc[:5].sum()

        annual_rows.append({
            "year": int(year),
            "records": int(n),
            "country": top_country,
            "countries": len(counts),
            "country_hhi": hhi,
            "effective_countries": (
                1 / hhi
                if hhi > 0
                else np.nan
            ),
            "country_shannon": shannon,
            "effective_countries_shannon":
                np.exp(shannon),
            "country_gini":
                country_gini(
                    counts.to_numpy()
                ),
            "country_top1": top1,
            "country_top3": top3,
            "country_top5": top5,
            "country_jaccard_previous":
                jaccard,
        })

        ranks = np.arange(
            1,
            len(counts) + 1
        )

        for country, count, share, rank in zip(
            counts.index,
            counts.values,
            shares.values,
            ranks
        ):

            share_rows.append({
                "year": int(year),
                "country": country,
                "records": int(count),
                "share": float(share),
                "rank": int(rank),
            })

        previous_countries = current

    annual = pd.DataFrame(
        annual_rows
    ).sort_values("year")

    shares = pd.DataFrame(
        share_rows
    ).sort_values(
        ["year", "rank"]
    )

    save_table(
        annual,
        "country_concentration_annual.csv"
    )

    save_table(
        shares,
        "country_shares_annual.csv"
    )

    return annual, shares


def geography_expansion(primary):

    rows = []

    for year, g in primary.groupby("year"):

        country_count = len(
            set(
                g.loc[
                    g["country_clean"] != "Unknown",
                    "country_clean"
                ]
            )
        )

        state_count = len(
            set(
                g["state_province"].replace(
                    "",
                    np.nan
                ).dropna()
            )
        )

        city_count = len(
            set(
                g["city"].replace(
                    "",
                    np.nan
                ).dropna()
            )
        )

        site_count = len(
            set(
                g["gpc_site"].replace(
                    "",
                    np.nan
                ).dropna()
            )
        )

        growers = g[
            g["grower_name"] != ""
        ]["grower_name"].nunique()

        rows.append({
            "year": int(year),
            "countries": country_count,
            "states_provinces": state_count,
            "cities": city_count,
            "gpc_sites": site_count,
            "unique_growers": growers,
        })

    out = pd.DataFrame(rows)

    save_table(
        out,
        "geographic_expansion.csv"
    )

    return out


def early_late_country_comparison(
    shares
):

    early = shares[
        shares["year"].between(
            2013,
            2016
        )
    ]

    late = shares[
        shares["year"].between(
            2022,
            2025
        )
    ]

    early_counts = (
        early.groupby("country")["records"]
        .sum()
    )

    late_counts = (
        late.groupby("country")["records"]
        .sum()
    )

    early_total = early_counts.sum()
    late_total = late_counts.sum()

    countries = sorted(
        set(early_counts.index)
        |
        set(late_counts.index)
    )

    rows = []

    for country in countries:

        e = (
            early_counts.get(country, 0)
            / early_total
            if early_total
            else 0
        )

        l = (
            late_counts.get(country, 0)
            / late_total
            if late_total
            else 0
        )

        rows.append({
            "country": country,
            "early_share":
                e,
            "late_share":
                l,
            "change":
                l - e,
            "early_records":
                early_counts.get(country, 0),
            "late_records":
                late_counts.get(country, 0),
        })

    out = pd.DataFrame(rows).sort_values(
        "late_share",
        ascending=False
    )

    save_table(
        out,
        "early_vs_late_country_comparison.csv"
    )

    return out


# ============================================================
# GROWER METRICS
# ============================================================

def grower_metrics(primary):

    rows = []

    first_seen = {}

    for year, g in primary.groupby("year"):

        growers = g[
            g["grower_name"] != ""
        ]["grower_name"]

        counts = growers.value_counts()

        unique = len(counts)

        repeat = (
            (counts > 1).sum()
            if unique
            else 0
        )

        # Growers appearing for the first time this year
        current = set(
            counts.index
        )

        new = (
            current - set(first_seen)
        )

        rows.append({
            "year": int(year),
            "unique_growers": unique,
            "grower_records": len(growers),
            "repeat_growers":
                int(repeat),
            "repeat_grower_fraction":
                repeat / unique
                if unique
                else np.nan,
            "new_growers":
                len(new),
            "new_grower_fraction":
                len(new) / unique
                if unique
                else np.nan,
            "mean_records_per_grower":
                len(growers) / unique
                if unique
                else np.nan,
        })

        for name in current:
            first_seen.setdefault(
                name,
                int(year)
            )

    out = pd.DataFrame(rows)

    save_table(
        out,
        "grower_metrics.csv"
    )

    return out


# ============================================================
# ESTIMATED WEIGHT / OTT
# ============================================================

def measurement_analysis(primary):

    rows = []

    if "estimated_weight" in primary.columns:

        temp = primary[
            [
                "weight_lbs",
                "estimated_weight"
            ]
        ].dropna()

        if len(temp):

            actual = temp[
                "weight_lbs"
            ].to_numpy(float)

            estimated = temp[
                "estimated_weight"
            ].to_numpy(float)

            error = (
                estimated - actual
            )

            rows.append({
                "analysis": "estimated_weight_vs_actual",
                "n": len(temp),
                "pearson_r":
                    stats.pearsonr(
                        actual,
                        estimated
                    ).statistic,
                "pearson_p":
                    stats.pearsonr(
                        actual,
                        estimated
                    ).pvalue,
                "mae":
                    np.mean(
                        np.abs(error)
                    ),
                "rmse":
                    np.sqrt(
                        np.mean(
                            error ** 2
                        )
                    ),
                "mean_error":
                    np.mean(error),
            })

    if "ott" in primary.columns:

        temp = primary[
            [
                "weight_lbs",
                "ott"
            ]
        ].dropna()

        if len(temp) >= 3:

            actual = temp[
                "weight_lbs"
            ].to_numpy(float)

            ott = temp[
                "ott"
            ].to_numpy(float)

            pear = stats.pearsonr(
                actual,
                ott
            )

            spear = stats.spearmanr(
                actual,
                ott
            )

            rows.append({
                "analysis": "ott_vs_actual",
                "n": len(temp),
                "pearson_r":
                    pear.statistic,
                "pearson_p":
                    pear.pvalue,
                "spearman_r":
                    spear.statistic,
                "spearman_p":
                    spear.pvalue,
            })

    out = pd.DataFrame(rows)

    save_table(
        out,
        "measurement_analysis.csv"
    )

    return out


# ============================================================
# EXACT PERMUTATION TEST
# ============================================================

def exact_mean_permutation_p(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    n1 = len(a)
    n2 = len(b)

    if n1 < 2 or n2 < 2:
        return np.nan

    pooled = np.concatenate(
        [a, b]
    )

    observed = abs(
        np.mean(b) -
        np.mean(a)
    )

    count = 0
    total = 0

    for idx in itertools.combinations(
        range(len(pooled)),
        n1
    ):

        idx = set(idx)

        group_a = np.array([
            pooled[i]
            for i in range(len(pooled))
            if i in idx
        ])

        group_b = np.array([
            pooled[i]
            for i in range(len(pooled))
            if i not in idx
        ])

        diff = abs(
            np.mean(group_b) -
            np.mean(group_a)
        )

        if diff >= observed - 1e-12:
            count += 1

        total += 1

    return (
        (count + 1) /
        (total + 1)
    )


def cliffs_delta(a, b):

    a = np.asarray(a)
    b = np.asarray(b)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if not len(a) or not len(b):
        return np.nan

    # U statistic
    u = stats.mannwhitneyu(
        b,
        a,
        alternative="two-sided"
    ).statistic

    return (
        2 * u /
        (len(a) * len(b))
        - 1
    )


def early_late_tests(annual):

    rows = []

    metrics = [
        "record_max",
        "mean_weight",
        "median_weight",
        "record_improvement",
        "mother_effective",
        "pollinator_effective",
        "countries",
        "states_provinces",
        "cities",
        "gpc_sites",
        "effective_countries",
        "country_hhi",
        "country_top1",
        "country_top5",
        "unique_growers",
    ]

    for metric in metrics:

        if metric not in annual.columns:
            continue

        early = annual[
            annual["year"].between(
                2013,
                2016
            )
        ][metric].dropna()

        late = annual[
            annual["year"].between(
                2022,
                2025
            )
        ][metric].dropna()

        if len(early) < 2 or len(late) < 2:
            continue

        mw = stats.mannwhitneyu(
            late,
            early,
            alternative="two-sided"
        )

        perm = exact_mean_permutation_p(
            early,
            late
        )

        rows.append({
            "metric": metric,
            "early_mean": early.mean(),
            "late_mean": late.mean(),
            "difference":
                late.mean() -
                early.mean(),
            "early_median":
                early.median(),
            "late_median":
                late.median(),
            "mann_whitney_u":
                mw.statistic,
            "mann_whitney_p":
                mw.pvalue,
            "cliffs_delta":
                cliffs_delta(
                    late,
                    early
                ),
            "exact_permutation_p":
                perm,
            "family":
                "early_late",
        })

    return pd.DataFrame(rows)


# ============================================================
# CROSS-DOMAIN CORRELATIONS
# ============================================================

def cross_domain_correlations(annual):

    metrics = [
        "record_max",
        "mean_weight",
        "mother_effective",
        "pollinator_effective",
        "mother_top10_share",
        "pollinator_top10_share",
        "countries",
        "states_provinces",
        "cities",
        "gpc_sites",
        "effective_countries",
        "country_hhi",
        "country_top1",
        "country_top5",
        "unique_growers",
    ]

    metrics = [
        m for m in metrics
        if m in annual.columns
    ]

    rows = []

    for i, m1 in enumerate(metrics):

        for m2 in metrics[i + 1:]:

            temp = annual[
                [m1, m2]
            ].dropna()

            if len(temp) < 5:
                continue

            pear = stats.pearsonr(
                temp[m1],
                temp[m2]
            )

            spear = stats.spearmanr(
                temp[m1],
                temp[m2]
            )

            rows.append({
                "metric_1": m1,
                "metric_2": m2,
                "n": len(temp),
                "pearson_r":
                    pear.statistic,
                "pearson_p":
                    pear.pvalue,
                "spearman_r":
                    spear.statistic,
                "spearman_p":
                    spear.pvalue,
                "family":
                    "cross_domain",
            })

    return pd.DataFrame(rows)


# ============================================================
# ANNUAL MASTER TABLE
# ============================================================

def build_annual_master(
    annual_weight,
    mother,
    pollinator,
    geo,
    countries,
    growers
):
    """
    Combine annual analysis tables without creating pandas merge
    collisions such as records_x / records_y.

    Several component tables intentionally contain a generic `records`
    column. The annual weight table is the authoritative source for the
    number of weighted records, so duplicate generic columns are omitted
    from later tables. Other columns are retained unless they have already
    been supplied by an earlier table.
    """

    annual = annual_weight.copy()

    # Normalize and collapse accidental duplicate column names.
    annual = annual.loc[:, ~annual.columns.duplicated()].copy()

    for other in [
        mother,
        pollinator,
        geo,
        countries,
        growers
    ]:

        if other is None or len(other) == 0:
            continue

        other = other.copy()
        other = other.loc[:, ~other.columns.duplicated()].copy()

        if "year" not in other.columns:
            continue

        # Keep one row per year. These tables should already be annual,
        # but this prevents a malformed/duplicated input from multiplying
        # rows during the merge.
        other = other.sort_values("year").drop_duplicates(
            subset=["year"],
            keep="last"
        )

        # Never merge columns that are already present. In particular,
        # this prevents generic `records` columns from becoming
        # records_x / records_y and later colliding with another merge.
        keep_cols = [
            c for c in other.columns
            if c == "year" or c not in annual.columns
        ]

        other = other[keep_cols]

        annual = annual.merge(
            other,
            on="year",
            how="left",
            validate="one_to_one"
        )

        annual = annual.loc[
            :, ~annual.columns.duplicated()
        ].copy()

    annual = annual.sort_values("year").reset_index(drop=True)

    save_table(
        annual,
        "annual_master.csv"
    )

    return annual


# ============================================================
# MODEL ANALYSIS
# ============================================================

def record_model_analysis(annual):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ][
        ["year", "record_max"]
    ].dropna()

    x = temp["year"].to_numpy(float)
    y = temp["record_max"].to_numpy(float)

    # Linear
    linear = linear_regression(
        x,
        y
    )

    # Quadratic
    quadratic = fit_polynomial(
        x,
        y,
        2
    )

    # Quadratic bootstrap
    quad_coefs = []

    for _ in range(BOOTSTRAP_N):

        idx = rng.integers(
            0,
            len(x),
            size=len(x)
        )

        xb = x[idx]
        yb = y[idx]

        if len(np.unique(xb)) < 5:
            continue

        try:
            model = fit_polynomial(
                xb,
                yb,
                2
            )

            # beta = intercept,
            # linear,
            # quadratic
            quad_coefs.append(
                model["beta"][2]
            )

        except Exception:
            continue

    quad_coefs = np.asarray(
        quad_coefs
    )

    if len(quad_coefs):

        quad_bootstrap = {
            "median":
                np.median(quad_coefs),
            "ci_low":
                np.percentile(
                    quad_coefs,
                    2.5
                ),
            "ci_high":
                np.percentile(
                    quad_coefs,
                    97.5
                ),
        }

    else:

        quad_bootstrap = {
            "median": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    # Changepoint
    cp_table, cp_model = search_changepoints(
        x,
        y
    )

    if len(cp_table):
        save_table(
            cp_table,
            "changepoint_candidates.csv"
        )

    cp_boot = bootstrap_changepoint(
        x,
        y,
        cp_model
    )

    if len(cp_boot):
        save_table(
            cp_boot,
            "changepoint_bootstrap.csv"
        )

    model_rows = [
        {
            "model": "linear",
            "r2": linear["r2"],
            "rss": linear["rss"],
            "aic": np.nan,
            "bic": np.nan,
            "slope": linear["slope"],
        },
        {
            "model": "quadratic",
            "r2": quadratic["r2"],
            "rss": quadratic["rss"],
            "aic": quadratic["aic"],
            "bic": quadratic["bic"],
            "slope":
                quadratic["beta"][1],
            "quadratic_coefficient":
                quadratic["beta"][2],
        },
    ]

    if cp_model is not None:

        model_rows.append({
            "model":
                f"piecewise_cp_{cp_model['breakpoint']:.0f}",
            "r2":
                1 - (
                    cp_model["rss"]
                    /
                    np.sum(
                        (
                            y -
                            np.mean(y)
                        ) ** 2
                    )
                ),
            "rss":
                cp_model["rss"],
            "aic":
                cp_model["aic"],
            "bic":
                cp_model["bic"],
            "slope":
                cp_model["slope_before"],
            "slope_change":
                cp_model["slope_change"],
            "slope_after":
                cp_model["slope_after"],
        })

    model_table = pd.DataFrame(
        model_rows
    )

    save_table(
        model_table,
        "record_weight_models.csv"
    )

    # Corrected compatibility handling:
    # fitted_values may already be ndarray.
    diagnostics = [
        residual_diagnostics(
            linear,
            "linear"
        ),
        residual_diagnostics(
            quadratic,
            "quadratic"
        ),
    ]

    if cp_model is not None:

        diagnostics.append(
            residual_diagnostics(
                cp_model,
                "piecewise"
            )
        )

    diagnostics = pd.DataFrame(
        diagnostics
    )

    save_table(
        diagnostics,
        "residual_diagnostics.csv"
    )

    # Leave-one-out
    loo = leave_one_out_slopes(
        x,
        y
    )

    save_table(
        loo,
        "leave_one_out_record_slopes.csv"
    )

    # Bootstrap linear
    linear_boot = bootstrap_linear_slope(
        x,
        y
    )

    bootstrap_table = pd.DataFrame([
        {
            "model":
                "linear",
            **linear_boot,
        },
        {
            "model":
                "quadratic_coefficient",
            **quad_bootstrap,
        },
    ])

    save_table(
        bootstrap_table,
        "bootstrap_model_parameters.csv"
    )

    return {
        "temp": temp,
        "linear": linear,
        "quadratic": quadratic,
        "cp_table": cp_table,
        "cp_model": cp_model,
        "diagnostics": diagnostics,
    }



# ============================================================
# ADDITIONAL COMPETITIVE-DEPTH / PEDIGREE ANALYSES
# ============================================================

def weight_threshold_analysis(primary):
    """Track the upper tail of the competition by weight threshold."""
    thresholds = [1000, 1500, 2000, 2500]
    rows = []

    for year, g in primary.groupby("year"):
        w = pd.to_numeric(g["weight_lbs"], errors="coerce").dropna()
        if len(w) == 0:
            continue

        row = {"year": int(year), "records": len(w)}
        for threshold in thresholds:
            count = int((w >= threshold).sum())
            row[f"count_ge_{threshold}"] = count
            row[f"pct_ge_{threshold}"] = 100 * count / len(w)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year")
    save_table(out, "weight_thresholds_annual.csv")
    return out


def weight_distribution_shape(primary):
    """Quantify annual spread and extremity of the recorded weight distribution."""
    rows = []

    for year, g in primary.groupby("year"):
        w = pd.to_numeric(g["weight_lbs"], errors="coerce").dropna()
        if len(w) < 3:
            continue

        q10, q25, q50, q75, q90, q95, q99 = (
            w.quantile([.10, .25, .50, .75, .90, .95, .99])
        )
        mean = w.mean()
        sd = w.std(ddof=1)

        rows.append({
            "year": int(year),
            "records": len(w),
            "mean_weight": mean,
            "median_weight": q50,
            "iqr": q75 - q25,
            "p90_minus_median": q90 - q50,
            "p95_minus_median": q95 - q50,
            "p99_minus_median": q99 - q50,
            "record_minus_p99": w.max() - q99,
            "record_to_p99_ratio": w.max() / q99 if q99 > 0 else np.nan,
            "coefficient_of_variation_pct": (
                100 * sd / mean if mean != 0 else np.nan
            ),
            "skewness": stats.skew(w, bias=False),
            "kurtosis": stats.kurtosis(w, fisher=True, bias=False),
        })

    out = pd.DataFrame(rows).sort_values("year")
    save_table(out, "weight_distribution_shape_annual.csv")
    return out


def grower_concentration_analysis(primary):
    """
    Measure concentration of recorded pumpkins among growers.
    This describes dataset participation/reporting concentration.
    """
    rows = []

    for year, g in primary.groupby("year"):
        growers = g[
            g["grower_name"].astype(str).str.strip() != ""
        ]["grower_name"]
        counts = growers.value_counts()

        if len(counts) == 0:
            continue

        shares = counts / counts.sum()

        rows.append({
            "year": int(year),
            "grower_records": int(counts.sum()),
            "unique_growers": int(len(counts)),
            "grower_top1_share": float(shares.iloc[0]),
            "grower_top5_share": float(shares.iloc[:5].sum()),
            "grower_top10_share": float(shares.iloc[:10].sum()),
            "grower_hhi": float(np.sum(shares.to_numpy() ** 2)),
            "effective_growers_hhi": float(
                1 / np.sum(shares.to_numpy() ** 2)
            ),
            "gini_growers": float(gini(counts.to_numpy())),
        })

    out = pd.DataFrame(rows).sort_values("year")
    save_table(out, "grower_concentration_annual.csv")
    return out


def parent_pair_analysis(primary):
    """
    Analyze seed-mother × pollinator combinations.
    This is pedigree structure, not genomic diversity.
    """
    temp = primary[
        ["year", "seed_mother", "pollinator_father"]
    ].copy()

    temp["seed_mother"] = temp["seed_mother"].astype(str).str.strip()
    temp["pollinator_father"] = (
        temp["pollinator_father"].astype(str).str.strip()
    )

    temp = temp[
        (temp["seed_mother"] != "") &
        (temp["pollinator_father"] != "")
    ]

    rows = []

    for year, g in temp.groupby("year"):
        pairs = (
            g.groupby(["seed_mother", "pollinator_father"])
            .size()
            .sort_values(ascending=False)
        )

        if len(pairs) == 0:
            continue

        shares = pairs / pairs.sum()

        rows.append({
            "year": int(year),
            "named_parent_pairs": int(pairs.sum()),
            "unique_parent_pairs": int(len(pairs)),
            "pair_unique_ratio": len(pairs) / pairs.sum(),
            "pair_top1_share": float(shares.iloc[0]),
            "pair_top5_share": float(shares.iloc[:5].sum()),
            "pair_top10_share": float(shares.iloc[:10].sum()),
            "pair_hhi": float(np.sum(shares.to_numpy() ** 2)),
            "effective_parent_pairs": float(
                1 / np.sum(shares.to_numpy() ** 2)
            ),
        })

    annual = pd.DataFrame(rows).sort_values("year")
    save_table(annual, "parent_pair_annual_metrics.csv")

    pair_counts = (
        temp.groupby(["seed_mother", "pollinator_father"])
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    save_table(pair_counts, "parent_pair_frequency.csv")

    return annual, pair_counts


def elite_pedigree_analysis(primary, percentile=0.99):
    """
    Compare documented pedigree concentration in the top 1% by weight
    against the full recorded population.
    """
    rows = []

    for year, g in primary.groupby("year"):
        g = g.copy()
        w = pd.to_numeric(g["weight_lbs"], errors="coerce")
        g = g[w.notna()].copy()
        g["weight_lbs"] = w[w.notna()]

        if len(g) < 20:
            continue

        cutoff = g["weight_lbs"].quantile(percentile)
        elite = g[g["weight_lbs"] >= cutoff].copy()

        row = {
            "year": int(year),
            "population_n": len(g),
            "elite_n": len(elite),
            "elite_cutoff_lbs": cutoff,
        }

        for col, prefix in [
            ("seed_mother", "mother"),
            ("pollinator_father", "pollinator"),
        ]:
            all_named = g[
                g[col].astype(str).str.strip() != ""
            ][col].astype(str).str.strip()

            elite_named = elite[
                elite[col].astype(str).str.strip() != ""
            ][col].astype(str).str.strip()

            if len(all_named) == 0 or len(elite_named) == 0:
                continue

            all_counts = all_named.value_counts()
            elite_counts = elite_named.value_counts()
            all_shares = all_counts / all_counts.sum()
            elite_shares = elite_counts / elite_counts.sum()

            row[f"{prefix}_elite_top1"] = elite_shares.iloc[0]
            row[f"{prefix}_elite_top5"] = elite_shares.iloc[:5].sum()
            row[f"{prefix}_elite_top10"] = elite_shares.iloc[:10].sum()
            row[f"{prefix}_elite_hhi"] = np.sum(
                elite_shares.to_numpy() ** 2
            )
            row[f"{prefix}_all_top10"] = all_shares.iloc[:10].sum()
            row[f"{prefix}_elite_vs_all_top10"] = (
                row[f"{prefix}_elite_top10"] -
                row[f"{prefix}_all_top10"]
            )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("year")
    save_table(out, "elite_pedigree_annual.csv")
    return out


def threshold_trend_tests(thresholds):
    """Test annual trends in the fraction above each weight threshold."""
    rows = []

    for col in [
        c for c in thresholds.columns
        if c.startswith("pct_ge_")
    ]:
        temp = thresholds[["year", col]].dropna()
        if len(temp) < 5:
            continue

        lr = stats.linregress(temp["year"], temp[col])
        sp = stats.spearmanr(temp["year"], temp[col])

        rows.append({
            "metric": col,
            "n_years": len(temp),
            "slope_pct_points_per_year": lr.slope,
            "linear_r2": lr.rvalue ** 2,
            "linear_p": lr.pvalue,
            "spearman_rho": sp.statistic,
            "spearman_p": sp.pvalue,
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["fdr_linear"] = benjamini_hochberg(
            out["linear_p"].to_numpy(float)
        )
        out["fdr_spearman"] = benjamini_hochberg(
            out["spearman_p"].to_numpy(float)
        )

    save_table(out, "weight_threshold_trend_tests.csv")
    return out


def plot_weight_thresholds(thresholds):
    fig, ax = plt.subplots(figsize=(11, 6))

    for threshold in [1000, 1500, 2000, 2500]:
        col = f"pct_ge_{threshold}"
        if col in thresholds:
            ax.plot(
                thresholds["year"],
                thresholds[col],
                marker="o",
                linewidth=2,
                label=f">= {threshold:,} lb"
            )

    ax.set_title("Upper-tail participation by weight threshold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Pumpkins meeting threshold (%)")
    ax.legend()
    save_fig(fig, "figure8_weight_thresholds.png")


def plot_distribution_shape(shape):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        shape["year"], shape["iqr"],
        marker="o", linewidth=2, label="IQR"
    )
    ax.plot(
        shape["year"], shape["p99_minus_median"],
        marker="o", linewidth=2, label="P99 − median"
    )
    ax.plot(
        shape["year"], shape["record_minus_p99"],
        marker="o", linewidth=2, label="Record − P99"
    )

    ax.set_title("How the weight distribution's upper tail changed")
    ax.set_xlabel("Year")
    ax.set_ylabel("Weight spread (lb)")
    ax.legend()
    save_fig(fig, "figure9_weight_distribution_shape.png")


def plot_grower_concentration(growers):
    fig, ax = plt.subplots(figsize=(11, 6))

    for col, label in [
        ("grower_top1_share", "Largest grower"),
        ("grower_top5_share", "Top 5 growers"),
        ("grower_top10_share", "Top 10 growers"),
    ]:
        if col in growers:
            ax.plot(
                growers["year"],
                growers[col] * 100,
                marker="o",
                linewidth=2,
                label=label
            )

    ax.set_title("Concentration of recorded pumpkins among growers")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of recorded pumpkins (%)")
    ax.legend()
    save_fig(fig, "figure10_grower_concentration.png")


def plot_parent_pairs(pair_annual):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(
        pair_annual["year"],
        pair_annual["unique_parent_pairs"],
        marker="o",
        linewidth=2,
        label="Unique parent pairs"
    )
    ax.plot(
        pair_annual["year"],
        pair_annual["effective_parent_pairs"],
        marker="o",
        linewidth=2,
        label="Effective parent pairs"
    )

    ax.set_title("Documented seed-mother × pollinator pair diversity")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of parent pairs")
    ax.legend()
    save_fig(fig, "figure11_parent_pair_diversity.png")


def plot_elite_pedigree(elite):
    fig, ax = plt.subplots(figsize=(11, 6))

    if "mother_elite_vs_all_top10" in elite:
        ax.plot(
            elite["year"],
            elite["mother_elite_vs_all_top10"] * 100,
            marker="o",
            linewidth=2,
            label="Mother"
        )

    if "pollinator_elite_vs_all_top10" in elite:
        ax.plot(
            elite["year"],
            elite["pollinator_elite_vs_all_top10"] * 100,
            marker="o",
            linewidth=2,
            label="Pollinator"
        )

    ax.axhline(0, linewidth=1)
    ax.set_title(
        "Pedigree concentration difference in the top 1% by weight"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(
        "Elite − all-pumpkin top-10 share (percentage points)"
    )
    ax.legend()
    save_fig(fig, "figure12_elite_pedigree_concentration.png")


# ============================================================
# PLOT 1: RECORD TRAJECTORY
# ============================================================

def plot_record_trajectory(
    annual,
    model_info
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ].copy()

    x = temp["year"].to_numpy(float)
    y = temp["record_max"].to_numpy(float)

    model = model_info["linear"]

    xx = np.linspace(
        x.min(),
        x.max(),
        300
    )

    yy = (
        model["intercept"]
        +
        model["slope"] * xx
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.fill_between(
        xx,
        yy - 60,
        yy + 60,
        color=BLUE,
        alpha=0.08
    )

    ax.plot(
        xx,
        yy,
        color=BLUE,
        linewidth=2.5,
        label="Linear trend"
    )

    ax.scatter(
        x,
        y,
        s=90,
        color=ORANGE,
        edgecolor=TEXT,
        linewidth=0.8,
        zorder=5,
        label="Annual record"
    )

    # 2026 separate
    current = annual[
        annual["year"] == 2026
    ]

    if len(current):

        ax.scatter(
            current["year"],
            current["record_max"],
            s=130,
            marker="*",
            color=PINK,
            edgecolor=TEXT,
            linewidth=0.8,
            zorder=6,
            label="2026 separate"
        )

    ax.set_title(
        "Giant Pumpkin Record-Weight Trajectory",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Record weight (lb)"
    )

    ax.legend(
        frameon=False
    )

    ax.text(
        0.02,
        0.95,
        (
            f"slope = {model['slope']:.1f} lb/year\n"
            f"R² = {model['r2']:.3f}\n"
            f"p = {model['p_value']:.3g}"
        ),
        transform=ax.transAxes,
        va="top",
        color=TEXT,
        fontsize=11
    )

    ax.xaxis.set_major_locator(
        mticker.MaxNLocator(
            integer=True
        )
    )

    return save_fig(
        fig,
        "01_record_weight_trajectory.png"
    )


# ============================================================
# PLOT 2: WEIGHT DISTRIBUTION FAN
# ============================================================

def plot_weight_distribution(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.fill_between(
        temp["year"],
        temp["q10_weight"],
        temp["q90_weight"],
        color=CYAN,
        alpha=0.12,
        label="10th-90th percentile"
    )

    ax.fill_between(
        temp["year"],
        temp["q25_weight"],
        temp["q75_weight"],
        color=PURPLE,
        alpha=0.20,
        label="25th-75th percentile"
    )

    ax.plot(
        temp["year"],
        temp["median_weight"],
        color=TEXT,
        linewidth=2.2,
        label="Median"
    )

    ax.plot(
        temp["year"],
        temp["mean_weight"],
        color=ORANGE,
        linewidth=2,
        label="Mean"
    )

    ax.plot(
        temp["year"],
        temp["record_max"],
        color=PINK,
        linewidth=2.5,
        marker="o",
        label="Record"
    )

    ax.set_title(
        "Weight Distribution Through Time",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Weight (lb)"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    return save_fig(
        fig,
        "02_weight_distribution.png"
    )


# ============================================================
# PLOT 3: YEAR TO YEAR IMPROVEMENT
# ============================================================

def plot_record_improvement(
    annual
):

    temp = annual[
        annual["year"].between(
            2014,
            PRIMARY_END
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    values = temp[
        "record_improvement"
    ]

    ax.axhline(
        0,
        color=MUTED,
        linewidth=1
    )

    ax.bar(
        temp["year"],
        values,
        color=[
            GREEN if v >= 0 else RED
            for v in values
        ],
        alpha=0.85
    )

    fit = linear_regression(
        temp["year"],
        values
    )

    if fit:

        xx = np.linspace(
            temp["year"].min(),
            temp["year"].max(),
            200
        )

        yy = (
            fit["intercept"]
            +
            fit["slope"] * xx
        )

        ax.plot(
            xx,
            yy,
            color=BLUE,
            linewidth=2.5,
            label="Trend in annual improvement"
        )

        ax.text(
            0.02,
            0.95,
            (
                f"slope = {fit['slope']:.2f} lb/year²\n"
                f"p = {fit['p_value']:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            color=TEXT
        )

    ax.set_title(
        "Year-to-Year Change in the Record",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Change from previous year's record (lb)"
    )

    return save_fig(
        fig,
        "03_record_improvement.png"
    )


# ============================================================
# PLOT 4: MODEL COMPARISON
# ============================================================

def plot_model_comparison(
    annual,
    model_info
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    x = temp["year"].to_numpy(float)
    y = temp["record_max"].to_numpy(float)

    linear = model_info["linear"]
    quadratic = model_info["quadratic"]
    cp = model_info["cp_model"]

    xx = np.linspace(
        x.min(),
        x.max(),
        500
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.scatter(
        x,
        y,
        color=ORANGE,
        s=80,
        edgecolor=TEXT,
        linewidth=0.7,
        zorder=5,
        label="Observed record"
    )

    linear_y = (
        linear["intercept"]
        +
        linear["slope"] * xx
    )

    ax.plot(
        xx,
        linear_y,
        color=BLUE,
        linewidth=2.5,
        label="Linear"
    )

    beta = quadratic["beta"]

    quad_y = (
        beta[0]
        +
        beta[1] * xx
        +
        beta[2] * xx ** 2
    )

    ax.plot(
        xx,
        quad_y,
        color=PURPLE,
        linewidth=2.5,
        label="Quadratic"
    )

    if cp is not None:

        hinge = np.maximum(
            0,
            xx - cp["breakpoint"]
        )

        cp_y = (
            cp["beta"][0]
            +
            cp["beta"][1] * xx
            +
            cp["beta"][2] * hinge
        )

        ax.plot(
            xx,
            cp_y,
            color=GREEN,
            linewidth=2.5,
            linestyle="--",
            label=(
                f"Piecewise, breakpoint "
                f"{cp['breakpoint']:.0f}"
            )
        )

        ax.axvline(
            cp["breakpoint"],
            color=GREEN,
            alpha=0.3,
            linestyle=":"
        )

    ax.set_title(
        "Linear vs Quadratic vs Piecewise Record Models",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Record weight (lb)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "04_record_model_comparison.png"
    )


# ============================================================
# PLOT 5: LEAVE ONE OUT
# ============================================================

def plot_loo(
    loo
):

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.axhline(
        loo["slope"].mean(),
        color=BLUE,
        linewidth=2,
        linestyle="--",
        label="Mean leave-one-out slope"
    )

    ax.scatter(
        loo["omitted_year"],
        loo["slope"],
        s=90,
        color=ORANGE,
        edgecolor=TEXT,
        linewidth=0.7
    )

    ax.set_title(
        "Sensitivity of Record Trend to Individual Years",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Omitted year"
    )

    ax.set_ylabel(
        "Estimated slope (lb/year)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "05_leave_one_out_sensitivity.png"
    )


# ============================================================
# PLOT 6: RESIDUALS
# ============================================================

def plot_residuals(
    model_info
):

    model = model_info["linear"]

    x = model["x"]
    fitted = model["fitted"]
    resid = model["residuals"]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.axhline(
        0,
        color=MUTED,
        linewidth=1
    )

    ax.scatter(
        x,
        resid,
        s=80,
        color=CYAN,
        edgecolor=TEXT,
        linewidth=0.7
    )

    ax.plot(
        x,
        np.zeros_like(x),
        color=BLUE,
        alpha=0.3
    )

    ax.set_title(
        "Linear Model Residuals",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Residual (lb)"
    )

    return save_fig(
        fig,
        "06_record_model_residuals.png"
    )


# ============================================================
# PLOT 7: PARENT EFFECTIVE NUMBERS
# ============================================================

def plot_parent_effective(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        temp["year"],
        temp["mother_effective"],
        color=CYAN,
        linewidth=2.8,
        marker="o",
        label="Seed mother"
    )

    ax.plot(
        temp["year"],
        temp["pollinator_effective"],
        color=PINK,
        linewidth=2.8,
        marker="o",
        label="Pollinator father"
    )

    ax.set_title(
        "Effective Documented Parent Number",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Effective number of parents"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "07_effective_parent_number.png"
    )


# ============================================================
# PLOT 8: TOP 10 PARENT SHARE
# ============================================================

def plot_parent_top10(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        temp["year"],
        temp["mother_top10_share"] * 100,
        color=CYAN,
        linewidth=2.5,
        marker="o",
        label="Mother"
    )

    ax.plot(
        temp["year"],
        temp["pollinator_top10_share"] * 100,
        color=PINK,
        linewidth=2.5,
        marker="o",
        label="Pollinator"
    )

    ax.set_title(
        "Share of Documented Parent Records Held by Top 10 Parents",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Top-10 share (%)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "08_top10_parent_share.png"
    )


# ============================================================
# PLOT 9: PARENT SIMPSON / CONCENTRATION
# ============================================================

def plot_parent_concentration(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        temp["year"],
        temp["mother_simpson"],
        color=CYAN,
        linewidth=2.5,
        marker="o",
        label="Mother Simpson concentration"
    )

    ax.plot(
        temp["year"],
        temp["pollinator_simpson"],
        color=PINK,
        linewidth=2.5,
        marker="o",
        label="Pollinator Simpson concentration"
    )

    ax.set_title(
        "Pedigree Concentration Through Time",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Simpson concentration"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "09_pedigree_concentration.png"
    )


# ============================================================
# PLOT 10: PARENT JACCARD HEATMAP
# ============================================================

def plot_parent_jaccard(
    matrix
):

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="mako",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={
            "label":
                "Jaccard similarity"
        }
    )

    ax.set_title(
        "Year-to-Year Pedigree Parent Overlap",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Year"
    )

    return save_fig(
        fig,
        "10_parent_jaccard_heatmap.png"
    )


# ============================================================
# PLOT 11: PARENT LORENZ
# ============================================================

def plot_parent_lorenz(
    primary,
    column
):

    periods = {
        "2013-2016": (
            2013,
            2016,
            CYAN
        ),
        "2017-2020": (
            2017,
            2020,
            PURPLE
        ),
        "2021-2025": (
            2021,
            2025,
            PINK
        ),
    }

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=MUTED,
        linestyle="--",
        label="Equal distribution"
    )

    for label, (start, end, color) in periods.items():

        g = primary[
            primary["year"].between(
                start,
                end
            )
        ]

        values = g[column].map(
            norm_text
        )

        named = values[
            values.map(classify_parent) == "named"
        ]

        counts = (
            named.value_counts()
            .sort_values()
        )

        if len(counts) == 0:
            continue

        shares = (
            counts /
            counts.sum()
        )

        cumulative_parents = (
            np.arange(1, len(counts) + 1)
            / len(counts)
        )

        cumulative_records = np.cumsum(
            shares
        )

        ax.plot(
            cumulative_parents,
            cumulative_records,
            linewidth=2.5,
            color=color,
            label=label
        )

    ax.set_title(
        "Pedigree Parent Concentration: Lorenz Curves",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Cumulative fraction of named parents"
    )

    ax.set_ylabel(
        "Cumulative fraction of parent records"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "11_parent_lorenz.png"
    )


# ============================================================
# PLOT 12: RECURRING PARENTS
# ============================================================

def plot_recurring_parents(
    frequency
):

    if len(frequency) == 0:
        return None

    top = (
        frequency
        .head(15)
        .sort_values("years_present")
    )

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    labels = [
        str(x)[:28]
        for x in top["parent"]
    ]

    ax.barh(
        labels,
        top["years_present"],
        color=PURPLE,
        alpha=0.85
    )

    ax.set_title(
        "Most Persistent Documented Parent Lines",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Years present in dataset"
    )

    return save_fig(
        fig,
        "12_recurring_parent_lines.png"
    )


# ============================================================
# PLOT 13: GEOGRAPHIC EXPANSION
# ============================================================

def plot_geographic_expansion(
    geo
):

    metrics = [
        (
            "countries",
            "Countries",
            CYAN
        ),
        (
            "states_provinces",
            "States / provinces",
            GREEN
        ),
        (
            "cities",
            "Cities",
            ORANGE
        ),
        (
            "gpc_sites",
            "GPC sites",
            PINK
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    axes = axes.ravel()

    for ax, (
        metric,
        label,
        color
    ) in zip(
        axes,
        metrics
    ):

        ax.plot(
            geo["year"],
            geo[metric],
            color=color,
            linewidth=2.8,
            marker="o"
        )

        ax.set_title(
            label,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Year"
        )

        ax.set_ylabel(
            "Count"
        )

    fig.suptitle(
        "Geographic Expansion of the Recorded Competitive Pumpkin Dataset",
        fontsize=19,
        fontweight="bold",
        color=TEXT
    )

    return save_fig(
        fig,
        "13_geographic_expansion.png"
    )


# ============================================================
# PLOT 14: COUNTRY CONCENTRATION
# ============================================================

def plot_country_concentration(
    countries
):

    fig, ax1 = plt.subplots(
        figsize=(12, 7)
    )

    ax1.plot(
        countries["year"],
        countries["country_hhi"],
        color=PINK,
        linewidth=2.7,
        marker="o",
        label="HHI"
    )

    ax1.set_ylabel(
        "Country HHI",
        color=PINK
    )

    ax2 = ax1.twinx()

    ax2.plot(
        countries["year"],
        countries["effective_countries"],
        color=CYAN,
        linewidth=2.7,
        marker="o",
        label="Effective countries"
    )

    ax2.set_ylabel(
        "Effective number of countries",
        color=CYAN
    )

    ax1.set_xlabel(
        "Year"
    )

    ax1.set_title(
        "Country Representation Concentration",
        fontsize=20,
        fontweight="bold"
    )

    return save_fig(
        fig,
        "14_country_concentration.png"
    )


# ============================================================
# PLOT 15: TOP COUNTRY SHARES
# ============================================================

def plot_top_country_shares(
    countries
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        countries["year"],
        countries["country_top1"] * 100,
        color=PINK,
        linewidth=2.5,
        marker="o",
        label="Top 1"
    )

    ax.plot(
        countries["year"],
        countries["country_top3"] * 100,
        color=PURPLE,
        linewidth=2.5,
        marker="o",
        label="Top 3"
    )

    ax.plot(
        countries["year"],
        countries["country_top5"] * 100,
        color=CYAN,
        linewidth=2.5,
        marker="o",
        label="Top 5"
    )

    ax.set_title(
        "Country Representation Concentration",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Share of records (%)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "15_top_country_shares.png"
    )


# ============================================================
# PLOT 16: COUNTRY HEATMAP
# ============================================================

def plot_country_heatmap(
    shares
):

    totals = (
        shares
        .groupby("country")["records"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_countries = (
        totals
        .head(12)
        .index
        .tolist()
    )

    temp = shares[
        shares["country"].isin(
            top_countries
        )
    ]

    pivot = temp.pivot(
        index="country",
        columns="year",
        values="share"
    ).fillna(0)

    pivot = pivot.loc[
        top_countries
    ]

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    sns.heatmap(
        pivot * 100,
        cmap="mako",
        annot=True,
        fmt=".1f",
        linewidths=0.3,
        cbar_kws={
            "label":
                "Share of records (%)"
        }
    )

    ax.set_title(
        "Top Countries by Annual Dataset Representation",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Country"
    )

    return save_fig(
        fig,
        "16_country_share_heatmap.png"
    )


# ============================================================
# PLOT 17: EARLY VS LATE COUNTRY CHANGE
# ============================================================

def plot_country_change(
    comparison
):

    temp = comparison.copy()

    # Only show countries with meaningful late representation
    temp = temp[
        (
            temp["early_share"] > 0.005
        )
        |
        (
            temp["late_share"] > 0.005
        )
    ]

    temp = temp.sort_values(
        "change"
    )

    fig, ax = plt.subplots(
        figsize=(11, 10)
    )

    ax.axvline(
        0,
        color=MUTED,
        linewidth=1
    )

    ax.barh(
        temp["country"],
        temp["change"] * 100,
        color=[
            GREEN if x > 0 else RED
            for x in temp["change"]
        ],
        alpha=0.85
    )

    ax.set_title(
        "Change in Country Representation: Early vs Late Period",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Change in share (percentage points)"
    )

    return save_fig(
        fig,
        "17_country_representation_change.png"
    )


# ============================================================
# PLOT 18: GROWER ECOSYSTEM
# ============================================================

def plot_grower_ecosystem(
    growers
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        growers["year"],
        growers["unique_growers"],
        color=CYAN,
        linewidth=2.7,
        marker="o",
        label="Unique growers"
    )

    ax2 = ax.twinx()

    ax2.plot(
        growers["year"],
        growers["new_grower_fraction"] * 100,
        color=ORANGE,
        linewidth=2.5,
        marker="o",
        label="New-grower fraction"
    )

    ax2.set_ylabel(
        "New growers among annual growers (%)",
        color=ORANGE
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Unique growers",
        color=CYAN
    )

    ax.set_title(
        "Grower Ecosystem Through Time",
        fontsize=20,
        fontweight="bold"
    )

    return save_fig(
        fig,
        "18_grower_ecosystem.png"
    )


# ============================================================
# PLOT 19: RECORD VS GEOGRAPHIC DIVERSITY
# ============================================================

def plot_record_vs_countries(
    annual
):

    temp = annual[
        [
            "record_max",
            "effective_countries"
        ]
    ].dropna()

    if len(temp) < 4:
        return None

    x = temp[
        "effective_countries"
    ].to_numpy(float)

    y = temp[
        "record_max"
    ].to_numpy(float)

    fit = linear_regression(
        x,
        y
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        x,
        y,
        color=ORANGE,
        s=100,
        edgecolor=TEXT,
        linewidth=0.8
    )

    if fit:

        xx = np.linspace(
            x.min(),
            x.max(),
            200
        )

        yy = (
            fit["intercept"]
            +
            fit["slope"] * xx
        )

        ax.plot(
            xx,
            yy,
            color=BLUE,
            linewidth=2.5
        )

        rho, p = stats.spearmanr(
            x,
            y
        )

        ax.text(
            0.03,
            0.95,
            (
                f"Pearson r = "
                f"{stats.pearsonr(x, y).statistic:.3f}\n"
                f"Spearman ρ = {rho:.3f}\n"
                f"p = {p:.3f}"
            ),
            transform=ax.transAxes,
            va="top",
            color=TEXT
        )

    ax.set_title(
        "Record Weight vs Effective Country Representation",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Effective number of countries"
    )

    ax.set_ylabel(
        "Record weight (lb)"
    )

    return save_fig(
        fig,
        "19_record_vs_country_diversity.png"
    )


# ============================================================
# PLOT 20: RECORD VS PEDIGREE DIVERSITY
# ============================================================

def plot_record_vs_pedigree(
    annual
):

    temp = annual[
        [
            "record_max",
            "mother_effective",
            "pollinator_effective"
        ]
    ].dropna()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    for ax, metric, label, color in [
        (
            axes[0],
            "mother_effective",
            "Effective mother number",
            CYAN
        ),
        (
            axes[1],
            "pollinator_effective",
            "Effective pollinator number",
            PINK
        ),
    ]:

        x = temp[metric].to_numpy(float)
        y = temp["record_max"].to_numpy(float)

        ax.scatter(
            x,
            y,
            color=color,
            s=90,
            edgecolor=TEXT,
            linewidth=0.7
        )

        if len(x) >= 4:

            fit = linear_regression(
                x,
                y
            )

            xx = np.linspace(
                x.min(),
                x.max(),
                200
            )

            yy = (
                fit["intercept"]
                +
                fit["slope"] * xx
            )

            ax.plot(
                xx,
                yy,
                color=BLUE,
                linewidth=2
            )

            rho, p = stats.spearmanr(
                x,
                y
            )

            ax.text(
                0.04,
                0.95,
                (
                    f"ρ = {rho:.3f}\n"
                    f"p = {p:.3f}"
                ),
                transform=ax.transAxes,
                va="top",
                color=TEXT
            )

        ax.set_title(
            label,
            fontweight="bold"
        )

        ax.set_xlabel(
            label
        )

        ax.set_ylabel(
            "Record weight (lb)"
        )

    fig.suptitle(
        "Record Weight vs Documented Pedigree Diversity",
        fontsize=18,
        fontweight="bold",
        color=TEXT
    )

    return save_fig(
        fig,
        "20_record_vs_pedigree_diversity.png"
    )


# ============================================================
# PLOT 21: ESTIMATED VS ACTUAL
# ============================================================

def plot_estimated_vs_actual(
    primary
):

    if "estimated_weight" not in primary.columns:
        return None

    temp = primary[
        [
            "weight_lbs",
            "estimated_weight"
        ]
    ].dropna()

    if len(temp) < 10:
        return None

    actual = temp[
        "weight_lbs"
    ].to_numpy(float)

    estimated = temp[
        "estimated_weight"
    ].to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # Downsample for rendering
    if len(actual) > 10000:

        idx = rng.choice(
            len(actual),
            size=10000,
            replace=False
        )

        actual_plot = actual[idx]
        estimated_plot = estimated[idx]

    else:

        actual_plot = actual
        estimated_plot = estimated

    ax.scatter(
        actual_plot,
        estimated_plot,
        s=8,
        alpha=0.18,
        color=CYAN
    )

    lo = min(
        actual.min(),
        estimated.min()
    )

    hi = max(
        actual.max(),
        estimated.max()
    )

    ax.plot(
        [lo, hi],
        [lo, hi],
        color=ORANGE,
        linewidth=2.5,
        label="1:1 line"
    )

    error = estimated - actual

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    r = stats.pearsonr(
        actual,
        estimated
    ).statistic

    ax.text(
        0.04,
        0.95,
        (
            f"n = {len(temp):,}\n"
            f"r = {r:.3f}\n"
            f"MAE = {mae:.1f} lb\n"
            f"RMSE = {rmse:.1f} lb"
        ),
        transform=ax.transAxes,
        va="top",
        color=TEXT
    )

    ax.set_title(
        "Chart-Estimated Weight vs Actual Weight",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Actual weight (lb)"
    )

    ax.set_ylabel(
        "Estimated weight (lb)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "21_estimated_vs_actual.png"
    )


# ============================================================
# PLOT 22: OTT
# ============================================================

def plot_ott(
    primary
):

    if "ott" not in primary.columns:
        return None

    temp = primary[
        [
            "weight_lbs",
            "ott"
        ]
    ].dropna()

    if len(temp) < 10:
        return None

    x = temp["ott"].to_numpy(float)
    y = temp["weight_lbs"].to_numpy(float)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    hb = ax.hexbin(
        x,
        y,
        gridsize=45,
        mincnt=1,
        cmap="mako"
    )

    fig.colorbar(
        hb,
        ax=ax,
        label="Records"
    )

    ax.set_title(
        "OTT Measurement vs Actual Pumpkin Weight",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "OTT"
    )

    ax.set_ylabel(
        "Weight (lb)"
    )

    return save_fig(
        fig,
        "22_ott_vs_weight.png"
    )


# ============================================================
# PLOT 23: CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(
    annual
):

    metrics = [
        "record_max",
        "mean_weight",
        "mother_effective",
        "pollinator_effective",
        "countries",
        "gpc_sites",
        "effective_countries",
        "country_hhi",
        "unique_growers",
    ]

    metrics = [
        x for x in metrics
        if x in annual.columns
    ]

    temp = annual[
        metrics
    ]

    corr = temp.corr(
        method="spearman"
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    sns.heatmap(
        corr,
        ax=ax,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        square=True
    )

    ax.set_title(
        "Spearman Correlation Matrix of Annual Metrics",
        fontsize=18,
        fontweight="bold"
    )

    return save_fig(
        fig,
        "23_annual_metric_correlation_heatmap.png"
    )


# ============================================================
# PLOT 24: PERIOD WEIGHT DISTRIBUTIONS
# ============================================================

def plot_period_distributions(
    primary
):

    df = primary[
        ["year", "weight_lbs"]
    ].dropna().copy()

    df["period"] = pd.cut(
        df["year"],
        bins=[
            2012,
            2016,
            2020,
            2025
        ],
        labels=[
            "2013-2016",
            "2017-2020",
            "2021-2025"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    sns.violinplot(
        data=df,
        x="period",
        y="weight_lbs",
        inner="quartile",
        cut=0,
        linewidth=1,
        ax=ax
    )

    ax.set_title(
        "Annual Weight Distributions by Period",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Period"
    )

    ax.set_ylabel(
        "Weight (lb)"
    )

    return save_fig(
        fig,
        "24_period_weight_distributions.png"
    )


# ============================================================
# PLOT 25: CUMULATIVE RECORD PROGRESS
# ============================================================

def plot_cumulative_record(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ].copy()

    first = temp[
        "record_max"
    ].iloc[0]

    temp["gain_from_2013"] = (
        temp["record_max"] -
        first
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.fill_between(
        temp["year"],
        0,
        temp["gain_from_2013"],
        color=PURPLE,
        alpha=0.18
    )

    ax.plot(
        temp["year"],
        temp["gain_from_2013"],
        color=PURPLE,
        linewidth=3,
        marker="o"
    )

    ax.axhline(
        0,
        color=MUTED,
        linewidth=1
    )

    ax.set_title(
        "Cumulative Record Gain Relative to 2013",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Gain in record weight (lb)"
    )

    return save_fig(
        fig,
        "25_cumulative_record_gain.png"
    )


# ============================================================
# PLOT 26: ANNUAL RECORD VS MEAN
# ============================================================

def plot_record_vs_mean(
    annual
):

    temp = annual[
        annual["year"].between(
            PRIMARY_START,
            PRIMARY_END
        )
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        temp["year"],
        temp["record_max"],
        color=PINK,
        linewidth=2.8,
        marker="o",
        label="Record"
    )

    ax.plot(
        temp["year"],
        temp["mean_weight"],
        color=CYAN,
        linewidth=2.8,
        marker="o",
        label="Mean"
    )

    ax.fill_between(
        temp["year"],
        temp["mean_weight"],
        temp["record_max"],
        color=PURPLE,
        alpha=0.08
    )

    ax.set_title(
        "Record vs Mean Pumpkin Weight",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Weight (lb)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "26_record_vs_mean.png"
    )


# ============================================================
# PLOT 27: COUNTRY JACCARD
# ============================================================

def plot_country_jaccard(
    countries
):

    temp = countries.dropna(
        subset=[
            "country_jaccard_previous"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        temp["year"],
        temp["country_jaccard_previous"],
        color=GREEN,
        linewidth=2.7,
        marker="o"
    )

    ax.set_title(
        "Year-to-Year Country Set Overlap",
        fontsize=19,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Jaccard similarity"
    )

    return save_fig(
        fig,
        "27_country_jaccard.png"
    )


# ============================================================
# PLOT 28: NEW PARENT FRACTION
# ============================================================

def plot_new_parent_fraction(
    annual
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        annual["year"],
        annual["mother_new_parent_fraction"] * 100,
        color=CYAN,
        linewidth=2.5,
        marker="o",
        label="Mother"
    )

    ax.plot(
        annual["year"],
        annual["pollinator_new_parent_fraction"] * 100,
        color=PINK,
        linewidth=2.5,
        marker="o",
        label="Pollinator"
    )

    ax.set_title(
        "Fraction of Annual Named Parents Not Seen Previously",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "New-parent fraction (%)"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "28_new_parent_fraction.png"
    )


# ============================================================
# PLOT 29: COUNTRY SHANNON
# ============================================================

def plot_country_shannon(
    countries
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.plot(
        countries["year"],
        countries["country_shannon"],
        color=PURPLE,
        linewidth=2.8,
        marker="o",
        label="Shannon entropy"
    )

    ax.plot(
        countries["year"],
        countries["effective_countries_shannon"],
        color=CYAN,
        linewidth=2.8,
        marker="o",
        label="Effective Shannon countries"
    )

    ax.set_title(
        "Country Representation Diversity",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Year"
    )

    ax.set_ylabel(
        "Diversity metric"
    )

    ax.legend(
        frameon=False
    )

    return save_fig(
        fig,
        "29_country_shannon_diversity.png"
    )


# ============================================================
# GALLERY
# ============================================================

def make_gallery():

    images = sorted(
        [
            p for p in FIG_DIR.glob("*.png")
            if p.name != "00_gallery.png"
        ]
    )

    if not images:
        return None

    cols = 3
    rows = math.ceil(
        len(images) / cols
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            18,
            rows * 6
        )
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor(BG)

    for ax, path in zip(
        axes,
        images
    ):

        try:
            img = plt.imread(
                path
            )

            ax.imshow(img)
            ax.set_title(
                path.stem.replace(
                    "_",
                    " "
                ),
                color=TEXT,
                fontsize=11
            )
            ax.axis("off")

        except Exception:
            ax.axis("off")

    fig.patch.set_facecolor(BG)

    fig.savefig(
        FIG_DIR / "00_gallery.png",
        dpi=110,
        bbox_inches="tight",
        facecolor=BG
    )

    plt.close(fig)

    return FIG_DIR / "00_gallery.png"


# ============================================================
# SUMMARY TEXT
# ============================================================

def write_summary(
    annual,
    model_info,
    countries,
    geo,
    trends,
    early_late
):

    lines = []

    lines.append(
        "GIANT PUMPKIN MEGA ANALYSIS"
    )

    lines.append(
        "=" * 60
    )

    lines.append("")

    lines.append(
        f"Primary period: "
        f"{PRIMARY_START}-{PRIMARY_END}"
    )

    lines.append(
        f"Primary records: "
        f"{annual.loc[annual['year'].between(PRIMARY_START, PRIMARY_END), 'records'].sum():,}"
    )

    lines.append("")

    # Record trend
    fit = model_info["linear"]

    lines.append(
        "RECORD WEIGHT TREND"
    )

    lines.append(
        f"Slope: {fit['slope']:.4f} lb/year"
    )

    lines.append(
        f"95% CI: "
        f"[{fit['slope_ci_low']:.4f}, "
        f"{fit['slope_ci_high']:.4f}]"
    )

    lines.append(
        f"R²: {fit['r2']:.4f}"
    )

    lines.append(
        f"p: {fit['p_value']:.8g}"
    )

    lines.append("")

    # Quadratic
    quad = model_info["quadratic"]

    lines.append(
        "QUADRATIC MODEL"
    )

    lines.append(
        f"Quadratic coefficient: "
        f"{quad['beta'][2]:.8f}"
    )

    lines.append(
        f"R²: {quad['r2']:.4f}"
    )

    lines.append(
        f"AIC: {quad['aic']:.4f}"
    )

    lines.append(
        f"BIC: {quad['bic']:.4f}"
    )

    lines.append("")

    # Changepoint
    cp = model_info["cp_model"]

    if cp is not None:

        lines.append(
            "BEST PIECEWISE CHANGEPOINT"
        )

        lines.append(
            f"Breakpoint: "
            f"{cp['breakpoint']:.0f}"
        )

        lines.append(
            f"Slope before: "
            f"{cp['slope_before']:.4f}"
        )

        lines.append(
            f"Slope change: "
            f"{cp['slope_change']:.4f}"
        )

        lines.append(
            f"Slope after: "
            f"{cp['slope_after']:.4f}"
        )

        lines.append(
            f"BIC: {cp['bic']:.4f}"
        )

        lines.append("")

    # Geographic
    lines.append(
        "GEOGRAPHIC EXPANSION"
    )

    first_geo = geo.iloc[0]
    last_geo = geo.iloc[-1]

    for metric in [
        "countries",
        "states_provinces",
        "cities",
        "gpc_sites",
    ]:

        lines.append(
            f"{metric}: "
            f"{first_geo[metric]:.0f} -> "
            f"{last_geo[metric]:.0f}"
        )

    lines.append("")

    # Country concentration
    first_country = countries.iloc[0]
    last_country = countries.iloc[-1]

    lines.append(
        "COUNTRY CONCENTRATION"
    )

    lines.append(
        f"HHI: "
        f"{first_country['country_hhi']:.4f} -> "
        f"{last_country['country_hhi']:.4f}"
    )

    lines.append(
        f"Effective countries: "
        f"{first_country['effective_countries']:.3f} -> "
        f"{last_country['effective_countries']:.3f}"
    )

    lines.append(
        f"Top country share: "
        f"{first_country['country_top1'] * 100:.2f}% -> "
        f"{last_country['country_top1'] * 100:.2f}%"
    )

    lines.append("")

    # Significant trend rows
    if len(trends):

        sig = trends[
            trends["fdr_q"] < 0.05
        ].sort_values(
            "fdr_q"
        )

        lines.append(
            "FDR-SIGNIFICANT TREND TESTS"
        )

        if len(sig):

            for _, row in sig.head(30).iterrows():

                lines.append(
                    f"{row['metric']} | "
                    f"{row['test']} | "
                    f"effect={row['effect']:.6g} | "
                    f"p={row['p_value']:.4g} | "
                    f"q={row['fdr_q']:.4g}"
                )

        else:

            lines.append(
                "None at q < 0.05."
            )

    lines.append("")

    # Output
    lines.append(
        f"Tables: {TABLE_DIR}"
    )

    lines.append(
        f"Figures: {FIG_DIR}"
    )

    with open(
        OUTPUT_DIR / "analysis_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

# ============================================================
# DEEP RQ1 / RQ2 ANALYSIS PACK
# ============================================================

def _deep_yearly_parent_stats(primary, col, prefix):
    d = primary[["year", col, "weight_lbs"]].copy()
    d[col] = d[col].map(norm_text)
    d = d[(d[col] != "") & (~d[col].str.lower().isin(
        {"unknown", "unk", "none", "nan", "n/a", "na", "-", "0"}
    ))]
    years = sorted(d["year"].dropna().unique())
    rows = []
    prev = set()

    for y in years:
        cur = d[d["year"] == y]
        vals = cur[col].tolist()
        counts = pd.Series(vals).value_counts()
        names = set(counts.index)

        n = len(vals)
        k = len(names)
        probs = counts.to_numpy(float) / max(n, 1)
        entropy = -np.sum(probs * np.log(probs)) if len(probs) else np.nan
        effective = np.exp(entropy) if len(probs) else np.nan

        new_fraction = len(names - prev) / k if k else np.nan
        reuse_fraction = len(names & prev) / k if k else np.nan
        jaccard = len(names & prev) / len(names | prev) if (names | prev) else np.nan

        top10 = counts.head(10).sum() / n if n else np.nan
        hhi = np.sum(probs ** 2) if len(probs) else np.nan

        rows.append({
            "year": y,
            f"{prefix}_named_records": n,
            f"{prefix}_unique": k,
            f"{prefix}_entropy": entropy,
            f"{prefix}_effective_shannon": effective,
            f"{prefix}_new_parent_fraction": new_fraction,
            f"{prefix}_reused_parent_fraction": reuse_fraction,
            f"{prefix}_jaccard_previous": jaccard,
            f"{prefix}_top10_share": top10,
            f"{prefix}_hhi": hhi,
        })
        prev = names

    return pd.DataFrame(rows)


def _deep_plot(df, x, ys, filename, title, ylabel, labels=None):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for i, y in enumerate(ys):
        label = labels[i] if labels else y
        ax.plot(df[x], df[y], marker="o", linewidth=2, markersize=4, label=label)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    save_fig(fig, filename)


def _deep_trend(rows, df, metric, label):
    if metric not in df.columns:
        return
    x = pd.to_numeric(df["year"], errors="coerce")
    y = pd.to_numeric(df[metric], errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(y)
    if m.sum() < 4:
        return
    xx = x[m].to_numpy(float)
    yy = y[m].to_numpy(float)
    lr = stats.linregress(xx, yy)
    sp = stats.spearmanr(xx, yy)
    rows.append({
        "analysis": label,
        "metric": metric,
        "slope_per_year": lr.slope,
        "r_squared": lr.rvalue ** 2,
        "linear_p": lr.pvalue,
        "spearman_rho": sp.statistic,
        "spearman_p": sp.pvalue,
        "n_years": len(xx),
    })


def deep_rq1_rq2_analysis(primary, annual):
    """
    Twenty additional analyses, split evenly between RQ1 and RQ2.

    RQ1:
      1  parent completeness / documentation
      2  Shannon pedigree entropy
      3  annual parent novelty
      4  parent turnover
      5  parent persistence
      6  parent frequency / Zipf concentration
      7  pedigree completeness vs weight
      8  missing-parent weight bias
      9  elite-parent enrichment
     10  parent reuse vs performance

    RQ2:
     11  full weight-quantile trajectories
     12  Theil-Sen record trend
     13  record-to-P99 gap
     14  record-breaker margin
     15  rolling three-year record slope
     16  record acceleration
     17  annual weight volatility
     18  coefficient-of-variation trajectory
     19  record-model residual diagnostics
     20  all-candidate changepoint slopes
    """

    print("Running 20 additional deep RQ1/RQ2 analyses...")

    out = []

    # --------------------------------------------------------
    # RQ1-1: Documentation completeness
    # --------------------------------------------------------

    rows = []
    for y in sorted(primary["year"].unique()):
        d = primary[primary["year"] == y]
        m = d["seed_mother"].map(norm_text) != ""
        f = d["pollinator_father"].map(norm_text) != ""
        both = m & f
        rows.append({
            "year": y,
            "mother_documented_pct": 100 * m.mean(),
            "pollinator_documented_pct": 100 * f.mean(),
            "both_documented_pct": 100 * both.mean(),
        })
    completeness = pd.DataFrame(rows)
    save_table(completeness, "deep_rq1_01_parent_documentation_completeness.csv")
    _deep_plot(
        completeness, "year",
        ["mother_documented_pct", "pollinator_documented_pct", "both_documented_pct"],
        "deep_rq1_01_parent_documentation.png",
        "Parent documentation completeness",
        "Documented records (%)",
        ["Mother", "Pollinator", "Both"]
    )

    # --------------------------------------------------------
    # RQ1-2 to RQ1-4: entropy, novelty, turnover
    # --------------------------------------------------------

    mother_deep = _deep_yearly_parent_stats(primary, "seed_mother", "mother")
    poll_deep = _deep_yearly_parent_stats(primary, "pollinator_father", "pollinator")
    deep_parent = mother_deep.merge(poll_deep, on="year", how="outer").sort_values("year")
    save_table(deep_parent, "deep_rq1_02_to_04_parent_entropy_turnover.csv")

    _deep_plot(
        deep_parent, "year",
        ["mother_entropy", "pollinator_entropy"],
        "deep_rq1_02_parent_entropy.png",
        "Pedigree entropy over time",
        "Shannon entropy",
        ["Mother", "Pollinator"]
    )
    _deep_plot(
        deep_parent, "year",
        ["mother_new_parent_fraction", "pollinator_new_parent_fraction"],
        "deep_rq1_03_parent_novelty.png",
        "Annual parent novelty",
        "New parents among active parents (%)",
        ["Mother", "Pollinator"]
    )
    _deep_plot(
        deep_parent, "year",
        ["mother_jaccard_previous", "pollinator_jaccard_previous"],
        "deep_rq1_04_parent_turnover.png",
        "Year-to-year parent-set similarity",
        "Jaccard similarity",
        ["Mother", "Pollinator"]
    )

    # --------------------------------------------------------
    # RQ1-5: Parent persistence / lifespan
    # --------------------------------------------------------

    persistence_rows = []
    for col, prefix in [("seed_mother", "mother"), ("pollinator_father", "pollinator")]:
        d = primary[["year", col]].copy()
        d[col] = d[col].map(norm_text)
        d = d[d[col] != ""]
        spans = (
            d.groupby(col)["year"]
            .agg(first_year="min", last_year="max", years_seen="nunique")
        )
        for y in sorted(d["year"].unique()):
            active = d[d["year"] <= y]
            active_names = set(active[col])
            seen_to_date = d[d["year"] <= y].groupby(col)["year"].nunique()
            persistence_rows.append({
                "year": y,
                "parent_type": prefix,
                "active_parent_lines": len(active_names),
                "median_years_seen_to_date": float(seen_to_date.median()) if len(seen_to_date) else np.nan,
                "persistent_3plus_years_pct": 100 * (seen_to_date >= 3).mean() if len(seen_to_date) else np.nan,
            })
        save_table(spans.reset_index(), f"deep_rq1_05_{prefix}_parent_lifespans.csv")

    persistence = pd.DataFrame(persistence_rows)
    save_table(persistence, "deep_rq1_05_parent_persistence.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for prefix in ["mother", "pollinator"]:
        q = persistence[persistence.parent_type == prefix]
        ax.plot(q.year, q.persistent_3plus_years_pct, marker="o", label=prefix.title())
    ax.set_title("Persistence of recurring parent lines", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Parent lines seen in ≥3 years (%)")
    ax.legend()
    save_fig(fig, "deep_rq1_05_parent_persistence.png")

    # --------------------------------------------------------
    # RQ1-6: Frequency-rank / Zipf concentration
    # --------------------------------------------------------

    zipf_rows = []
    for col, prefix in [("seed_mother", "mother"), ("pollinator_father", "pollinator")]:
        for y in sorted(primary["year"].unique()):
            d = primary[primary.year == y][col].map(norm_text)
            d = d[d != ""]
            vc = d.value_counts().to_numpy(float)
            if len(vc) >= 3:
                ranks = np.arange(1, len(vc) + 1, dtype=float)
                lr = stats.linregress(np.log(ranks), np.log(vc))
                zipf_rows.append({
                    "year": y, "parent_type": prefix,
                    "zipf_slope": lr.slope,
                    "zipf_r2": lr.rvalue ** 2,
                    "unique_parents": len(vc),
                })
    zipf = pd.DataFrame(zipf_rows)
    save_table(zipf, "deep_rq1_06_parent_frequency_zipf.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for prefix in ["mother", "pollinator"]:
        q = zipf[zipf.parent_type == prefix]
        ax.plot(q.year, q.zipf_slope, marker="o", label=prefix.title())
    ax.set_title("Parent frequency-rank concentration", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Log frequency-rank slope")
    ax.legend()
    save_fig(fig, "deep_rq1_06_parent_zipf.png")

    # --------------------------------------------------------
    # RQ1-7: Pedigree completeness vs pumpkin weight
    # --------------------------------------------------------

    d = primary.copy()
    mother_named = d.seed_mother.map(norm_text) != ""
    father_named = d.pollinator_father.map(norm_text) != ""
    d["pedigree_complete"] = np.select(
        [mother_named & father_named, mother_named | father_named],
        ["Both parents", "One parent"],
        default="No parents"
    )
    comp_weight = (
        d.groupby(["year", "pedigree_complete"])["weight_lbs"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    save_table(comp_weight, "deep_rq1_07_pedigree_completeness_weight.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for cat in ["Both parents", "One parent", "No parents"]:
        q = comp_weight[comp_weight.pedigree_complete == cat]
        ax.plot(q.year, q["mean"], marker="o", label=cat)
    ax.set_title("Mean weight by pedigree documentation completeness", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean weight (lb)")
    ax.legend()
    save_fig(fig, "deep_rq1_07_pedigree_completeness_weight.png")

    # --------------------------------------------------------
    # RQ1-8: Missing-parent weight bias
    # --------------------------------------------------------

    bias_rows = []
    for y in sorted(d.year.unique()):
        q = d[d.year == y]
        for typ, mask in [
            ("mother", mother_named),
            ("pollinator", father_named),
            ("both", mother_named & father_named),
        ]:
            present = q[mask[q.index]]
            missing = q[~mask[q.index]]
            if len(present) and len(missing):
                diff = present.weight_lbs.mean() - missing.weight_lbs.mean()
                bias_rows.append({
                    "year": y, "parent_type": typ,
                    "documented_mean_weight": present.weight_lbs.mean(),
                    "undocumented_mean_weight": missing.weight_lbs.mean(),
                    "difference_lbs": diff,
                })
    bias = pd.DataFrame(bias_rows)
    save_table(bias, "deep_rq1_08_missing_parent_weight_bias.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for typ in ["mother", "pollinator", "both"]:
        q = bias[bias.parent_type == typ]
        ax.plot(q.year, q.difference_lbs, marker="o", label=typ.title())
    ax.axhline(0, linewidth=1)
    ax.set_title("Weight difference associated with parent documentation", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Documented minus undocumented mean (lb)")
    ax.legend()
    save_fig(fig, "deep_rq1_08_missing_parent_bias.png")

    # --------------------------------------------------------
    # RQ1-9: Elite-parent enrichment
    # --------------------------------------------------------

    elite_rows = []
    for y in sorted(d.year.unique()):
        q = d[d.year == y].dropna(subset=["weight_lbs"]).copy()
        if len(q) < 20:
            continue
        cutoff = q.weight_lbs.quantile(.99)
        elite = q[q.weight_lbs >= cutoff]
        for col, prefix in [("seed_mother", "mother"), ("pollinator_father", "pollinator")]:
            allp = q[col].map(norm_text)
            elp = elite[col].map(norm_text)
            allp = allp[allp != ""]
            elp = elp[elp != ""]
            if len(allp) and len(elp):
                ac = allp.value_counts()
                ec = elp.value_counts()
                for topn in [1, 5, 10]:
                    allshare = ac.head(topn).sum() / len(allp)
                    elshare = ec.head(topn).sum() / len(elp)
                    elite_rows.append({
                        "year": y, "parent_type": prefix, "top_n": topn,
                        "overall_share": allshare,
                        "elite_share": elshare,
                        "enrichment_pp": 100 * (elshare - allshare),
                    })
    elite = pd.DataFrame(elite_rows)
    save_table(elite, "deep_rq1_09_elite_parent_enrichment.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for prefix in ["mother", "pollinator"]:
        q = elite[(elite.parent_type == prefix) & (elite.top_n == 10)]
        ax.plot(q.year, q.enrichment_pp, marker="o", label=prefix.title())
    ax.axhline(0, linewidth=1)
    ax.set_title("Top-10 parent enrichment in the annual top 1%", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Elite minus overall share (percentage points)")
    ax.legend()
    save_fig(fig, "deep_rq1_09_elite_parent_enrichment.png")

    # --------------------------------------------------------
    # RQ1-10: Parent reuse vs performance
    # --------------------------------------------------------

    reuse_rows = []
    for col, prefix in [("seed_mother", "mother"), ("pollinator_father", "pollinator")]:
        d2 = d.copy()
        d2[col] = d2[col].map(norm_text)
        d2 = d2[d2[col] != ""]
        reuse = d2.groupby(col)["year"].nunique()
        d2["years_parent_seen"] = d2[col].map(reuse)
        for y, q in d2.groupby("year"):
            rho, pv = stats.spearmanr(q["years_parent_seen"], q["weight_lbs"])
            reuse_rows.append({
                "year": y, "parent_type": prefix,
                "spearman_rho": rho, "p_value": pv,
                "mean_weight": q.weight_lbs.mean(),
            })
    reuse_perf = pd.DataFrame(reuse_rows)
    save_table(reuse_perf, "deep_rq1_10_parent_reuse_vs_weight.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for prefix in ["mother", "pollinator"]:
        q = reuse_perf[reuse_perf.parent_type == prefix]
        ax.plot(q.year, q.spearman_rho, marker="o", label=prefix.title())
    ax.axhline(0, linewidth=1)
    ax.set_title("Association between parent reuse and pumpkin weight", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Spearman correlation")
    ax.legend()
    save_fig(fig, "deep_rq1_10_parent_reuse_weight.png")

    # --------------------------------------------------------
    # RQ2-11: Quantile trajectories
    # --------------------------------------------------------

    quant_rows = []
    for y, q in d.groupby("year"):
        w = q.weight_lbs.dropna()
        quant_rows.append({
            "year": y,
            "p50": w.quantile(.50),
            "p75": w.quantile(.75),
            "p90": w.quantile(.90),
            "p95": w.quantile(.95),
            "p99": w.quantile(.99),
            "p99_9": w.quantile(.999),
            "max": w.max(),
        })
    quant = pd.DataFrame(quant_rows).sort_values("year")
    save_table(quant, "deep_rq2_11_weight_quantile_trajectories.csv")

    _deep_plot(
        quant, "year",
        ["p50", "p75", "p90", "p95", "p99", "max"],
        "deep_rq2_11_weight_quantiles.png",
        "Weight distribution trajectories",
        "Weight (lb)",
        ["Median", "P75", "P90", "P95", "P99", "Maximum"]
    )

    # --------------------------------------------------------
    # RQ2-12: Theil-Sen robust record trend
    # --------------------------------------------------------

    q = quant[quant.year <= PRIMARY_END]
    sen = stats.theilslopes(q["max"], q["year"], 0.95)
    save_table(pd.DataFrame([{
        "slope_lbs_per_year": sen.slope,
        "intercept": sen.intercept,
        "lower_slope_95ci": sen.low_slope,
        "upper_slope_95ci": sen.high_slope,
        "n_years": len(q),
    }]), "deep_rq2_12_theil_sen_record_trend.csv")

    # --------------------------------------------------------
    # RQ2-13: Record-to-P99 gap
    # --------------------------------------------------------

    quant["record_minus_p99"] = quant["max"] - quant["p99"]
    quant["record_p99_ratio"] = quant["max"] / quant["p99"]
    save_table(
        quant[["year", "record_minus_p99", "record_p99_ratio"]],
        "deep_rq2_13_record_extremeness.csv"
    )
    _deep_plot(
        quant, "year", ["record_minus_p99"],
        "deep_rq2_13_record_minus_p99.png",
        "Distance between record and P99",
        "Record minus P99 (lb)",
        ["Record − P99"]
    )

    # --------------------------------------------------------
    # RQ2-14: Record-breaker margin
    # --------------------------------------------------------

    records = quant[quant.year <= PRIMARY_END].copy()
    records["previous_record"] = records["max"].cummax().shift(1)
    records["record_breaker"] = records["max"] > records["previous_record"]
    records["break_margin_lbs"] = np.where(
        records.record_breaker,
        records["max"] - records["previous_record"],
        np.nan
    )
    save_table(records, "deep_rq2_14_record_breakers.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    q = records.dropna(subset=["break_margin_lbs"])
    ax.bar(q.year.astype(str), q.break_margin_lbs)
    ax.set_title("Yearly improvement when a new record was set", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("New record margin (lb)")
    save_fig(fig, "deep_rq2_14_record_breaker_margins.png")

    # --------------------------------------------------------
    # RQ2-15: Rolling three-year record slope
    # --------------------------------------------------------

    roll_rows = []
    vals = records.reset_index(drop=True)
    for i in range(2, len(vals)):
        z = vals.iloc[i-2:i+1]
        lr = stats.linregress(z.year, z["max"])
        roll_rows.append({
            "end_year": int(z.year.iloc[-1]),
            "three_year_slope": lr.slope,
        })
    rolling = pd.DataFrame(roll_rows)
    save_table(rolling, "deep_rq2_15_rolling_record_slopes.csv")
    _deep_plot(
        rolling, "end_year", ["three_year_slope"],
        "deep_rq2_15_rolling_record_slope.png",
        "Rolling three-year record progression",
        "Slope (lb/year)",
        ["3-year slope"]
    )

    # --------------------------------------------------------
    # RQ2-16: Acceleration / second differences
    # --------------------------------------------------------

    accel = records[["year", "max"]].copy()
    accel["first_difference"] = accel["max"].diff()
    accel["second_difference"] = accel["first_difference"].diff()
    save_table(accel, "deep_rq2_16_record_acceleration.csv")

    _deep_plot(
        accel.dropna(subset=["second_difference"]),
        "year", ["second_difference"],
        "deep_rq2_16_record_acceleration.png",
        "Record-weight acceleration",
        "Second difference (lb/year²)",
        ["Second difference"]
    )

    # --------------------------------------------------------
    # RQ2-17: Annual volatility
    # --------------------------------------------------------

    vol_rows = []
    for y, z in d.groupby("year"):
        w = z.weight_lbs.dropna()
        vol_rows.append({
            "year": y,
            "sd": w.std(),
            "iqr": w.quantile(.75) - w.quantile(.25),
            "mad": np.median(np.abs(w - np.median(w))),
            "range": w.max() - w.min(),
        })
    volatility = pd.DataFrame(vol_rows).sort_values("year")
    save_table(volatility, "deep_rq2_17_weight_volatility.csv")
    _deep_plot(
        volatility, "year", ["sd", "iqr", "mad"],
        "deep_rq2_17_weight_volatility.png",
        "Annual weight dispersion",
        "Dispersion (lb)",
        ["Standard deviation", "IQR", "MAD"]
    )

    # --------------------------------------------------------
    # RQ2-18: Coefficient of variation
    # --------------------------------------------------------

    volatility["cv_pct"] = 100 * volatility["sd"] / quant["p50"].values
    save_table(volatility[["year", "cv_pct"]], "deep_rq2_18_weight_cv.csv")
    _deep_plot(
        volatility, "year", ["cv_pct"],
        "deep_rq2_18_weight_cv.png",
        "Relative weight variability",
        "Coefficient of variation (%)",
        ["CV"]
    )

    # --------------------------------------------------------
    # RQ2-19: Residual diagnostics
    # --------------------------------------------------------

    rec = records.copy()
    lr = stats.linregress(rec.year, rec["max"])
    rec["fitted"] = lr.intercept + lr.slope * rec.year
    rec["residual"] = rec["max"] - rec["fitted"]
    rec["abs_residual"] = rec.residual.abs()

    jb_stat, jb_p = safe_jarque_bera(rec.residual)
    dw = float(np.sum(np.diff(rec.residual) ** 2) / np.sum(rec.residual ** 2))
    resid_summary = pd.DataFrame([{
        "linear_slope": lr.slope,
        "r_squared": lr.rvalue ** 2,
        "jarque_bera": jb_stat,
        "jarque_bera_p": jb_p,
        "durbin_watson": dw,
        "residual_sd": rec.residual.std(),
        "mean_abs_residual": rec.abs_residual.mean(),
    }])
    save_table(resid_summary, "deep_rq2_19_record_residual_diagnostics.csv")
    save_table(rec, "deep_rq2_19_record_residuals.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axhline(0, linewidth=1)
    ax.plot(rec.year, rec.residual, marker="o", linewidth=2)
    ax.set_title("Residuals from linear record-weight trend", fontsize=15, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Residual (lb)")
    save_fig(fig, "deep_rq2_19_record_residuals.png")

    # --------------------------------------------------------
    # RQ2-20: Exhaustive candidate changepoints
    # --------------------------------------------------------

    cp_rows = []
    yy = records["max"].to_numpy(float)
    xx = records["year"].to_numpy(float)
    for i in range(2, len(xx) - 2):
        b = int(xx[i])
        left = stats.linregress(xx[:i], yy[:i])
        right = stats.linregress(xx[i:], yy[i:])
        pred_left = left.intercept + left.slope * xx[:i]
        pred_right = right.intercept + right.slope * xx[i:]
        rss = np.sum((yy[:i] - pred_left) ** 2) + np.sum((yy[i:] - pred_right) ** 2)
        cp_rows.append({
            "candidate_break_year": b,
            "left_slope": left.slope,
            "right_slope": right.slope,
            "slope_change": right.slope - left.slope,
            "rss": rss,
        })
    cps = pd.DataFrame(cp_rows)
    if len(cps):
        n = len(xx)
        cps["bic"] = n * np.log(cps["rss"] / n) + 4 * np.log(n)
        cps["aic"] = n * np.log(cps["rss"] / n) + 6
    save_table(cps, "deep_rq2_20_all_candidate_changepoints.csv")

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    if len(cps):
        ax.plot(cps.candidate_break_year, cps.slope_change, marker="o", label="Slope change")
        ax.axhline(0, linewidth=1)
        ax.set_title("Candidate changepoint slope changes", fontsize=15, fontweight="bold")
        ax.set_xlabel("Candidate breakpoint")
        ax.set_ylabel("Right slope minus left slope (lb/year)")
    save_fig(fig, "deep_rq2_20_changepoint_slope_changes.png")

    # --------------------------------------------------------
    # Compact statistical trend table for the 20 analyses
    # --------------------------------------------------------

    for metric, label in [
        ("mother_entropy", "RQ1-2 mother entropy"),
        ("pollinator_entropy", "RQ1-2 pollinator entropy"),
        ("mother_new_parent_fraction", "RQ1-3 mother novelty"),
        ("pollinator_new_parent_fraction", "RQ1-3 pollinator novelty"),
        ("mother_jaccard_previous", "RQ1-4 mother turnover"),
        ("pollinator_jaccard_previous", "RQ1-4 pollinator turnover"),
        ("p50", "RQ2-11 median weight"),
        ("p90", "RQ2-11 P90 weight"),
        ("p95", "RQ2-11 P95 weight"),
        ("p99", "RQ2-11 P99 weight"),
        ("max", "RQ2-11 record weight"),
        ("record_minus_p99", "RQ2-13 record-P99 gap"),
        ("record_p99_ratio", "RQ2-13 record/P99 ratio"),
        ("sd", "RQ2-17 standard deviation"),
        ("iqr", "RQ2-17 IQR"),
        ("cv_pct", "RQ2-18 coefficient of variation"),
    ]:
        source = deep_parent if metric in deep_parent.columns else (
            quant if metric in quant.columns else volatility
        )
        _deep_trend(out, source, metric, label)

    trend20 = pd.DataFrame(out)
    save_table(trend20, "deep_rq1_rq2_trend_tests.csv")

    return {
        "completeness": completeness,
        "parent": deep_parent,
        "persistence": persistence,
        "zipf": zipf,
        "completeness_weight": comp_weight,
        "bias": bias,
        "elite": elite,
        "reuse_performance": reuse_perf,
        "quantiles": quant,
        "record_breakers": records,
        "rolling": rolling,
        "acceleration": accel,
        "volatility": volatility,
        "changepoints": cps,
        "trend_tests": trend20,
    }


def main():

    ensure_dirs()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df, primary, separate = load_data()

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    print("\nRunning data-quality analysis...")

    quality = data_quality(
        primary,
        separate
    )

    # --------------------------------------------------------
    # ANNUAL WEIGHTS
    # --------------------------------------------------------

    print("Running annual weight analysis...")

    annual_weight = annual_weight_summary(
        primary,
        separate
    )

    # --------------------------------------------------------
    # RQ1 PARENT ANALYSIS
    # --------------------------------------------------------

    print("Running pedigree analysis...")

    mother = parent_metrics(
        primary,
        "seed_mother",
        "mother"
    )

    pollinator = parent_metrics(
        primary,
        "pollinator_father",
        "pollinator"
    )

    save_table(
        mother,
        "mother_annual_metrics.csv"
    )

    save_table(
        pollinator,
        "pollinator_annual_metrics.csv"
    )

    mother_period = parent_period_metrics(
        primary,
        "seed_mother",
        "mother"
    )

    pollinator_period = parent_period_metrics(
        primary,
        "pollinator_father",
        "pollinator"
    )

    save_table(
        mother_period,
        "mother_period_comparison.csv"
    )

    save_table(
        pollinator_period,
        "pollinator_period_comparison.csv"
    )

    mother_freq = parent_frequency_table(
        primary,
        "seed_mother"
    )

    pollinator_freq = parent_frequency_table(
        primary,
        "pollinator_father"
    )

    save_table(
        mother_freq,
        "mother_parent_frequency.csv"
    )

    save_table(
        pollinator_freq,
        "pollinator_parent_frequency.csv"
    )

    mother_jaccard, mother_sets = (
        parent_jaccard_matrix(
            primary,
            "seed_mother"
        )
    )

    poll_jaccard, poll_sets = (
        parent_jaccard_matrix(
            primary,
            "pollinator_father"
        )
    )

    mother_jaccard.to_csv(
        TABLE_DIR /
        "mother_parent_jaccard_matrix.csv"
    )

    poll_jaccard.to_csv(
        TABLE_DIR /
        "pollinator_parent_jaccard_matrix.csv"
    )

    # --------------------------------------------------------
    # GEOGRAPHY
    # --------------------------------------------------------

    print("Running geographic analysis...")

    geo = geography_expansion(
        primary
    )

    countries, country_shares = (
        country_metrics(
            primary
        )
    )

    country_change = (
        early_late_country_comparison(
            country_shares
        )
    )

    # --------------------------------------------------------
    # GROWERS
    # --------------------------------------------------------

    print("Running grower analysis...")

    growers = grower_metrics(
        primary
    )

    # --------------------------------------------------------
    # MASTER TABLE
    # --------------------------------------------------------

    annual = build_annual_master(
        annual_weight,
        mother,
        pollinator,
        geo,
        countries,
        growers
    )

    # --------------------------------------------------------
    # RECORD MODEL ANALYSIS
    # --------------------------------------------------------

    print("Running regression and changepoint analysis...")

    model_info = record_model_analysis(
        annual
    )

    # --------------------------------------------------------
    # TREND TESTS
    # --------------------------------------------------------

    trend_metrics = [
        "record_max",
        "mean_weight",
        "median_weight",
        "q90_weight",
        "q95_weight",
        "q99_weight",
        "record_improvement",
        "mean_improvement",

        "mother_unique_ratio",
        "mother_top1_share",
        "mother_top5_share",
        "mother_top10_share",
        "mother_simpson",
        "mother_effective",
        "mother_gini",
        "mother_new_parent_fraction",
        "mother_jaccard_previous",

        "pollinator_unique_ratio",
        "pollinator_top1_share",
        "pollinator_top5_share",
        "pollinator_top10_share",
        "pollinator_simpson",
        "pollinator_effective",
        "pollinator_gini",
        "pollinator_new_parent_fraction",
        "pollinator_jaccard_previous",

        "countries",
        "states_provinces",
        "cities",
        "gpc_sites",
        "unique_growers",

        "country_hhi",
        "effective_countries",
        "country_shannon",
        "effective_countries_shannon",
        "country_top1",
        "country_top3",
        "country_top5",
        "country_jaccard_previous",

        "repeat_grower_fraction",
        "new_grower_fraction",
        "mean_records_per_grower",
    ]

    trends = trend_analysis(
        annual,
        trend_metrics
    )

    trends = add_fdr(
        trends
    )

    save_table(
        trends,
        "trend_tests_with_fdr.csv"
    )

    # --------------------------------------------------------
    # EARLY / LATE
    # --------------------------------------------------------

    early_late = early_late_tests(
        annual
    )

    if len(early_late):

        # FDR for both early/late tests separately
        early_late["q_mann_whitney"] = (
            benjamini_hochberg(
                early_late[
                    "mann_whitney_p"
                ].to_numpy(float)
            )
        )

        early_late["q_permutation"] = (
            benjamini_hochberg(
                early_late[
                    "exact_permutation_p"
                ].to_numpy(float)
            )
        )

    save_table(
        early_late,
        "early_vs_late_statistical_tests.csv"
    )

    # --------------------------------------------------------
    # CROSS-DOMAIN
    # --------------------------------------------------------

    cross = cross_domain_correlations(
        annual
    )

    if len(cross):

        cross["q_pearson"] = (
            benjamini_hochberg(
                cross[
                    "pearson_p"
                ].to_numpy(float)
            )
        )

        cross["q_spearman"] = (
            benjamini_hochberg(
                cross[
                    "spearman_p"
                ].to_numpy(float)
            )
        )

    save_table(
        cross,
        "cross_domain_correlations.csv"
    )

    # --------------------------------------------------------
    # MEASUREMENT ANALYSIS
    # --------------------------------------------------------

    print("Running OTT / estimated-weight analysis...")

    measurement = measurement_analysis(
        primary
    )

    # --------------------------------------------------------
    # ADDITIONAL ANALYSES
    # --------------------------------------------------------

    print("Running competitive-depth and pedigree-pair analyses...")

    thresholds = weight_threshold_analysis(
        primary
    )

    threshold_tests = threshold_trend_tests(
        thresholds
    )

    distribution_shape = weight_distribution_shape(
        primary
    )

    grower_concentration = grower_concentration_analysis(
        primary
    )

    parent_pairs, parent_pair_frequency = (
        parent_pair_analysis(
            primary
        )
    )

    elite_pedigree = elite_pedigree_analysis(
        primary
    )

    # --------------------------------------------------------
    # DEEP RQ1 / RQ2 ANALYSIS PACK
    # --------------------------------------------------------

    deep_results = deep_rq1_rq2_analysis(
        primary,
        annual
    )

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------

    print("\nGenerating plots...")

    plot_record_trajectory(
        annual,
        model_info
    )

    plot_weight_distribution(
        annual
    )

    plot_record_improvement(
        annual
    )

    plot_model_comparison(
        annual,
        model_info
    )

    loo = pd.read_csv(
        TABLE_DIR /
        "leave_one_out_record_slopes.csv"
    )

    plot_loo(
        loo
    )

    plot_residuals(
        model_info
    )

    plot_parent_effective(
        annual
    )

    plot_parent_top10(
        annual
    )

    plot_parent_concentration(
        annual
    )

    plot_parent_jaccard(
        mother_jaccard
    )

    plot_parent_lorenz(
        primary,
        "seed_mother"
    )

    plot_recurring_parents(
        mother_freq
    )

    plot_geographic_expansion(
        geo
    )

    plot_country_concentration(
        countries
    )

    plot_top_country_shares(
        countries
    )

    plot_country_heatmap(
        country_shares
    )

    plot_country_change(
        country_change
    )

    plot_grower_ecosystem(
        growers
    )

    plot_record_vs_countries(
        annual
    )

    plot_record_vs_pedigree(
        annual
    )

    plot_estimated_vs_actual(
        primary
    )

    plot_ott(
        primary
    )

    plot_weight_thresholds(
        thresholds
    )

    plot_distribution_shape(
        distribution_shape
    )

    plot_grower_concentration(
        grower_concentration
    )

    plot_parent_pairs(
        parent_pairs
    )

    plot_elite_pedigree(
        elite_pedigree
    )

    plot_correlation_heatmap(
        annual
    )

    plot_period_distributions(
        primary
    )

    plot_cumulative_record(
        annual
    )

    plot_record_vs_mean(
        annual
    )

    plot_country_jaccard(
        countries
    )

    plot_new_parent_fraction(
        annual
    )

    plot_country_shannon(
        countries
    )

    # --------------------------------------------------------
    # 2026 SEPARATE TABLE
    # --------------------------------------------------------

    if len(separate):

        separate_summary = (
            annual[
                annual["year"] == 2026
            ]
        )

        save_table(
            separate_summary,
            "2026_separate_summary.csv"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    write_summary(
        annual,
        model_info,
        countries,
        geo,
        trends,
        early_late
    )

    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------

    print("Building figure gallery...")

    gallery = make_gallery()

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Tables:  {TABLE_DIR.resolve()}"
    )

    print(
        f"Figures: {FIG_DIR.resolve()}"
    )

    if gallery:
        print(
            f"Gallery: {gallery.resolve()}"
        )

    print()
    print(
        "Key outputs:"
    )

    print(
        "  annual_master.csv"
    )

    print(
        "  trend_tests_with_fdr.csv"
    )

    print(
        "  record_weight_models.csv"
    )

    print(
        "  residual_diagnostics.csv"
    )

    print(
        "  country_concentration_annual.csv"
    )

    print(
        "  mother_annual_metrics.csv"
    )

    print(
        "  pollinator_annual_metrics.csv"
    )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()
