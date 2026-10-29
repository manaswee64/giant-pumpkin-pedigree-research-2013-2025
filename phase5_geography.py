import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycountry
from scipy.stats import spearmanr, linregress

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 5: GEOGRAPHIC DISTRIBUTION ANALYSIS
# ============================================================

INPUT_FILE = "data.csv"
OUTPUT_DIR = "phase5_output"

START_YEAR = 2013
END_YEAR = 2025

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("PHASE 5: GEOGRAPHIC DISTRIBUTION ANALYSIS")
print("=" * 75)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} records.")

required_columns = [
    "Year",
    "Weight_lbs",
    "Country",
    "State_Province",
    "City",
    "GPC_Site"
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Numeric conversion
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Weight_lbs"] = pd.to_numeric(
    df["Weight_lbs"],
    errors="coerce"
)

# Restrict analysis period
df = df[
    (df["Year"] >= START_YEAR) &
    (df["Year"] <= END_YEAR)
].copy()

df["Year"] = df["Year"].astype(int)

print(
    f"Analysis period: {START_YEAR}-{END_YEAR}"
)
print(
    f"Records included: {len(df):,}"
)


# ============================================================
# 2. CLEAN GEOGRAPHIC FIELDS
# ============================================================

print("\nCleaning geographic fields...")


def clean_location(value):
    """
    Basic normalization only.
    Does not attempt to infer missing locations.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "":
        return np.nan

    # Common missing values
    missing = {
        "nan",
        "none",
        "null",
        "unknown",
        "unk",
        "?",
        "-",
        "--",
        "n/a",
        "na"
    }

    if value.lower() in missing:
        return np.nan

    # Collapse repeated whitespace
    value = re.sub(r"\s+", " ", value)

    return value


for column in [
    "Country",
    "State_Province",
    "City",
    "GPC_Site"
]:
    df[column] = df[column].apply(clean_location)


# ============================================================
# 2A. COUNTRY VALIDATION / NORMALIZATION
# ============================================================
# The Country column is used for all RQ3 country-level analyses.
# Do not infer a country from State_Province or City. Instead,
# normalize known country aliases and mark values that are not
# recognized as countries as missing. This prevents values such as
# "Ontario" (a Canadian province) from being counted as a country.

COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "united states of america": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "u.k": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "russia": "Russian Federation",
    "south korea": "Korea, Republic of",
    "republic of korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "czech republic": "Czechia",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "iran": "Iran, Islamic Republic of",
    "vietnam": "Viet Nam",
    "laos": "Lao People's Democratic Republic",
    "moldova": "Moldova, Republic of",
    "tanzania": "Tanzania, United Republic of",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "bolivia": "Bolivia, Plurinational State of",
    "brunei": "Brunei Darussalam",
    "syria": "Syrian Arab Republic",
    "taiwan": "Taiwan, Province of China",
}

# pycountry supplies the country-name reference rather than a hand-built
# list. Alpha-2/alpha-3 codes are also accepted because datasets often
# contain ISO codes.
COUNTRY_NAME_LOOKUP = {}
for country in pycountry.countries:
    COUNTRY_NAME_LOOKUP[country.name.casefold()] = country.name
    if getattr(country, "official_name", None):
        COUNTRY_NAME_LOOKUP[country.official_name.casefold()] = country.name
    if getattr(country, "common_name", None):
        COUNTRY_NAME_LOOKUP[country.common_name.casefold()] = country.name

COUNTRY_CODE_LOOKUP = {}
for country in pycountry.countries:
    COUNTRY_CODE_LOOKUP[country.alpha_2.casefold()] = country.name
    COUNTRY_CODE_LOOKUP[country.alpha_3.casefold()] = country.name

# Dataset-friendly canonical names. These are aliases only, not inferred
# geography.
COUNTRY_CANONICAL = {
    "United States of America": "United States",
    "United Kingdom": "United Kingdom",
    "Korea, Republic of": "South Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Russian Federation": "Russia",
    "Czechia": "Czech Republic",
    "Viet Nam": "Vietnam",
    "Iran, Islamic Republic of": "Iran",
    "Taiwan, Province of China": "Taiwan",
    "Moldova, Republic of": "Moldova",
    "Bolivia, Plurinational State of": "Bolivia",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Tanzania, United Republic of": "Tanzania",
    "Lao People's Democratic Republic": "Laos",
    "Brunei Darussalam": "Brunei",
    "Syrian Arab Republic": "Syria",
}

country_audit_rows = []

def clean_country(value):
    if pd.isna(value):
        return np.nan

    raw = str(value).strip()
    if raw == "":
        return np.nan

    key = re.sub(r"\s+", " ", raw).casefold()
    if key in {"nan", "none", "null", "unknown", "unk", "?", "-", "--", "n/a", "na"}:
        return np.nan

    # Explicit aliases first.
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]

    # Accept ISO alpha-2 / alpha-3 country codes.
    if key in COUNTRY_CODE_LOOKUP:
        return COUNTRY_CODE_LOOKUP[key]

    # Match official/common country names.
    if key in COUNTRY_NAME_LOOKUP:
        return COUNTRY_CANONICAL.get(COUNTRY_NAME_LOOKUP[key], COUNTRY_NAME_LOOKUP[key])

    # Unrecognized values are deliberately not reassigned to a country.
    return np.nan

raw_country = df["Country"].copy()
df["Country"] = df["Country"].apply(clean_country)

# Create an audit showing every non-empty raw value and its treatment.
for raw, cleaned, records in (
    pd.DataFrame({
        "raw_country": raw_country,
        "clean_country": df["Country"]
    })
    .dropna(subset=["raw_country"])
    .groupby(
        ["raw_country", "clean_country"],
        dropna=False
    )
    .size()
    .reset_index(name="records")
    .sort_values("records", ascending=False)
).itertuples(index=False):

    country_audit_rows.append({
        "raw_country": raw,
        "clean_country": cleaned,
        "records": records,
        "status": (
            "accepted"
            if pd.notna(cleaned)
            else "unrecognized_removed"
        )
    })

country_audit = pd.DataFrame(country_audit_rows)
country_audit.to_csv(
    f"{OUTPUT_DIR}/country_cleaning_audit.csv",
    index=False
)

removed_country_records = int(country_audit.loc[
    country_audit["status"] == "unrecognized_removed", "records"
].sum()) if not country_audit.empty else 0

print(f"\nCountry validation: {df['Country'].notna().sum():,} records have recognized countries.")
print(f"Unrecognized/non-country Country values excluded: {removed_country_records:,} records.")
print(f"Country audit: {OUTPUT_DIR}/country_cleaning_audit.csv")


# ============================================================
# 3. DATA COMPLETENESS
# ============================================================

print("\n" + "=" * 75)
print("PART 1: GEOGRAPHIC DATA COMPLETENESS")
print("=" * 75)

completeness_rows = []

for year, group in df.groupby("Year"):

    n = len(group)

    row = {
        "year": year,
        "records": n,

        "country_recorded": group["Country"].notna().sum(),
        "state_recorded": group["State_Province"].notna().sum(),
        "city_recorded": group["City"].notna().sum(),
        "gpc_site_recorded": group["GPC_Site"].notna().sum(),
    }

    row["country_pct"] = (
        row["country_recorded"] / n * 100
    )

    row["state_pct"] = (
        row["state_recorded"] / n * 100
    )

    row["city_pct"] = (
        row["city_recorded"] / n * 100
    )

    row["gpc_site_pct"] = (
        row["gpc_site_recorded"] / n * 100
    )

    completeness_rows.append(row)

completeness = pd.DataFrame(completeness_rows)

print(
    completeness.to_string(index=False)
)

completeness.to_csv(
    f"{OUTPUT_DIR}/geographic_completeness.csv",
    index=False
)


# ============================================================
# 4. ANNUAL GEOGRAPHIC DIVERSITY
# ============================================================

print("\n" + "=" * 75)
print("PART 2: ANNUAL GEOGRAPHIC DIVERSITY")
print("=" * 75)

geo_rows = []

for year, group in df.groupby("Year"):

    geo_rows.append({
        "year": year,

        "countries":
            group["Country"].dropna().nunique(),

        "states_provinces":
            group["State_Province"].dropna().nunique(),

        "cities":
            group["City"].dropna().nunique(),

        "gpc_sites":
            group["GPC_Site"].dropna().nunique()
    })

geo_stats = pd.DataFrame(geo_rows)

print(
    geo_stats.to_string(index=False)
)

geo_stats.to_csv(
    f"{OUTPUT_DIR}/annual_geographic_diversity.csv",
    index=False
)


# ============================================================
# 5. COUNTRY DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("PART 3: COUNTRY DISTRIBUTION")
print("=" * 75)

country_counts = (
    df.dropna(subset=["Country"])
      .groupby(["Year", "Country"])
      .size()
      .reset_index(name="records")
)

country_counts["share"] = (
    country_counts
    .groupby("Year")["records"]
    .transform(lambda x: x / x.sum())
)

country_counts["share_pct"] = (
    country_counts["share"] * 100
)

country_counts.to_csv(
    f"{OUTPUT_DIR}/country_distribution.csv",
    index=False
)

print("\nTop countries overall:")

overall_country = (
    df["Country"]
    .dropna()
    .value_counts()
    .head(20)
)

print(overall_country.to_string())


# ============================================================
# 6. TOP COUNTRY EACH YEAR
# ============================================================

print("\n" + "=" * 75)
print("PART 4: DOMINANT COUNTRY BY YEAR")
print("=" * 75)

top_country_rows = []

for year in sorted(df["Year"].unique()):

    counts = (
        df.loc[df["Year"] == year, "Country"]
        .dropna()
        .value_counts()
    )

    if len(counts) == 0:
        continue

    top_country = counts.index[0]
    top_count = counts.iloc[0]

    total = counts.sum()

    top_country_rows.append({
        "year": year,
        "top_country": top_country,
        "records": top_count,
        "share_pct": top_count / total * 100
    })

top_country_yearly = pd.DataFrame(top_country_rows)

print(
    top_country_yearly.to_string(index=False)
)

top_country_yearly.to_csv(
    f"{OUTPUT_DIR}/top_country_by_year.csv",
    index=False
)


# ============================================================
# 7. COUNTRY CONCENTRATION
# ============================================================

print("\n" + "=" * 75)
print("PART 5: COUNTRY CONCENTRATION")
print("=" * 75)


def hhi(values):
    """
    Herfindahl-Hirschman Index.
    Uses shares from 0 to 1.
    """
    values = np.asarray(values, dtype=float)

    if values.sum() == 0:
        return np.nan

    shares = values / values.sum()

    return np.sum(shares ** 2)


country_concentration_rows = []

for year, group in df.groupby("Year"):

    counts = (
        group["Country"]
        .dropna()
        .value_counts()
    )

    if len(counts) == 0:
        continue

    shares = counts / counts.sum()

    country_concentration_rows.append({
        "year": year,

        "country_hhi": hhi(counts),

        "effective_countries":
            1 / hhi(counts),

        "top_country_share_pct":
            shares.iloc[0] * 100,

        "top_3_country_share_pct":
            shares.head(3).sum() * 100,

        "top_5_country_share_pct":
            shares.head(5).sum() * 100,

        "countries":
            len(counts)
    })

country_concentration = pd.DataFrame(
    country_concentration_rows
)

print(
    country_concentration.to_string(index=False)
)

country_concentration.to_csv(
    f"{OUTPUT_DIR}/country_concentration.csv",
    index=False
)


# ============================================================
# 8. STATE/PROVINCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("PART 6: STATE/PROVINCE DISTRIBUTION")
print("=" * 75)

state_counts = (
    df.dropna(subset=["State_Province"])
      .groupby(["Year", "State_Province"])
      .size()
      .reset_index(name="records")
)

state_counts["share"] = (
    state_counts
    .groupby("Year")["records"]
    .transform(lambda x: x / x.sum())
)

state_counts["share_pct"] = (
    state_counts["share"] * 100
)

state_counts.to_csv(
    f"{OUTPUT_DIR}/state_province_distribution.csv",
    index=False
)

print("\nTop states/provinces overall:")

overall_states = (
    df["State_Province"]
    .dropna()
    .value_counts()
    .head(30)
)

print(overall_states.to_string())


# ============================================================
# 9. GPC SITE DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("PART 7: GPC SITE DISTRIBUTION")
print("=" * 75)

site_counts = (
    df.dropna(subset=["GPC_Site"])
      .groupby(["Year", "GPC_Site"])
      .size()
      .reset_index(name="records")
)

site_counts["share"] = (
    site_counts
    .groupby("Year")["records"]
    .transform(lambda x: x / x.sum())
)

site_counts["share_pct"] = (
    site_counts["share"] * 100
)

site_counts.to_csv(
    f"{OUTPUT_DIR}/gpc_site_distribution.csv",
    index=False
)

print("\nTop GPC sites overall:")

overall_sites = (
    df["GPC_Site"]
    .dropna()
    .value_counts()
    .head(30)
)

print(overall_sites.to_string())


# ============================================================
# 10. GEOGRAPHIC CONCENTRATION BY YEAR
# ============================================================

print("\n" + "=" * 75)
print("PART 8: GEOGRAPHIC CONCENTRATION")
print("=" * 75)

concentration_rows = []

for year, group in df.groupby("Year"):

    row = {"year": year}

    for field, prefix in [
        ("Country", "country"),
        ("State_Province", "state"),
        ("GPC_Site", "site")
    ]:

        values = (
            group[field]
            .dropna()
            .value_counts()
        )

        if len(values) == 0:
            row[f"{prefix}_hhi"] = np.nan
            row[f"{prefix}_effective"] = np.nan
            row[f"{prefix}_top1_pct"] = np.nan
            row[f"{prefix}_top5_pct"] = np.nan

        else:

            shares = values / values.sum()

            row[f"{prefix}_hhi"] = hhi(values)

            row[f"{prefix}_effective"] = (
                1 / hhi(values)
            )

            row[f"{prefix}_top1_pct"] = (
                shares.head(1).sum() * 100
            )

            row[f"{prefix}_top5_pct"] = (
                shares.head(5).sum() * 100
            )

    concentration_rows.append(row)

concentration = pd.DataFrame(
    concentration_rows
)

print(
    concentration.to_string(index=False)
)

concentration.to_csv(
    f"{OUTPUT_DIR}/geographic_concentration.csv",
    index=False
)


# ============================================================
# 11. TREND TESTS
# ============================================================

print("\n" + "=" * 75)
print("PART 9: TEMPORAL TRENDS")
print("=" * 75)


def trend_test(years, values):

    mask = (
        pd.notna(years) &
        pd.notna(values)
    )

    years = np.asarray(years)[mask]
    values = np.asarray(values)[mask]

    if len(years) < 3:
        return {
            "slope": np.nan,
            "r_squared": np.nan,
            "p_linear": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan
        }

    linear = linregress(years, values)

    rho, p = spearmanr(
        years,
        values
    )

    return {
        "slope": linear.slope,
        "r_squared": linear.rvalue ** 2,
        "p_linear": linear.pvalue,
        "spearman_rho": rho,
        "spearman_p": p
    }


trend_rows = []

trend_variables = [
    ("countries", "countries"),
    ("states_provinces", "states_provinces"),
    ("cities", "cities"),
    ("gpc_sites", "gpc_sites")
]

for column, label in trend_variables:

    result = trend_test(
        geo_stats["year"],
        geo_stats[column]
    )

    trend_rows.append({
        "metric": label,
        **result
    })


# Country concentration trends
for column, label in [
    ("country_hhi", "country_hhi"),
    ("effective_countries", "effective_countries"),
    ("top_country_share_pct", "top_country_share_pct"),
    ("top_3_country_share_pct", "top_3_country_share_pct"),
    ("top_5_country_share_pct", "top_5_country_share_pct")
]:

    result = trend_test(
        country_concentration["year"],
        country_concentration[column]
    )

    trend_rows.append({
        "metric": label,
        **result
    })


trend_results = pd.DataFrame(trend_rows)

print(
    trend_results.to_string(index=False)
)

trend_results.to_csv(
    f"{OUTPUT_DIR}/geographic_trends.csv",
    index=False
)


# ============================================================
# 12. COUNTRY SHARES: EARLY VS LATE PERIOD
# ============================================================

print("\n" + "=" * 75)
print("PART 10: EARLY VS LATE GEOGRAPHIC DISTRIBUTION")
print("=" * 75)

early_years = list(range(2013, 2017))
late_years = list(range(2022, 2026))


def period_country_distribution(years):

    subset = df[df["Year"].isin(years)]

    counts = (
        subset["Country"]
        .dropna()
        .value_counts()
    )

    shares = (
        counts / counts.sum() * 100
    )

    return shares


early_country = period_country_distribution(
    early_years
)

late_country = period_country_distribution(
    late_years
)

countries = sorted(
    set(early_country.index) |
    set(late_country.index)
)

comparison_rows = []

for country in countries:

    early = early_country.get(
        country,
        0
    )

    late = late_country.get(
        country,
        0
    )

    comparison_rows.append({
        "country": country,
        "early_share_pct": early,
        "late_share_pct": late,
        "change_percentage_points": late - early
    })

period_country_comparison = (
    pd.DataFrame(comparison_rows)
    .sort_values(
        "change_percentage_points",
        ascending=False
    )
)

print(
    "\nLargest increases in share:"
)

print(
    period_country_comparison
    .head(15)
    .to_string(index=False)
)

print(
    "\nLargest decreases in share:"
)

print(
    period_country_comparison
    .tail(15)
    .sort_values(
        "change_percentage_points"
    )
    .to_string(index=False)
)

period_country_comparison.to_csv(
    f"{OUTPUT_DIR}/early_vs_late_country_share.csv",
    index=False
)


# ============================================================
# 13. TOP COUNTRIES OVER TIME
# ============================================================

print("\n" + "=" * 75)
print("PART 11: TOP COUNTRIES OVER TIME")
print("=" * 75)

# Find the 8 countries with the largest total representation
top_countries = (
    df["Country"]
    .dropna()
    .value_counts()
    .head(8)
    .index
    .tolist()
)

country_year_matrix = (
    country_counts[
        country_counts["Country"].isin(top_countries)
    ]
    .pivot(
        index="Year",
        columns="Country",
        values="share_pct"
    )
    .fillna(0)
)

country_year_matrix.to_csv(
    f"{OUTPUT_DIR}/top_country_share_over_time.csv"
)


# ============================================================
# 14. FIGURE: NUMBER OF COUNTRIES / STATES / SITES
# ============================================================

print("\nGenerating figures...")

plt.figure(figsize=(12, 7))

plt.plot(
    geo_stats["year"],
    geo_stats["countries"],
    marker="o",
    label="Countries"
)

plt.plot(
    geo_stats["year"],
    geo_stats["states_provinces"],
    marker="o",
    label="States / Provinces"
)

plt.plot(
    geo_stats["year"],
    geo_stats["gpc_sites"],
    marker="o",
    label="GPC Sites"
)

plt.xlabel("Year")
plt.ylabel("Number of unique locations")
plt.title(
    "Geographic Expansion of Competitive Giant Pumpkin Records"
)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/geographic_expansion.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. FIGURE: COUNTRY CONCENTRATION
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    country_concentration["year"],
    country_concentration["top_country_share_pct"],
    marker="o",
    label="Top country"
)

plt.plot(
    country_concentration["year"],
    country_concentration["top_3_country_share_pct"],
    marker="o",
    label="Top 3 countries"
)

plt.plot(
    country_concentration["year"],
    country_concentration["top_5_country_share_pct"],
    marker="o",
    label="Top 5 countries"
)

plt.xlabel("Year")
plt.ylabel("Share of records (%)")

plt.title(
    "Country-Level Concentration of Competitive Giant Pumpkin Records"
)

plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/country_concentration.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. FIGURE: EFFECTIVE NUMBER OF COUNTRIES
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    country_concentration["year"],
    country_concentration["effective_countries"],
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Effective number of countries")

plt.title(
    "Effective Geographic Diversity of Competitive Giant Pumpkin Records"
)

plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/effective_country_diversity.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. FIGURE: TOP COUNTRIES SHARE OVER TIME
# ============================================================

plt.figure(figsize=(12, 7))

for country in top_countries:

    if country in country_year_matrix.columns:

        plt.plot(
            country_year_matrix.index,
            country_year_matrix[country],
            marker="o",
            label=country
        )

plt.xlabel("Year")
plt.ylabel("Share of records (%)")

plt.title(
    "Changing Country Composition of Competitive Giant Pumpkin Records"
)

plt.grid(alpha=0.25)
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/country_composition_over_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()




# ============================================================
# 17A. FIGURE: COUNTRY DATA VALIDATION
# ============================================================

plt.figure(figsize=(12, 7))

country_quality = completeness.copy()
plt.plot(
    country_quality["year"],
    country_quality["country_pct"],
    marker="o",
    label="Recognized country recorded"
)

plt.xlabel("Year")
plt.ylabel("Records with recognized country (%)")
plt.title("Country Data Completeness After Validation")
plt.ylim(0, 105)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/country_data_validation.png",
    dpi=300
)
plt.close()


# ============================================================
# 18. SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("PHASE 5 COMPLETE")
print("=" * 75)

print("\nKEY OUTPUTS")
print("-" * 75)

print(
    f"Countries in {START_YEAR}: "
    f"{geo_stats.iloc[0]['countries']}"
)

print(
    f"Countries in {END_YEAR}: "
    f"{geo_stats.iloc[-1]['countries']}"
)

country_trend = trend_results[
    trend_results["metric"] == "countries"
].iloc[0]

print(
    f"\nCountry-count trend:"
)

print(
    f"  slope = {country_trend['slope']:.4f} "
    f"countries/year"
)

print(
    f"  p = {country_trend['p_linear']:.6f}"
)

print(
    "\nFiles written to:"
)

print(
    f"  {OUTPUT_DIR}/"
)

print("\nMain files:")

files = [
    "geographic_completeness.csv",
    "country_cleaning_audit.csv",
    "annual_geographic_diversity.csv",
    "country_distribution.csv",
    "country_concentration.csv",
    "state_province_distribution.csv",
    "gpc_site_distribution.csv",
    "geographic_concentration.csv",
    "geographic_trends.csv",
    "early_vs_late_country_share.csv",
    "top_country_share_over_time.csv",
    "geographic_expansion.png",
    "country_concentration.png",
    "effective_country_diversity.png",
    "country_composition_over_time.png",
    "country_data_validation.png"
]

for f in files:
    print(f"  {f}")

print("\nIMPORTANT:")
print(
    "This analysis measures geographic representation and "
    "concentration in the recorded dataset."
)
print(
    "It does not establish climate causation or geographic "
    "center of gravity without coordinate data."
)
print(
    "Changes may also reflect participation, reporting, or "
    "database coverage rather than actual cultivation shifts."
)

print("\nDone.")
