import pandas as pd
import numpy as np
from scipy.stats import linregress, spearmanr


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    try:
        df = pd.read_csv("data.csv")

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Convert important columns
        df["Year"] = pd.to_numeric(
            df["Year"],
            errors="coerce"
        )

        df["Weight_lbs"] = pd.to_numeric(
            df["Weight_lbs"],
            errors="coerce"
        )

        # Remove rows without a year
        df = df.dropna(subset=["Year"])

        df["Year"] = df["Year"].astype(int)

        print(f"Loaded {len(df):,} pumpkin records.")

        return df

    except FileNotFoundError:

        print("ERROR: data.csv was not found.")
        return None

    except Exception as e:

        print(f"ERROR loading data: {e}")
        return None


# ============================================================
# PARENT CLASSIFICATION
# ============================================================

def classify_parent(value):

    if pd.isna(value):
        return "missing", None

    value = str(value).strip()

    if value == "":
        return "missing", None

    lower = value.casefold()

    if lower == "self":
        return "self", None

    if lower == "open":
        return "open", None

    if lower in {
        "unknown",
        "unk",
        "?",
        "n/a",
        "na",
        "none"
    }:
        return "unknown", None

    return "named", value


# ============================================================
# CLEAN PARENT NAME
# ============================================================

def clean_parent_name(value):

    if value is None:
        return None

    value = str(value).strip()

    # Collapse repeated spaces
    value = " ".join(value.split())

    return value


# ============================================================
# GET NAMED PARENTS
# ============================================================

def get_named_parents(series):

    named_parents = []

    for value in series:

        category, parent = classify_parent(value)

        if category == "named":

            parent = clean_parent_name(parent)

            named_parents.append(parent)

    return named_parents


# ============================================================
# CALCULATE YEARLY PEDIGREE METRICS
# ============================================================

def calculate_yearly_metrics(df):

    results = []

    # IMPORTANT:
    # 2026 is excluded from the primary analysis because
    # it is an incomplete/current year.

    analysis_df = df[
        (df["Year"] >= 2013) &
        (df["Year"] <= 2025)
    ].copy()

    for year, group in analysis_df.groupby("Year"):

        row = {
            "Year": int(year)
        }

        # ----------------------------------------------------
        # MOTHER
        # ----------------------------------------------------

        mothers = get_named_parents(
            group["Seed_Mother"]
        )

        if len(mothers) > 0:

            mother_counts = (
                pd.Series(mothers)
                .value_counts()
            )

            mother_total = mother_counts.sum()
            mother_unique = len(mother_counts)

            # Unique-parent ratio
            mother_unique_ratio = (
                mother_unique /
                mother_total
            )

            # Top 10 share
            mother_top10 = (
                mother_counts.head(10).sum()
                / mother_total
            )

            # Simpson concentration
            mother_proportions = (
                mother_counts /
                mother_total
            )

            mother_simpson = np.sum(
                mother_proportions ** 2
            )

            # Effective number
            mother_effective = (
                1 / mother_simpson
            )

        else:

            mother_total = 0
            mother_unique = np.nan
            mother_unique_ratio = np.nan
            mother_top10 = np.nan
            mother_simpson = np.nan
            mother_effective = np.nan

        # ----------------------------------------------------
        # POLLINATOR
        # ----------------------------------------------------

        pollinators = get_named_parents(
            group["Pollinator_Father"]
        )

        if len(pollinators) > 0:

            pollinator_counts = (
                pd.Series(pollinators)
                .value_counts()
            )

            pollinator_total = (
                pollinator_counts.sum()
            )

            pollinator_unique = (
                len(pollinator_counts)
            )

            pollinator_unique_ratio = (
                pollinator_unique /
                pollinator_total
            )

            pollinator_top10 = (
                pollinator_counts.head(10).sum()
                / pollinator_total
            )

            pollinator_proportions = (
                pollinator_counts /
                pollinator_total
            )

            pollinator_simpson = np.sum(
                pollinator_proportions ** 2
            )

            pollinator_effective = (
                1 / pollinator_simpson
            )

        else:

            pollinator_total = 0
            pollinator_unique = np.nan
            pollinator_unique_ratio = np.nan
            pollinator_top10 = np.nan
            pollinator_simpson = np.nan
            pollinator_effective = np.nan

        # ----------------------------------------------------
        # SAVE ROW
        # ----------------------------------------------------

        row.update({

            "Mother_Named": mother_total,
            "Mother_Unique": mother_unique,
            "Mother_Unique_Ratio": mother_unique_ratio,
            "Mother_Top10_Share": mother_top10,
            "Mother_Simpson": mother_simpson,
            "Mother_Effective_Number": mother_effective,

            "Pollinator_Named": pollinator_total,
            "Pollinator_Unique": pollinator_unique,
            "Pollinator_Unique_Ratio":
                pollinator_unique_ratio,
            "Pollinator_Top10_Share":
                pollinator_top10,
            "Pollinator_Simpson":
                pollinator_simpson,
            "Pollinator_Effective_Number":
                pollinator_effective
        })

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# PERIOD ASSIGNMENT
# ============================================================

def assign_period(year):

    if 2013 <= year <= 2016:
        return "2013-2016"

    elif 2017 <= year <= 2020:
        return "2017-2020"

    elif 2021 <= year <= 2025:
        return "2021-2025"

    else:
        return None


# ============================================================
# PERIOD COMPARISON
# ============================================================

def calculate_period_comparison(yearly):

    df = yearly.copy()

    df["Period"] = df["Year"].apply(
        assign_period
    )

    df = df.dropna(subset=["Period"])

    metrics = [
        "Mother_Unique_Ratio",
        "Pollinator_Unique_Ratio",

        "Mother_Top10_Share",
        "Pollinator_Top10_Share",

        "Mother_Simpson",
        "Pollinator_Simpson",

        "Mother_Effective_Number",
        "Pollinator_Effective_Number"
    ]

    period_results = (
        df.groupby("Period")[metrics]
        .agg(["mean", "std"])
    )

    return period_results


# ============================================================
# LINEAR TREND TEST
# ============================================================

def calculate_linear_trends(yearly):

    results = []

    metrics = [

        (
            "Mother_Unique_Ratio",
            "Mother unique-parent ratio"
        ),

        (
            "Pollinator_Unique_Ratio",
            "Pollinator unique-parent ratio"
        ),

        (
            "Mother_Top10_Share",
            "Mother Top-10 share"
        ),

        (
            "Pollinator_Top10_Share",
            "Pollinator Top-10 share"
        ),

        (
            "Mother_Simpson",
            "Mother Simpson concentration"
        ),

        (
            "Pollinator_Simpson",
            "Pollinator Simpson concentration"
        ),

        (
            "Mother_Effective_Number",
            "Mother effective number"
        ),

        (
            "Pollinator_Effective_Number",
            "Pollinator effective number"
        )
    ]

    for column, name in metrics:

        data = yearly[
            ["Year", column]
        ].dropna()

        x = data["Year"].values
        y = data[column].values

        if len(x) < 3:
            continue

        result = linregress(x, y)

        results.append({

            "Metric": name,

            "Slope_per_year":
                result.slope,

            "Intercept":
                result.intercept,

            "R_squared":
                result.rvalue ** 2,

            "P_value":
                result.pvalue,

            "Standard_Error":
                result.stderr,

            "Direction":
                (
                    "Increasing"
                    if result.slope > 0
                    else "Decreasing"
                )

        })

    return pd.DataFrame(results)


# ============================================================
# SPEARMAN TREND TEST
# ============================================================

def calculate_spearman_trends(yearly):

    results = []

    metrics = [

        (
            "Mother_Unique_Ratio",
            "Mother unique-parent ratio"
        ),

        (
            "Pollinator_Unique_Ratio",
            "Pollinator unique-parent ratio"
        ),

        (
            "Mother_Top10_Share",
            "Mother Top-10 share"
        ),

        (
            "Pollinator_Top10_Share",
            "Pollinator Top-10 share"
        ),

        (
            "Mother_Simpson",
            "Mother Simpson concentration"
        ),

        (
            "Pollinator_Simpson",
            "Pollinator Simpson concentration"
        ),

        (
            "Mother_Effective_Number",
            "Mother effective number"
        ),

        (
            "Pollinator_Effective_Number",
            "Pollinator effective number"
        )
    ]

    for column, name in metrics:

        data = yearly[
            ["Year", column]
        ].dropna()

        if len(data) < 3:
            continue

        rho, p_value = spearmanr(
            data["Year"],
            data[column]
        )

        results.append({

            "Metric": name,

            "Spearman_Rho":
                rho,

            "P_value":
                p_value,

            "Direction":
                (
                    "Increasing"
                    if rho > 0
                    else "Decreasing"
                )

        })

    return pd.DataFrame(results)


# ============================================================
# PRINT PERIOD COMPARISON NICELY
# ============================================================

def print_period_comparison(period_results):

    print("\n")
    print("=" * 80)
    print("PERIOD COMPARISON")
    print("=" * 80)

    for period in [
        "2013-2016",
        "2017-2020",
        "2021-2025"
    ]:

        print(f"\n{period}")
        print("-" * 50)

        row = period_results.loc[period]

        metrics = [
            ("Mother unique ratio",
             "Mother_Unique_Ratio"),

            ("Pollinator unique ratio",
             "Pollinator_Unique_Ratio"),

            ("Mother Top-10 share",
             "Mother_Top10_Share"),

            ("Pollinator Top-10 share",
             "Pollinator_Top10_Share"),

            ("Mother Simpson concentration",
             "Mother_Simpson"),

            ("Pollinator Simpson concentration",
             "Pollinator_Simpson"),

            ("Mother effective number",
             "Mother_Effective_Number"),

            ("Pollinator effective number",
             "Pollinator_Effective_Number")
        ]

        for label, column in metrics:

            mean = row[(column, "mean")]
            std = row[(column, "std")]

            print(
                f"{label:35s}: "
                f"{mean:.4f} ± {std:.4f}"
            )


# ============================================================
# PRINT LINEAR TRENDS
# ============================================================

def print_linear_trends(trends):

    print("\n")
    print("=" * 80)
    print("LINEAR TEMPORAL TRENDS: 2013-2025")
    print("=" * 80)

    for _, row in trends.iterrows():

        print(f"\n{row['Metric']}")

        print(
            f"  Slope/year: "
            f"{row['Slope_per_year']:.6f}"
        )

        print(
            f"  R²: "
            f"{row['R_squared']:.4f}"
        )

        print(
            f"  p-value: "
            f"{row['P_value']:.6f}"
        )

        print(
            f"  Direction: "
            f"{row['Direction']}"
        )


# ============================================================
# PRINT SPEARMAN RESULTS
# ============================================================

def print_spearman_results(results):

    print("\n")
    print("=" * 80)
    print("SPEARMAN TEMPORAL TRENDS: 2013-2025")
    print("=" * 80)

    for _, row in results.iterrows():

        print(f"\n{row['Metric']}")

        print(
            f"  Spearman rho: "
            f"{row['Spearman_Rho']:.4f}"
        )

        print(
            f"  p-value: "
            f"{row['P_value']:.6f}"
        )

        print(
            f"  Direction: "
            f"{row['Direction']}"
        )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def run_phase_3_trend_analysis(df):

    print("\n")
    print("=" * 80)
    print("RQ1: PEDIGREE DIVERSITY + TEMPORAL ANALYSIS")
    print("=" * 80)

    print(
        "\nPrimary analysis period: 2013-2025"
    )

    print(
        "2026 excluded because it is a partial/current year."
    )

    # --------------------------------------------------------
    # YEARLY METRICS
    # --------------------------------------------------------

    print("\nCalculating yearly pedigree metrics...")

    yearly = calculate_yearly_metrics(df)

    print("\nYEARLY METRICS")
    print("-" * 80)

    print(
        yearly.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # PERIOD COMPARISON
    # --------------------------------------------------------

    period_results = calculate_period_comparison(
        yearly
    )

    print_period_comparison(
        period_results
    )

    # --------------------------------------------------------
    # LINEAR TRENDS
    # --------------------------------------------------------

    linear_results = calculate_linear_trends(
        yearly
    )

    print_linear_trends(
        linear_results
    )

    # --------------------------------------------------------
    # SPEARMAN
    # --------------------------------------------------------

    spearman_results = calculate_spearman_trends(
        yearly
    )

    print_spearman_results(
        spearman_results
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    yearly.to_csv(
        "rq1_yearly_metrics.csv",
        index=False
    )

    period_results.to_csv(
        "rq1_period_comparison.csv"
    )

    linear_results.to_csv(
        "rq1_linear_trends.csv",
        index=False
    )

    spearman_results.to_csv(
        "rq1_spearman_trends.csv",
        index=False
    )

    print("\n")
    print("=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print("""
rq1_yearly_metrics.csv
rq1_period_comparison.csv
rq1_linear_trends.csv
rq1_spearman_trends.csv
""")

    return (
        yearly,
        period_results,
        linear_results,
        spearman_results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    while True:

        print("\n")
        print("=" * 60)
        print("GIANT PUMPKIN RQ1 ANALYSIS")
        print("=" * 60)

        print("1. Run complete temporal analysis")
        print("2. Show yearly metrics")
        print("3. Show period comparison")
        print("4. Show linear trend tests")
        print("5. Show Spearman trend tests")
        print("0. Exit")

        choice = input(
            "\nChoose an option: "
        ).strip()

        # ----------------------------------------------------
        # COMPLETE ANALYSIS
        # ----------------------------------------------------

        if choice == "1":

            results = run_phase_3_trend_analysis(
                df
            )

        # ----------------------------------------------------
        # YEARLY
        # ----------------------------------------------------

        elif choice == "2":

            yearly = calculate_yearly_metrics(
                df
            )

            print("\nYEARLY METRICS")
            print("-" * 80)

            print(
                yearly.to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}"
                )
            )

        # ----------------------------------------------------
        # PERIOD
        # ----------------------------------------------------

        elif choice == "3":

            yearly = calculate_yearly_metrics(
                df
            )

            period_results = calculate_period_comparison(
                yearly
            )

            print_period_comparison(
                period_results
            )

        # ----------------------------------------------------
        # LINEAR
        # ----------------------------------------------------

        elif choice == "4":

            yearly = calculate_yearly_metrics(
                df
            )

            results = calculate_linear_trends(
                yearly
            )

            print_linear_trends(
                results
            )

        # ----------------------------------------------------
        # SPEARMAN
        # ----------------------------------------------------

        elif choice == "5":

            yearly = calculate_yearly_metrics(
                df
            )

            results = calculate_spearman_trends(
                yearly
            )

            print_spearman_results(
                results
            )

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif choice == "0":

            print("\nGoodbye!")
            break

        else:

            print(
                "\nInvalid option. "
                "Please choose again."
            )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()