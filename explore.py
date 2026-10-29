import pandas as pd
import numpy as np


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
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Weight_lbs"] = pd.to_numeric(df["Weight_lbs"], errors="coerce")

        # Remove rows without a year
        df = df.dropna(subset=["Year"])

        # Convert year to integer
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
    """
    Classifies a pedigree entry into:

    named   = identifiable parent
    self    = self-pollination
    open    = open pollination
    unknown = explicitly unknown
    missing = blank / NaN

    Returns:
        (category, cleaned_parent_name)
    """

    if pd.isna(value):
        return "missing", None

    value = str(value).strip()

    if value == "":
        return "missing", None

    lower = value.casefold()

    # Self-pollination
    if lower == "self":
        return "self", None

    # Open pollination
    if lower == "open":
        return "open", None

    # Unknown parent
    if lower in {
        "unknown",
        "unk",
        "?",
        "n/a",
        "na",
        "none"
    }:
        return "unknown", None

    # Actual named parent
    return "named", value


# ============================================================
# CLEAN PARENT NAME
# ============================================================

def clean_parent_name(value):
    """
    Cleans harmless formatting differences in a named parent.

    Example:

        ' 2145.5 McMullen 2015  '
        ->
        '2145.5 McMullen 2015'
    """

    if value is None:
        return None

    value = str(value).strip()

    # Collapse repeated spaces
    value = " ".join(value.split())

    return value


# ============================================================
# PHASE 3A
# PEDIGREE DATA QUALITY BY YEAR
# ============================================================

def pedigree_data_quality_by_year(df):

    results = []

    for year, group in df.groupby("Year"):

        row = {
            "Year": int(year),
            "Pumpkins": len(group)
        }

        for column, prefix in [
            ("Seed_Mother", "Mother"),
            ("Pollinator_Father", "Pollinator")
        ]:

            classifications = group[column].apply(classify_parent)

            categories = classifications.apply(lambda x: x[0])

            named_count = (categories == "named").sum()
            self_count = (categories == "self").sum()
            open_count = (categories == "open").sum()
            unknown_count = (categories == "unknown").sum()
            missing_count = (categories == "missing").sum()

            row[f"{prefix}_Named"] = named_count
            row[f"{prefix}_Self"] = self_count
            row[f"{prefix}_Open"] = open_count
            row[f"{prefix}_Unknown"] = unknown_count
            row[f"{prefix}_Missing"] = missing_count

            # Percentage of all records
            total = len(group)

            row[f"{prefix}_Named_Pct"] = (
                named_count / total * 100
                if total > 0 else np.nan
            )

            row[f"{prefix}_Self_Pct"] = (
                self_count / total * 100
                if total > 0 else np.nan
            )

            row[f"{prefix}_Open_Pct"] = (
                open_count / total * 100
                if total > 0 else np.nan
            )

            row[f"{prefix}_Unknown_Pct"] = (
                unknown_count / total * 100
                if total > 0 else np.nan
            )

            row[f"{prefix}_Missing_Pct"] = (
                missing_count / total * 100
                if total > 0 else np.nan
            )

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# PHASE 3B
# UNIQUE NAMED PARENTS BY YEAR
# ============================================================

def parent_diversity_by_year(df):

    results = []

    for year, group in df.groupby("Year"):

        row = {
            "Year": int(year),
            "Pumpkins": len(group)
        }

        for column, prefix in [
            ("Seed_Mother", "Mother"),
            ("Pollinator_Father", "Pollinator")
        ]:

            named_parents = []

            for value in group[column]:

                category, parent = classify_parent(value)

                if category == "named":
                    parent = clean_parent_name(parent)
                    named_parents.append(parent)

            n_named = len(named_parents)
            n_unique = len(set(named_parents))

            if n_named > 0:
                unique_ratio = n_unique / n_named
            else:
                unique_ratio = np.nan

            row[f"{prefix}_Named"] = n_named
            row[f"{prefix}_Unique"] = n_unique
            row[f"{prefix}_Unique_Ratio"] = unique_ratio

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# PHASE 3C
# PARENT REUSE FREQUENCY
# ============================================================

def parent_reuse_by_year(df):

    results = []

    for year, group in df.groupby("Year"):

        row = {
            "Year": int(year)
        }

        for column, prefix in [
            ("Seed_Mother", "Mother"),
            ("Pollinator_Father", "Pollinator")
        ]:

            named_parents = []

            for value in group[column]:

                category, parent = classify_parent(value)

                if category == "named":
                    parent = clean_parent_name(parent)
                    named_parents.append(parent)

            if len(named_parents) == 0:
                row[f"{prefix}_Top1_Share"] = np.nan
                row[f"{prefix}_Top5_Share"] = np.nan
                row[f"{prefix}_Top10_Share"] = np.nan
                continue

            counts = pd.Series(named_parents).value_counts()

            total = counts.sum()

            top1 = counts.head(1).sum()
            top5 = counts.head(5).sum()
            top10 = counts.head(10).sum()

            row[f"{prefix}_Top1_Share"] = top1 / total
            row[f"{prefix}_Top5_Share"] = top5 / total
            row[f"{prefix}_Top10_Share"] = top10 / total

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# PHASE 3D
# SIMPSON CONCENTRATION + EFFECTIVE NUMBER
# ============================================================

def parent_concentration_by_year(df):

    results = []

    for year, group in df.groupby("Year"):

        row = {
            "Year": int(year)
        }

        for column, prefix in [
            ("Seed_Mother", "Mother"),
            ("Pollinator_Father", "Pollinator")
        ]:

            named_parents = []

            for value in group[column]:

                category, parent = classify_parent(value)

                if category == "named":
                    parent = clean_parent_name(parent)
                    named_parents.append(parent)

            if len(named_parents) == 0:

                row[f"{prefix}_Simpson"] = np.nan
                row[f"{prefix}_Effective_Number"] = np.nan

                continue

            counts = pd.Series(named_parents).value_counts()

            proportions = counts / counts.sum()

            # Simpson concentration
            simpson = np.sum(proportions ** 2)

            # Effective number of parents
            effective_number = 1 / simpson

            row[f"{prefix}_Simpson"] = simpson
            row[f"{prefix}_Effective_Number"] = effective_number

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# RUN ALL PHASE 3 ANALYSES
# ============================================================

def run_phase_3(df):

    print("\n" + "=" * 70)
    print("PHASE 3: PEDIGREE DIVERSITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. DATA QUALITY
    # --------------------------------------------------------

    print("\n[1] PEDIGREE DATA QUALITY BY YEAR")
    print("-" * 70)

    quality = pedigree_data_quality_by_year(df)

    print(quality.to_string(index=False))

    # --------------------------------------------------------
    # 2. UNIQUE PARENTS
    # --------------------------------------------------------

    print("\n[2] UNIQUE NAMED PARENTS BY YEAR")
    print("-" * 70)

    diversity = parent_diversity_by_year(df)

    print(diversity.to_string(index=False))

    # --------------------------------------------------------
    # 3. PARENT REUSE
    # --------------------------------------------------------

    print("\n[3] PARENT REUSE BY YEAR")
    print("-" * 70)

    reuse = parent_reuse_by_year(df)

    print(reuse.to_string(index=False))

    # --------------------------------------------------------
    # 4. CONCENTRATION
    # --------------------------------------------------------

    print("\n[4] PEDIGREE CONCENTRATION BY YEAR")
    print("-" * 70)

    concentration = parent_concentration_by_year(df)

    print(concentration.to_string(index=False))

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    quality.to_csv("rq1_pedigree_quality.csv", index=False)
    diversity.to_csv("rq1_pedigree_diversity.csv", index=False)
    reuse.to_csv("rq1_parent_reuse.csv", index=False)
    concentration.to_csv("rq1_parent_concentration.csv", index=False)

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print("""
rq1_pedigree_quality.csv
rq1_pedigree_diversity.csv
rq1_parent_reuse.csv
rq1_parent_concentration.csv
""")

    return quality, diversity, reuse, concentration


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    df = load_data()

    if df is None:
        return

    while True:

        print("\n" + "=" * 50)
        print("GIANT PUMPKIN ANALYSIS")
        print("=" * 50)

        print("1. Number of pumpkins")
        print("2. Number of years")
        print("3. Pumpkins per year")
        print("4. Average weight per year")
        print("5. Maximum weight per year")
        print("6. Number of unique growers")
        print("7. Number of countries")
        print("8. Number of GPC sites")
        print("9. Mother information percentage")
        print("10. Pollinator information percentage")
        print("11. Both parents percentage")
        print("12. Run all basic statistics")
        print()
        print("13. Pedigree data quality by year")
        print("14. Parent diversity by year")
        print("15. Parent reuse by year")
        print("16. Parent concentration by year")
        print("17. Run complete Phase 3")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        # ----------------------------------------------------
        # BASIC STATISTICS
        # ----------------------------------------------------

        if choice == "1":

            print(f"\nNumber of pumpkins: {len(df):,}")

        elif choice == "2":

            print(f"\nNumber of years: {df['Year'].nunique()}")

        elif choice == "3":

            print("\nPumpkins per year:")
            print(df["Year"].value_counts().sort_index())

        elif choice == "4":

            print("\nAverage weight per year:")
            print(
                df.groupby("Year")["Weight_lbs"]
                .mean()
                .round(2)
            )

        elif choice == "5":

            print("\nMaximum weight per year:")
            print(
                df.groupby("Year")["Weight_lbs"]
                .max()
                .round(2)
            )

        elif choice == "6":

            print(
                f"\nUnique growers: "
                f"{df['Grower_Name'].nunique():,}"
            )

        elif choice == "7":

            print(
                f"\nCountries: "
                f"{df['Country'].nunique():,}"
            )

        elif choice == "8":

            print(
                f"\nGPC sites: "
                f"{df['GPC_Site'].nunique():,}"
            )

        elif choice == "9":

            count = 0

            for value in df["Seed_Mother"]:

                category, _ = classify_parent(value)

                if category == "named":
                    count += 1

            percentage = count / len(df) * 100

            print(
                f"\nMother information: "
                f"{percentage:.2f}%"
            )

        elif choice == "10":

            count = 0

            for value in df["Pollinator_Father"]:

                category, _ = classify_parent(value)

                if category == "named":
                    count += 1

            percentage = count / len(df) * 100

            print(
                f"\nPollinator information: "
                f"{percentage:.2f}%"
            )

        elif choice == "11":

            both = 0

            for _, row in df.iterrows():

                mother_category, _ = classify_parent(
                    row["Seed_Mother"]
                )

                father_category, _ = classify_parent(
                    row["Pollinator_Father"]
                )

                if (
                    mother_category == "named"
                    and father_category == "named"
                ):
                    both += 1

            percentage = both / len(df) * 100

            print(
                f"\nBoth parents recorded: "
                f"{percentage:.2f}%"
            )

        elif choice == "12":

            print("\nRunning basic statistics...")

            print(f"\nNumber of pumpkins: {len(df):,}")
            print(f"Number of years: {df['Year'].nunique():,}")
            print(f"Unique growers: {df['Grower_Name'].nunique():,}")
            print(f"Countries: {df['Country'].nunique():,}")
            print(f"GPC sites: {df['GPC_Site'].nunique():,}")

            print("\nPumpkins per year:")
            print(df["Year"].value_counts().sort_index())

            print("\nAverage weight:")
            print(
                df.groupby("Year")["Weight_lbs"]
                .mean()
                .round(2)
            )

            print("\nMaximum weight:")
            print(
                df.groupby("Year")["Weight_lbs"]
                .max()
                .round(2)
            )

        # ----------------------------------------------------
        # PHASE 3
        # ----------------------------------------------------

        elif choice == "13":

            quality = pedigree_data_quality_by_year(df)

            print("\nPedigree data quality:")
            print(quality.to_string(index=False))

            quality.to_csv(
                "rq1_pedigree_quality.csv",
                index=False
            )

            print(
                "\nSaved as rq1_pedigree_quality.csv"
            )

        elif choice == "14":

            diversity = parent_diversity_by_year(df)

            print("\nParent diversity:")
            print(diversity.to_string(index=False))

            diversity.to_csv(
                "rq1_pedigree_diversity.csv",
                index=False
            )

            print(
                "\nSaved as rq1_pedigree_diversity.csv"
            )

        elif choice == "15":

            reuse = parent_reuse_by_year(df)

            print("\nParent reuse:")
            print(reuse.to_string(index=False))

            reuse.to_csv(
                "rq1_parent_reuse.csv",
                index=False
            )

            print(
                "\nSaved as rq1_parent_reuse.csv"
            )

        elif choice == "16":

            concentration = parent_concentration_by_year(df)

            print("\nParent concentration:")
            print(concentration.to_string(index=False))

            concentration.to_csv(
                "rq1_parent_concentration.csv",
                index=False
            )

            print(
                "\nSaved as rq1_parent_concentration.csv"
            )

        elif choice == "17":

            run_phase_3(df)

        elif choice == "0":

            print("\nGoodbye!")
            break

        else:

            print("\nInvalid option. Please choose again.")


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()