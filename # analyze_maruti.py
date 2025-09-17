# analyze_maruti.py
# -----------------------------------------------------------
# Performs analysis on a Cars24-style Maruti dataset:
# 1) City-wise: number of listings & average prices
# 2) Fuel type distribution (per city, percent)
# 3) Transmission counts & pricing trends (overall + per city)
# 4) Price vs Kilometers Driven (correlation, slope, R^2)
#
# Saves results as CSVs in --outdir and (optionally) charts as PNGs.
#
# Usage:
#   pip install pandas numpy matplotlib openpyxl
#   python analyze_maruti.py --excel maruti_cars_multiple_cities.xlsx --outdir outputs
#   python analyze_maruti.py --excel data.xlsx --outdir outputs --no-charts
# -----------------------------------------------------------

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- Cleaning helpers -----------------------

def clean_km(x):
    """Convert strings like '72,583 km' to float kilometers."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^\d]", "", str(x))
    return float(s) if s else np.nan


def clean_price_to_inr(x):
    """Convert '₹2.44 lakh' or '2.44 lakh' to INR (float, rupees)."""
    if pd.isna(x):
        return np.nan
    s = str(x).lower().strip()
    s = (
        s.replace("₹", "")
         .replace(",", "")
         .replace("inr", "")
         .replace("rs.", "")
         .replace("rs", "")
         .replace("rupees", "")
         .strip()
    )
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return np.nan
    num = float(m.group(1))
    if "lakh" in s:
        return num * 100_000
    if "crore" in s:
        return num * 10_000_000
    return num


def load_and_clean(excel_path: Path) -> pd.DataFrame:
    """Load first sheet, align column names, and clean core fields."""
    xl = pd.ExcelFile(excel_path)
    df = xl.parse(xl.sheet_names[0]).copy()

    # Try to map common/variant headers to expected names
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "city":
            col_map[c] = "City"
        elif lc == "year":
            col_map[c] = "Year"
        elif lc.startswith("model"):
            col_map[c] = "Model Name"
        elif "kilometer" in lc or lc == "km":
            col_map[c] = "Kilometer Driven"
        elif "owner" in lc:
            col_map[c] = "Number of Owners"
        elif "transmission" in lc:
            col_map[c] = "Transmission"
        elif "fuel" in lc:
            col_map[c] = "Fuel Type"
        elif "location" in lc:
            col_map[c] = "Location"
        elif "price" in lc:
            col_map[c] = "Price"
    if col_map:
        df = df.rename(columns=col_map)

    required = ["City", "Kilometer Driven", "Fuel Type", "Transmission", "Price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    # Clean fields
    df["Kilometer Driven"] = df["Kilometer Driven"].apply(clean_km)
    df["Price"] = df["Price"].apply(clean_price_to_inr)

    # Normalize categories
    for col in ["City", "Fuel Type", "Transmission"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Drop rows missing essentials
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# ----------------------- Analyses -----------------------

def analyze_city(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("City")
    out = pd.DataFrame({
        "Listings": g.size(),
        "Avg Price (INR)": g["Price"].mean().round(0),
        "Median Price (INR)": g["Price"].median().round(0),
        "Avg KM Driven": g["Kilometer Driven"].mean().round(0),
        "Median KM Driven": g["Kilometer Driven"].median().round(0),
    }).sort_values("Listings", ascending=False)
    return out


def fuel_distribution(df: pd.DataFrame):
    counts = df.pivot_table(
        index="City", columns="Fuel Type", values="Model Name",
        aggfunc="count", fill_value=0
    )
    perc = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
    return counts, perc


def transmission_analysis(df: pd.DataFrame):
    overall_counts = df["Transmission"].value_counts().rename("Count").to_frame()
    overall_price = df.groupby("Transmission")["Price"].agg(
        Avg_Price_INR="mean", Median_Price_INR="median", Listings="count"
    ).round(0)

    by_city_counts = df.pivot_table(
        index="City", columns="Transmission", values="Model Name",
        aggfunc="count", fill_value=0
    )
    by_city_avg_price = df.groupby(["City", "Transmission"])["Price"].mean().round(0).unstack(fill_value=0)
    return overall_counts, overall_price, by_city_counts, by_city_avg_price


def price_vs_km(df: pd.DataFrame):
    x = df["Kilometer Driven"].values
    y = df["Price"].values

    # Correlation
    r = float(pd.Series(y).corr(pd.Series(x)))

    # Linear regression y = a + b x  (numpy returns [b, a])
    b, a = np.polyfit(x, y, 1)
    y_pred = a + b * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    metrics = pd.DataFrame({
        "Metric": ["Pearson r (Price vs KM)", "Slope (₹ per km)", "Slope (₹ per 10k km)", "Intercept (₹)", "R^2"],
        "Value": [round(r, 3), round(b, 2), round(b * 10_000, 0), round(a, 0), round(r2, 3)]
    })

    # Bin KM for a practical view
    bins = [0, 20000, 40000, 60000, 80000, 100000, 150000, np.inf]
    labels = ["0-20k", "20-40k", "40-60k", "60-80k", "80-100k", "100-150k", "150k+"]
    df = df.copy()
    df["KM Bin"] = pd.cut(df["Kilometer Driven"], bins=bins, labels=labels, right=False)
    by_bin = df.groupby("KM Bin")["Price"].agg(count="count", mean="mean", median="median").round(0)

    return metrics, by_bin, (a, b)


# ----------------------- Charts (optional) -----------------------

def make_charts(df: pd.DataFrame, outdir: Path, a_b: tuple[float, float]):
    outdir.mkdir(parents=True, exist_ok=True)

    # City-wise: listings & average price
    city_summary = analyze_city(df)
    plt.figure(figsize=(9, 5.5))
    city_summary["Listings"].plot(kind="bar", edgecolor="black")
    plt.title("Listings per City")
    plt.ylabel("Count")
    plt.xlabel("City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "city_listings_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    (city_summary["Avg Price (INR)"] / 100000).plot(kind="bar", edgecolor="black")
    plt.title("Average Price per City (Lakh ₹)")
    plt.ylabel("Lakh ₹")
    plt.xlabel("City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "city_avg_price_bar.png", dpi=180)
    plt.close()

    # Fuel distribution (stacked percent)
    counts, perc = fuel_distribution(df)
    perc.plot(kind="bar", stacked=True, figsize=(10, 6), edgecolor="black")
    plt.title("Fuel Type Distribution per City (%)")
    plt.ylabel("Percent")
    plt.xlabel("City")
    plt.legend(title="Fuel Type")
    plt.tight_layout()
    plt.savefig(outdir / "fuel_type_distribution_stacked.png", dpi=180)
    plt.close()

    # Transmission counts (grouped)
    overall_counts, _, by_city_counts, by_city_avg_price = transmission_analysis(df)
    by_city_counts.plot(kind="bar", figsize=(10, 6), edgecolor="black")
    plt.title("Transmission Counts by City")
    plt.ylabel("Count")
    plt.xlabel("City")
    plt.legend(title="Transmission")
    plt.tight_layout()
    plt.savefig(outdir / "transmission_counts_by_city.png", dpi=180)
    plt.close()

    # Price vs KM scatter + regression line
    a, b = a_b
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Kilometer Driven"], df["Price"], alpha=0.6, s=25)
    xs = np.linspace(df["Kilometer Driven"].min(), df["Kilometer Driven"].max(), 100)
    plt.plot(xs, a + b * xs)  # regression line
    plt.title("Price vs Kilometers Driven")
    plt.xlabel("Kilometers Driven")
    plt.ylabel("Price (₹)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outdir / "price_vs_km_scatter.png", dpi=180)
    plt.close()


# ----------------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze Maruti used-car dataset.")
    ap.add_argument("--excel", required=True, help="Path to Excel file (first sheet will be used).")
    ap.add_argument("--outdir", default="outputs", help="Folder to save CSVs (and charts).")
    ap.add_argument("--no-charts", action="store_true", help="Skip generating PNG charts.")
    args = ap.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(excel_path)

    # 1) City-wise summary
    city_summary = analyze_city(df)
    city_summary.to_csv(outdir / "city_summary.csv")

    # 2) Fuel distribution (counts & percent)
    fuel_counts, fuel_percent = fuel_distribution(df)
    fuel_counts.to_csv(outdir / "fuel_counts_by_city.csv")
    fuel_percent.to_csv(outdir / "fuel_percent_by_city.csv")

    # 3) Transmission analysis
    trans_counts, trans_price, city_trans_counts, city_trans_price = transmission_analysis(df)
    trans_counts.to_csv(outdir / "transmission_counts_overall.csv")
    trans_price.to_csv(outdir / "transmission_pricing_overall.csv")
    city_trans_counts.to_csv(outdir / "transmission_counts_by_city.csv")
    city_trans_price.to_csv(outdir / "avg_price_by_city_and_transmission.csv")

    # 4) Price vs KM
    metrics, km_bin_price, a_b = price_vs_km(df)
    metrics.to_csv(outdir / "price_vs_km_metrics.csv", index=False)
    km_bin_price.to_csv(outdir / "avg_price_by_km_bins.csv")

    # Optional charts
    if not args.no_charts:
        make_charts(df, outdir, a_b)

    # Console highlights
    top_city_listings = city_summary["Listings"].idxmax()
    highest_avg_city = city_summary["Avg Price (INR)"].idxmax()
    lowest_avg_city = city_summary["Avg Price (INR)"].idxmin()
    manual = int(trans_counts.loc["Manual", "Count"]) if "Manual" in trans_counts.index else 0
    automatic = int(trans_counts.loc["Automatic", "Count"]) if "Automatic" in trans_counts.index else 0

    print("\n=== SUMMARY ===")
    print(f"Top city by listings: {top_city_listings}")
    print(f"Highest average price city: {highest_avg_city}")
    print(f"Lowest average price city: {lowest_avg_city}")
    print(f"Transmission counts (overall): Manual={manual}, Automatic={automatic}")
    print(metrics.to_string(index=False))
    print(f"\nAll CSVs saved to: {outdir}")
    if not args.no_charts:
        print("Charts saved (PNG) in the same folder.")


if __name__ == "__main__":
    main()
