# save as make_maruti_viz.py and run:
#   python make_maruti_viz.py --excel maruti_cars_multiple_cities.xlsx --outdir outputs

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


def clean_km(x: str) -> int | None:
    if pd.isna(x):
        return None
    # keep digits only
    digits = re.sub(r"[^\d]", "", str(x))
    return int(digits) if digits else None


def clean_price_to_inr(x: str) -> float | None:
    """Convert strings like '₹2.44 lakh' to INR (float, rupees)."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    # common formats in your sheet: '₹0.82 lakh', '₹2.44 lakh'
    s = s.replace("₹", "").replace(",", "").lower().strip()
    s = s.replace("inr", "")
    s = s.replace("rs.", "").replace("rs", "")
    s = s.replace("rupees", "")
    s = s.strip()
    # pull number
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    # detect unit
    if "lakh" in s:
        return num * 100_000
    if "crore" in s:
        return num * 10_000_000
    # assume already INR if no unit mentioned
    return num


def load_and_clean(excel_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(excel_path)
    # assume first sheet if name not known
    df = xl.parse(xl.sheet_names[0]).copy()

    # Standardize expected columns (case-sensitive as per your file)
    expected_cols = {
        "City": "City",
        "Year": "Year",
        "Model Name": "Model Name",
        "Kilometer Driven": "Kilometer Driven",
        "Number of Owners": "Number of Owners",
        "Transmission": "Transmission",
        "Fuel Type": "Fuel Type",
        "Location": "Location",
        "Price": "Price",
    }
    # (If your headers differ slightly, you could map them here.)

    # Clean columns
    df["Kilometer Driven"] = df["Kilometer Driven"].apply(clean_km)
    df["Price"] = df["Price"].apply(clean_price_to_inr)

    # Drop rows missing core fields used in charts
    df = df.dropna(subset=["City", "Price", "Kilometer Driven", "Fuel Type", "Transmission"]).reset_index(drop=True)

    # Normalize text categories
    df["City"] = df["City"].astype(str).str.strip()
    df["Fuel Type"] = df["Fuel Type"].astype(str).str.strip().str.title()
    df["Transmission"] = df["Transmission"].astype(str).str.strip().str.title()

    return df


def plot_bar_avg_price_per_city(df: pd.DataFrame, outpath: Path) -> Path:
    avg_price_city = df.groupby("City")["Price"].mean().sort_values()
    plt.figure(figsize=(9, 5.5))
    avg_price_city.plot(kind="bar", edgecolor="black")
    plt.title("Average Maruti Suzuki Car Prices per City")
    plt.ylabel("Average Price (INR)")
    plt.xlabel("City")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath


def plot_pie_fuel_distribution(df: pd.DataFrame, outpath: Path) -> Path:
    fuel_dist = df["Fuel Type"].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(6.5, 6.5))
    fuel_dist.plot(kind="pie", autopct="%1.1f%%", startangle=140)
    plt.title("Fuel Type Distribution Across Cities")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath


def plot_scatter_price_vs_km(df: pd.DataFrame, outpath: Path) -> Path:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Kilometer Driven"], df["Price"], alpha=0.6, s=50)
    plt.title("Price vs Kilometers Driven")
    plt.xlabel("Kilometers Driven")
    plt.ylabel("Price (INR)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath


def plot_hist_transmission_by_city(df: pd.DataFrame, outpath: Path) -> Path:
    """
    'Histogram: Transmission preference in different cities'
    We'll render overlaid histograms (counts) for the categorical 'Transmission'
    by city. For categorical data, this will behave like grouped bars per category.
    """
    plt.figure(figsize=(10, 6))
    cities = df["City"].unique()
    for city in cities:
        subset = df[df["City"] == city]
        plt.hist(
            subset["Transmission"],
            bins=len(subset["Transmission"].unique()),
            alpha=0.6,
            label=city,
        )

    plt.title("Transmission Preference in Different Cities")
    plt.xlabel("Transmission Type")
    plt.ylabel("Count")
    plt.legend(title="City")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()
    return outpath


def build_pdf(images: list[tuple[str, Path]], pdf_path: Path) -> Path:
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Maruti Suzuki Car Data Visualizations", styles["Title"]))
    story.append(Spacer(1, 18))

    for title, img_path in images:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Image(str(img_path), width=420, height=300))
        story.append(Spacer(1, 18))

    doc.build(story)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations & PDF from Maruti cars Excel.")
    parser.add_argument("--excel", type=str, default="maruti_cars_multiple_cities.xlsx", help="Path to Excel file")
    parser.add_argument("--outdir", type=str, default="outputs", help="Folder to save images and PDF")
    args = parser.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(excel_path)

    # Make charts
    bar_png = plot_bar_avg_price_per_city(df, outdir / "bar_avg_price_per_city.png")
    pie_png = plot_pie_fuel_distribution(df, outdir / "pie_fuel_distribution.png")
    scatter_png = plot_scatter_price_vs_km(df, outdir / "scatter_price_vs_km.png")
    hist_png = plot_hist_transmission_by_city(df, outdir / "hist_transmission_by_city.png")

    # Build PDF
    pdf_path = outdir / "maruti_visualizations.pdf"
    images = [
        ("Average Car Prices per City", bar_png),
        ("Fuel Type Distribution", pie_png),
        ("Price vs Kilometers Driven", scatter_png),
        ("Transmission Preference Across Cities", hist_png),
    ]
    build_pdf(images, pdf_path)

    print(f"Saved charts to: {outdir}")
    print(f"PDF report: {pdf_path}")


if __name__ == "__main__":
    main()
