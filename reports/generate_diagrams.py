"""Generate exploratory diagrams from the transformed AQI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _apply_style() -> None:
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city", "aqi", "pm2_5"])
    return df


def plot_aqi_over_time_by_city(
    df: pd.DataFrame, output_dir: Path, max_cities: int
) -> Path:
    top_cities = df["city"].value_counts().head(max_cities).index
    subset = df[df["city"].isin(top_cities)].copy()
    subset["day"] = subset["date"].dt.to_period("D").dt.to_timestamp()

    daily = (
        subset.groupby(["day", "city"], as_index=False)["aqi"]
        .mean()
        .sort_values("day")
    )
    pivot = daily.pivot(index="day", columns="city", values="aqi")

    fig, ax = plt.subplots(figsize=(12, 6))
    for city in pivot.columns:
        ax.plot(pivot.index, pivot[city], label=city, linewidth=2)

    ax.set_title("Daily Average AQI by City")
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend(title="City", ncol=2, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = output_dir / "aqi_over_time_by_city.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_aqi_category_counts(df: pd.DataFrame, output_dir: Path) -> Path:
    counts = df["aqi_category"].value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color="#4c72b0")
    ax.set_title("AQI Category Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Record Count")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    output_path = output_dir / "aqi_category_counts.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_pm25_vs_aqi(df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["pm2_5"], df["aqi"], alpha=0.35, s=22, color="#dd8452")
    ax.set_title("PM2.5 vs AQI")
    ax.set_xlabel("PM2.5")
    ax.set_ylabel("AQI")
    ax.grid(True, linestyle="--", alpha=0.3)

    output_path = output_dir / "pm25_vs_aqi.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_aqi_distribution_by_city(
    df: pd.DataFrame, output_dir: Path, max_cities: int
) -> Path:
    top_cities = df["city"].value_counts().head(max_cities).index
    subset = df[df["city"].isin(top_cities)]

    fig, ax = plt.subplots(figsize=(10, 6))
    data = [subset[subset["city"] == city]["aqi"] for city in top_cities]
    ax.boxplot(data, tick_labels=top_cities, showfliers=False)
    ax.set_title("AQI Distribution for Top Cities")
    ax.set_xlabel("City")
    ax.set_ylabel("AQI")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    output_path = output_dir / "aqi_distribution_by_city.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AQI diagrams from transformed dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/transformed.csv"),
        help="Path to transformed CSV input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to write PNG diagrams.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=6,
        help="Number of top cities to include in city-based plots.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _apply_style()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    outputs = [
        plot_aqi_over_time_by_city(df, output_dir, args.max_cities),
        plot_aqi_category_counts(df, output_dir),
        plot_pm25_vs_aqi(df, output_dir),
        plot_aqi_distribution_by_city(df, output_dir, args.max_cities),
    ]

    print("Generated diagrams:")
    for path in outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
