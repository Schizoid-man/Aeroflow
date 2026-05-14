# Reports & Diagrams

This folder stores generated plots and evaluation artifacts. The diagrams are created from the transformed dataset (`datasets/transformed.csv`) and saved as PNG files.

## Generate diagrams

```bash
python reports/generate_diagrams.py
```

### Options

- `--input`: Path to the transformed CSV (default: `datasets/transformed.csv`).
- `--output-dir`: Output directory for the PNG files (default: `reports`).
- `--max-cities`: Number of top cities to include in city-based plots (default: `6`).

## Expected outputs

- `reports/aqi_over_time_by_city.png`
- `reports/aqi_category_counts.png`
- `reports/pm25_vs_aqi.png`
- `reports/aqi_distribution_by_city.png`
