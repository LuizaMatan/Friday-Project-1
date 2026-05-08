import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Plot Settings ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab10")

# ── Read in Data ──────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "data", "*.csv")

dfs = []
for f in glob.glob(path):
    dfs.append(pd.read_csv(f))

if not dfs:
    raise FileNotFoundError(f"No CSV files found at: {path}")

# ── Pivot & Merge ─────────────────────────────────────────────────────────────

# Columns to always drop — non-numeric or redundant across all schemas
DROP_COLS = [
    "Id", "IndicatorCode", "Indicator", "ValueType", "Value",
    "FactValueTranslationID", "FactComments", "Language", "DateModified",
    "FactValueUoM", "FactValueNumericPrefix", "FactValueNumericLowPrefix",
    "FactValueNumericHighPrefix", "IsLatestYear", "Location type",
    "Period type", "Location",
]

# Metadata columns that are not indicator values
METADATA_COLS = {
    "TimeDim", "SpatialDimensionValueCode", "SpatialDimension",
    "ParentLocationCode", "ParentLocation", "TimeDimension",
    "TimeDimensionValue", "TimeDimensionBegin", "TimeDimensionEnd",
    "DisaggregatingDimension1", "DisaggregatingDimension1ValueCode",
    "DisaggregatingDimension2", "DisaggregatingDimension2ValueCode",
    "DisaggregatingDimension3", "DisaggregatingDimension3ValueCode",
    "DataSourceDimension", "DataSourceDimensionValueCode",
    "Low", "High", "Comments", "Date", "Dim1 type", "Dim1", "Dim1ValueCode",
    "Dim2 type", "Dim2", "Dim2ValueCode", "Dim3 type", "Dim3", "Dim3ValueCode",
    "DataSourceDimValueCode", "DataSource",
}

def normalize(df):
    """Standardize column names across different WHO export formats."""
    # SpatialDimValueCode always holds the ISO country code
    if "SpatialDimValueCode" in df.columns:
        df = df.rename(columns={"SpatialDimValueCode": "SpatialDimensionValueCode"})

    # Normalize time key
    if "Period" in df.columns and "TimeDim" not in df.columns:
        df = df.rename(columns={"Period": "TimeDim"})

    # Normalize value column
    if "FactValueNumeric" in df.columns and "NumericValue" not in df.columns:
        df = df.rename(columns={"FactValueNumeric": "NumericValue"})

    return df

def pivot_indicator(df):
    df = normalize(df)
    indicator = df["IndicatorCode"].iloc[0]
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    return (
        df.drop(columns=drop_cols)
        .drop_duplicates(subset=["TimeDim", "SpatialDimensionValueCode"])
        .rename(columns={"NumericValue": indicator})
    )

pivoted = [pivot_indicator(d) for d in dfs]

df = pivoted[0]
for p in pivoted[1:]:
    df = pd.merge(df, p, how="outer", on=["TimeDim", "SpatialDimensionValueCode"],
                  suffixes=("", "_dup"))

# Drop duplicate metadata columns produced by repeated merges
df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

# ── Clean Data ────────────────────────────────────────────────────────────────

# Drop rows where merge keys are missing
df = df.dropna(subset=["TimeDim", "SpatialDimensionValueCode"])

# Drop indicator columns that are more than 50% missing before imputing
threshold = 0.5
df = df.dropna(thresh=int(len(df) * (1 - threshold)), axis=1)

# Impute remaining missing indicator values with median
indicator_cols = [
    c for c in df.columns
    if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
]

for col in indicator_cols:
    if df[col].isna().any():
        median = df[col].median()
        df[col] = df[col].fillna(median)
        print(f"Imputed '{col}' with median: {median:.4f}")

# ── Inspect ───────────────────────────────────────────────────────────────────

print(f"df type: {type(df)}")
print(f"\ndf rows: {df.shape[0]}, df cols: {df.shape[1]}")
print(f"\nFeatures:\n{df.dtypes.to_string()}")
