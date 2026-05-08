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

# Plot settings
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab10")


script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "data", "*.csv")

dfs = []
for f in glob.glob(path):
    dfs.append(pd.read_csv(f))

if not dfs:
    raise FileNotFoundError(f"No CSV files found at: {path}")

def normalize(df):
    """Standardize column names across different WHO export formats."""
    # Normalize spatial key
    if "SpatialDimValueCode" in df.columns and "SpatialDimensionValueCode" not in df.columns:
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
    drop_cols = [c for c in ["Id", "IndicatorCode"] if c in df.columns]
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

# Drop duplicate metadata columns from repeated merges
df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

print(f"df type: {type(df)}")
print(f"\ndf rows: {df.shape[0]}, df cols: {df.shape[1]}")
# print(f"\nFeatures:\n{df.dtypes.to_string()}")
# print(df.info())
# print(df.describe())

