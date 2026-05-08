import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Store data for plots :-)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("tab10")


# Read in Data:
path = "data/*.csv"

dfs = []

# Read in data
for f in glob.glob(path):
    curr_df = pd.read_csv(f)
    dfs.append(curr_df)

# Merge the frames
df = dfs[2]
print(df.columns)
for m_df in dfs[3:]:
    if "TimeDim" in m_df.columns and "SpacialDimensionValueCode" in m_df:
        df = pd.merge(df, m_df, how="outer", on=["TimeDim", "SpatialDimensionValueCode"])




# Checking the type and shape - we need to know num. rows and cols for later
print(f"df type: {type(df)}")
print(f"\ndf rows: {df.shape[0]}, df cols: {df.shape[1]}")
print(f"\nFeatures: {df.dtypes.to_string()}")


