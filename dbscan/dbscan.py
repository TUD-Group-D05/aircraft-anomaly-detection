"""
Plots and saves the results of clustering using the DBSCAN algorithm

Parameters:
dataset = "title of the dataset file"
extension = "extension of the dataset file"

filter = "filter dataset (bool)"
unwrap = "unwrap dataset (bool)"
file_output = "output files other than the outliers (bool)"

eps = 1.2
min_samples = 125

Returns:
Figure with the clustered trajectories and outliers, alongside with a .csv containing the outliying flights 
(or more files depending on parameters)
"""
# Import libraries
import pandas as pd
from random import sample
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from traffic.core import Traffic
from traffic.core.projection import EuroPP, CH1903
from traffic.drawing import countries
from itertools import islice, cycle
import matplotlib.pyplot as plt

# Define parameters
dataset = "arrival_dataset"
extension = ".pkl"
path = dataset + extension

filter = False
unwrap = False
file_output = False

eps = 1.2
min_samples = 125

# Define functions
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


# Filter dataset
if filter:
    dataset_filtered = pd.read_pickle(path)
    dataset_filtered = clean_dataset(dataset_filtered)
    dataset_filtered.data.dropna(how="any", inplace=True)
else:
    dataset_filtered = Traffic.from_file(dataset+"_filtered"+extension)

# Unwrap dataset
if unwrap:
    dataset_unwrapped = dataset_filtered.unwrap().eval()
else:
    dataset_unwrapped = Traffic.from_file(dataset + "_unwrapped" + extension)

# Cluster dataset
dataset_dbscan = dataset_unwrapped.clustering(
    nb_samples=15,
    projection=EuroPP(),
    features=["x", "y", "track_unwrapped"],
    clustering=DBSCAN(
        eps=eps,
        min_samples=min_samples),
    transform=StandardScaler(),).fit_predict()

# Print clusters and prepare data
print(dict(dataset_dbscan.groupby(["cluster"]).agg(
    {"flight_id": "nunique"}).flight_id))
outliers = dataset_dbscan.query(f"cluster == {-1}")
n_clusters = 1 + dataset_dbscan.data.cluster.max()

# Define plot colors
color_cycle = cycle(
    "#fc0707 #fc8a07 #fcc307 #f9d104 #d5f904 #5ef904 "
    "#04f962 #04f9cd #04c0f9 #0415f9 #8304f9 #e104f9 "
    "#f904c8 #f9047f #0e6b00 #00636b #6b0069 #6b2e00".split()
)
colors = list(islice(color_cycle, n_clusters))

# Plot the trajectories
with plt.style.context("traffic"):
    fig, ax = plt.subplots(1, figsize=(
        15, 10), subplot_kw=dict(projection=CH1903()))
    ax.add_feature(countries(facecolor="#ffffff", linewidth=0.5))
    for cluster in range(0, n_clusters):

        current_cluster = dataset_dbscan.query(f"cluster == {cluster}")
        centroid = current_cluster.centroid(15, projection=CH1903())
        centroid.plot(ax, color=colors[cluster], alpha=1, linewidth=2)
        centroid_mark = centroid.at_ratio(0.45)

        centroid_mark.plot(
            ax,
            color=colors[cluster],
            s=500,
            text_kw=dict(s=""),
        )
        sample_size = min(20, len(current_cluster))
        for flight_id in sample(current_cluster.flight_ids, sample_size):
            current_cluster[flight_id].plot(
                ax, color="grey", alpha=0.1, linewidth=2
            )

# Output files
if file_output:
    dataset_filtered.to_pickle(dataset+"_filtered"+extension)
    dataset_unwrapped.to_pickle(dataset + "_unwrapped" + extension)
    dataset_dbscan.to_pickle(dataset + "_clustered_" +
                             str(eps) + "_" + str(min_samples) + extension)

outliers = outliers.groupby(["flight_id"]).agg(
    {"flight_id": "nunique"}).flight_id
outliers.to_csv(dataset+"_outliers.csv")

# Show plot
plt.show()
