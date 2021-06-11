"""
Compares the outliers of different methods and gives the percentage of overlap with the neural network.

Parameters:
dbscan = "DBSCAN outliers input"
rw14, rw34, rw28 = "data point outliers inputs"
nn = "neural network outliers input"
approach ="approach outliers input"

Returns:
print the percentages of overlap
"""

# Import libraries
import pandas as pd

# Read up the DBSCAN outliers
print("Reading DBSCAN")
dbscan = pd.read_csv("dbscan_outliers.csv")

# Read and combine up data point outliers
print("Reading data points")
rw14 = pd.read_csv("outliers14.txt")
rw34 = pd.read_csv("outliers34.txt")
rw28 = pd.read_csv("outliers28.txt")
points = pd.concat([rw14, rw34, rw28], ignore_index=True)

# Read and clean up neural network outliers
print("Reading neural network")
nn = pd.read_csv("cnn_predictions.csv")
nn.drop(nn[nn['Anomaly T/F'] == False].index, inplace=True)
nn.reset_index(drop=True, inplace=True)
nn.rename(columns={'Flight ID': 'flight_id'}, inplace=True)

# Read and clean up approach outliers
print("Reading approach")
approach = pd.read_csv("approach.txt", delim_whitespace=True)
approach = approach[(approach == True).any(axis=1)]
approach.reset_index(drop=True, inplace=True)

print("Done reading")
print("===========================================================")

# Check for overlaping entries
overlap_dp = points.assign(overlap=points.flight_id.isin(nn.flight_id))
overlap_db = dbscan.assign(overlap=dbscan.flight_id.isin(nn.flight_id))
overlap_ap = approach.assign(overlap=approach.flight_id.isin(nn.flight_id))

# Print results
print(f"{round((((overlap_dp[overlap_dp.overlap == True].drop(overlap_dp.columns[1],axis = 1).size)/points.size)*100),2)}% of anomalies from data points are in neural network")
print(f"{round((((overlap_db[overlap_db.overlap == True].drop(overlap_db.columns[1],axis = 1).size)/dbscan.size)*100),2)}% of anomalies from DBSCAN are in neural network")
print(f"{round((((overlap_ap[overlap_ap.overlap == True].drop(overlap_ap.columns[1],axis = 1).size)/approach.size)*100),2)}% of anomalies from approach are in neural network")

print("===========================================================")
