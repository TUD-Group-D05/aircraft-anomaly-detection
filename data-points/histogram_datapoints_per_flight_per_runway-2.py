import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
from traffic.core import Traffic
from traffic.data.samples import quickstart
from datetime import datetime

#matplotlib.use('TkAgg')
# from traffic.core.projection import Amersfoort, GaussKruger, Lambert93, EuroPP
# from traffic.drawing import countries

# from traffic.data.samples import quickstart, airbus_tree, belevingsvlucht
# from traffic.core import loglevel
# loglevel('DEBUG')
# from traffic.drap
print('Starting')
new_arrival_data = Traffic.from_file("dataset/filtered/filtered_arrival_dataset.pkl")
arrival_data = pd.read_parquet("dataset/arrival_dataset.parquet", engine = "pyarrow")


quickstart_id = Traffic.from_file("dataset/filtered/filtered_arrival_dataset.pkl")



print('Hold up still running')

arrivals = arrival_data[(arrival_data['timestamp'] > '2019-09-30') & (arrival_data['timestamp'] < '2019-11-30')]


print('Here we go')
random_variable1 = arrivals.query("runway=='14'")
random_variable2 = random_variable1.groupby("flight_id")

quickstart_id2 = Traffic.from_file('random_variable1')

num_points = []
bins = 200

for name,group in random_variable2:
    num_points.append(group.agg("count").iloc[0])

#stdev line
x=np.mean(num_points)+3*np.std(num_points)

#print(num_points)
fig, ax = plt.subplots(figsize=(20,10))
ax.hist(num_points, bins, label="Frequency of the amount of data points")
plt.axvline(x, ls = "--", color='#2ca02c', alpha=0.7, label= "3 Times the standard deviation", linewidth=7.0)
plt.xlabel('Amount of data points', fontsize= 30)
plt.ylabel("Frequency", fontsize = 30)
ax.legend(loc="upper right", fontsize=17)
plt.title('Frequency fo the amount of data points required\n to visualize a flightpath for runway 14', fontsize = 35)
matplotlib.pyplot.xticks(fontsize=15)
matplotlib.pyplot.yticks(fontsize=15)
plt.show()


