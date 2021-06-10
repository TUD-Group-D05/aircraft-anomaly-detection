""" The Plan
        The end goal is to have a set of images of flight trajectories, labelled by flight ID and category, to be used as training data for a CNN.
        Make sure that your data file, e.g. arrival_dataset.parquet, is in the same folder as your python file.       
        1. Read the parquet file containing the dataset.
        2. From this file return a list of flight IDs.
        3. Use these flight IDs to get their respective location information at every time stamp.
        4. Create an image of each flight's trajectory.
            4.1 Find the geometric center of the trajectory.
            4.2 Create a plot around this center. Ensure that each plot has the same aspect ratio. This is required for the neural network.
        5. View this plot and identify which category (pattern) of flights it falls into, and label it.
        6. Save the image with the flight ID and the label to the folder 'images_labelled.' You need to create this folder in the same directory as your python file.
        7. Repeat steps 4-7 for as many images as desired. When you are done, you can exot the code with crtl + c.
    
    When labelling the trajectories, the following key was used:
    - Normal ID:0
    - Go around ID:1
    - Holding pattern ID:2
    - Double holding pattern ID:3
    - General delay ID:4
    - Extreme delay ID:5
    - Unknown ID:6

"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def read_data(filename="arrival_dataset.parquet"):
    # Read parquet file, using the pyarrow engine, as a Pandas dataframe
    arrival_data = pd.read_parquet(filename, engine="pyarrow")
    # Sort Pandas dataframe unique to 'flight_id' and convert to list
    flight_ids = arrival_data['flight_id'].unique().tolist()
    # Return the dataset and flight_ids
    return arrival_data, flight_ids

def train(arrival_data, flight_ids):
    # Setting up matplotlib
    fig, ax = plt.subplots()
    n = 1 # Counter variable to keep track of number of trajectories trained
    
    # If folder doesn't exist, make it.
    if not os.path.exists('images_labelled'):
        os.makedirs('images_labelled')

    try:
        for flight in flight_ids: # Loop through all flight_ids
            # Get data for specific flight (using its flight_id)
            flight_data = arrival_data.groupby("flight_id").get_group(flight)
            # Get the latitudes and longitudes for plotting
            latitudes = list(flight_data.latitude)
            longitudes = list(flight_data.longitude)
            ax.plot(longitudes, latitudes, color="black") # Plot these using matplotlib
            # Find boundary values of the trajectory
            min_x, min_y, max_x, max_y = min(longitudes), min(latitudes), max(longitudes), max(latitudes)
            # Find average (middle) values of the trajectory
            average_x, average_y = (( min_x + max_x ) / 2), (( min_y + max_y ) / 2)
            # Offset to use while plotting, this number ensures (for our case) that all flight trajectories 
            # are entirely displayed without making the plot unnecessarily large.
            offset = 0.75
            # Setting x and y bounds
            x_lower, x_upper = (average_x - offset), (average_x + offset)
            y_lower, y_upper = (average_y - offset), (average_y + offset)
            ax.set_ylim([y_lower, y_upper])
            ax.set_xlim([x_lower, x_upper])
            plt.gca().set_axis_off() # Ensure axis lines are not shown
            ax.set_aspect('equal', adjustable='box') # Set aspect ratio to 1:1
            plt.margins(0,0) # No margins
            plt.grid(False) # No grid
            plt.axis('off') # No axis
            plt.pause(0.001) # Used to prevent blocking
            # Get the classification number of current trajectory
            pattern_number = int(input(f"(n={n},fid={flight}) Enter pattern number: -"))
            n += 1 # Increment counter
            # Save figure to images_labelled folder
            fig.savefig(f"images_labelled/{flight}_{pattern_number}.png", bbox_inches = 'tight', pad_inches = 0)
            ax.clear() # Clear the plot
    except KeyboardInterrupt:
        print('Received Ctrl+C. Exiting.')