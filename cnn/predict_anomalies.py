""" The Plan
        The end goal is to have a pandas dataframe with all flight IDs, thier predicted category, and a T/F for whether each one is an anomaly or not.
        Make sure that the images dataset folder is present in the current working directory and that there is a Tensorflow model
        1. Get all images' paths in a list.
        2. Define the different classification names in order in a list.
        3. Load the Tensorflow model to be used for predictions
        4. For every image, read the image as grayscale and normalize the values.
            4.1 Extract flight id from the image filename.
            4.2 Reshape the image data to fit the expected input to the model.
            4.3 Use the model to get prediction for the image.
            4.4 Obtain the prediction with the highest probability, and add this to the list.
        5. Use Pandas to convert list into dataframe.
        6. Convert Pandas dataframe to csv and store file with given output name.
"""

import numpy as np, tensorflow as tf, pandas as pd, cv2, os
import matplotlib, matplotlib.pyplot as plt

def get_image_paths():
    current_dir = os.getcwd()  # Get current working directory path
    directory = "images_dataset" #Name of the dataset image folder
    # Read every image from its path as a numpy array and divide by 255 to get a fraction between 0 and 1. 
    # The images are all saved in a list.
    image_paths = [os.path.join(current_dir, directory, file) 
        for file in os.listdir(os.path.join(current_dir, directory)) if file.endswith('.png')]
    return image_paths

def predict(model_name, image_paths, output_name):
    # Define all of the classes that a flight can fall into, note that the order here matters since each of them has an index
    class_names = ['Normal', 'Go Around', 'Single Holding Pattern', 
    'Double Holding Pattern', 'General Delay', 'Extreme Delay', 'Unknown'] # Define all of the classes that a flight can fall into, note that the order here matters since each of them has an index
    # Create empty table that will hold all prediction records once complete
    final_list = []
    # Load the model using the given filename
    model = tf.keras.models.load_model(f'saved_models/{model_name}')

    number_of_flights = len(image_paths) # Get total number of flights in dataset
    
    number = 0 # Keep track of number of flights saved until now

    # For every image, read it and its flight ID, make an array of the correct size to input into the model, use the model to 
    # predict which class of trajectory it falls into and add these results into a list.
    for image_path in image_paths:
        number += 1
        testingimage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255 # Diving by 255 so that all color values in the array are between 0 and 1.
        flightid = ((image_path).split("\\")[-1][:-4]) # Get the flight ID
        testingimage = np.reshape(testingimage, (1, 369, 369, 1)) # Make array of color values with the correct shape for the model.
        predictions = model.predict(testingimage) # Predict the class inwhich the trajectory falls
        final_prediction = class_names[np.argmax(predictions)] # Take the most likely class as the label for the flight.
        anomaly = False if final_prediction == "Normal" else True
        final_list.append([flightid, final_prediction, anomaly])
        print(f"Predicted flight id={flightid} ({number} of {number_of_flights}).")
    
    # Make a dataframe containing all the information.
    df = pd.DataFrame(final_list, columns=['Flight ID', 'Prediction', 'Anomaly T/F'])
    df.to_csv(f'{output_name}.csv', index=False)

def generate_images_for_dataset(directory="images_dataset"):
    matplotlib.use('Agg') # Uses the Agg backend (does not render on screen)

    arrival_data = pd.read_parquet("arrival_dataset.parquet", engine = "pyarrow")
    flight_ids = arrival_data['flight_id'].unique().tolist()

    number_of_flights = len(flight_ids) # Get total number of flights in dataset

    print(f"Generating images for {number_of_flights} flights.")

    number = 0 # Keep track of number of flights saved until now
    
    try:
        for flight in flight_ids:
            number += 1
            # Get data for specific flight (using its flight_id)
            flight_data = arrival_data.groupby("flight_id").get_group(flight)
            # Get the latitudes and longitudes for plotting
            latitudes = list(flight_data.latitude)
            longitudes = list(flight_data.longitude)
            fig, ax = plt.subplots()
            ax.plot(longitudes, latitudes, color="black")
            # Find boundary values of the trajectory
            min_x, min_y, max_x, max_y = min(longitudes), min(latitudes), max(longitudes), max(latitudes) # Find boundary values of the trajectory
            average_x, average_y = (( min_x + max_x ) / 2), (( min_y + max_y ) / 2) # Find average (middle) values of the trajectory
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
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.grid(False) # No grid
            plt.axis('off') # No axis
            # Save figure to folder
            fig.savefig(f"{directory}/{flight}.png", bbox_inches = 'tight', pad_inches = 0)
            print(f"Saving {flight}.png ({number} of {number_of_flights}).")
            ax.clear() # Clear the plot
    except KeyboardInterrupt:
        print('Received Ctrl+C. Exiting.')