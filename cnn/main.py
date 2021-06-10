""" The Plan
    Run all of the functions from the various files.
    You can choose what part of the code is run with the menu at the beginning.
"""

from labelling_trajectories import train, read_data
from flight_identification_CNN import flight_identification
from predict_anomalies import generate_images_for_dataset, get_image_paths, predict

if __name__ == "__main__":
    model_name = "model369"
    
    print("""
    ======== CNN IDENTIFICATION!!!! ========
    
    OPTIONS:
    ========
    0. STOP PROGRAM!!!!

    1. READ PARQUET FILE AND START LABELLING!!!!

    2. BUILD TENSORFLOW MODEL!!!!

    3. GENERATE IMAGES FOR ALL FLIGHTS IN DATASET!!!!

    4. GENERATE TABLE OF PREDICTIONS!!!!

    PLEASE ENTER YOUR CHOICE!!!!

    """)

    choice = int(input("!!!! ===> "))

    if choice == 1:
        ## PART ONE - READING FILE AND LABELLING
        # Read parquet file and get back dataset and flight_ids
        arrival_data, flight_ids = read_data()
        # Using the dataset and flight_ids, start interactive labelling
        train(arrival_data, flight_ids)
    elif choice == 2:
        ## PART TWO - BUILD TENSORFLOW MODEL
        # Using trained/labelled flight images, build Tensorflow model
        flight_identification(model_name)
    elif choice == 3:
        ## PART THREE - GENERATE IMAGES
        # Generate trajectory images for all flights in dataset
        generate_images_for_dataset()
    elif choice == 4:
        ## PART FOUR - GENERATE TABLE OF PREDICTIONS
        # Get image paths for all images
        image_paths = get_image_paths()
        # Run prediction engine to generate table of predictions
        predict(model_name, image_paths, "model369")
    else:
        print("STOPPING!!!!")