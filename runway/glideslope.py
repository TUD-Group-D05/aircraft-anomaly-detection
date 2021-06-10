print("starting program")

from backend import time_statement
from glideslope_backend import Runway, Custom_Flight, get_arrival_dataset, plot_custom_flights, save_fig, process_standard_deviations, process_global_outliers

import matplotlib
matplotlib.use('Agg') #use if no plotting is required
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

time_statement("Starting time:")
print("starting reading")

arrival_dataset = get_arrival_dataset("dataset/filtered/filtered_arrival_dataset_1000.pkl") 

time_statement("Reading time:")

std = 2

runway_names = ["all", "14", "32","28","10","34","16",] #This is a really inefficient implementation

for runway_name in runway_names:
    print(f"processing all flights landing on runway: {runway_name}")
    i = 0
    glideslope_standard_deviations = []
    speed_standard_deviations = []
    track_standard_deviations = []
    custom_flights = []
    max_index = int(len(arrival_dataset))
    for flight in arrival_dataset[0:max_index]:
        i += 1
        progress = int(i/max_index*100)
        if progress%20==0 and progress != 0:
            print(f"{progress}%")
        #if flight.data["runway"].iloc[0] == "16":
        if runway_name == "all":
            if flight.data["runway"].iloc[0] != "N/A":
                custom_flight = Custom_Flight(flight)
                
                for approach in custom_flight.approaches:
                    glideslope_standard_deviations.append([approach.glideslope_standard_deviation, approach, custom_flight])
                    speed_standard_deviations.append([approach.speed_standard_deviation, approach, custom_flight])
                    track_standard_deviations.append([approach.track_standard_deviation, approach, custom_flight])

                custom_flights.append(custom_flight)
        else:
            if flight.data["runway"].iloc[0] == runway_name:
                custom_flight = Custom_Flight(flight)
                
                for approach in custom_flight.approaches:
                    glideslope_standard_deviations.append([approach.glideslope_standard_deviation, approach, custom_flight])
                    speed_standard_deviations.append([approach.speed_standard_deviation, approach, custom_flight])
                    track_standard_deviations.append([approach.track_standard_deviation, approach, custom_flight])

                custom_flights.append(custom_flight)

    time_statement(f"Processing time for runway {runway_name}:")

    if len(custom_flights) > 0:
        plot = plot_custom_flights(custom_flights, runway_name)
        save_fig(plot, f'{runway_name}/Figures/{std}/{len(custom_flights)}', 'all_flights.png')

        time_statement("Plotting time for runway {runway_name}:")

        vertical_outliers = process_standard_deviations(glideslope_standard_deviations, std, runway_name, number_of_bars = 100, x_title = "Vertical standard deviation from glideslope[m]", y_title = "Frequency", title = "Frequencies of the standard deviation in the altitude during approach")#, save_individual_plots= False)
        speed_outliers = process_standard_deviations(speed_standard_deviations, std, runway_name,number_of_bars = 100, x_title = 'Standard deviation of approach speed [kts]', y_title = "Frequency", title = 'Frequencies of the standard deviation in the approach speed')#, save_individual_plots= False)
        track_outliers = process_standard_deviations(track_standard_deviations, std, runway_name, number_of_bars = 100, x_title = 'Standard deviation of track angle [deg]', y_title = "Frequency", title = 'Frequencies of the standard deviation in the approach tracks')#, save_individual_plots= False)

        outliers_collection = {"vertical": vertical_outliers, "speed": speed_outliers, "track": track_outliers}
        process_global_outliers(outliers_collection, custom_flights, std, runway_name)

        plt.show()
    else:
        print(f"runway {runway_name} contains no flights")