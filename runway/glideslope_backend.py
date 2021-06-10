from traffic.core import Traffic, traffic
from traffic.data import airports
from traffic.core.projection import EuroPP
from traffic.drawing import countries

from backend import distance_points, convert_to_XY

import numpy as np
import math as math
from operator import itemgetter
import os

import matplotlib
#matplotlib.use('Agg') #use if no plotting is required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

class Runway():
    def __init__(self, name:str, threshold1:list, threshold2:list, nom_glideslope:float, plot_runway:str = ""):
        """
        Initialise the runway

        Parameters:
        name: str
        threshold1: list (latitude, longitude)
        threshold2: list (latitude, longitude)
        nom_glideslope:float (nominal glideslope on this runway) 
        plot_runway:str = "" (optional: decide if the runway needs to be plotted in xy space for debugging purposes)

        Returns:
        /
        """
        self.name = name
        self.nom_glideslope = nom_glideslope
        self.threshold1 = threshold1 #latitude, longitude [Deprecated]
        self.threshold1_xy = (convert_to_XY(self.threshold1[1], self.threshold1[0])) #x, y
        self.threshold2 = threshold2 #latitude, longitude [Deprecated]
        self.threshold2_xy = (convert_to_XY(self.threshold2[1], self.threshold2[0])) #x, y
        self.slope, self.perp_slope = self.calculate_slopes(self.threshold1_xy, self.threshold2_xy)
        
        self.threshold_perp_xy = self.calculate_second_threshold_point(self.perp_slope, self.threshold1_xy) #x, y

        #plot the relevant points for debugging purposes
        if name == plot_runway:
            print(f"runway name: {name}")
            print(f"threshold1 xy: {self.threshold1_xy}")
            print(f"threshold2 xy: {self.threshold2_xy}")
            print(f"slope: {self.slope}")
            print(f"perpendicular slope: {self.perp_slope}")
            
            plt.scatter(self.threshold1_xy[0], self.threshold1_xy[1], label= "thresh")
            plt.plot([self.threshold1_xy[0], self.threshold2_xy[0]], [self.threshold1_xy[1], self.threshold2_xy[1]], label= "runway")
            plt.scatter(self.threshold2_xy[0], self.threshold2_xy[1], label= "thresh2")
            plt.plot([self.threshold1_xy[0], self.threshold_perp_xy[0]], [self.threshold1_xy[1], self.threshold_perp_xy[1]], label= "thresh perp")
            plt.scatter(self.threshold_perp_xy[0], self.threshold_perp_xy[1], label= "thresh perp")
            plt.gca().set_aspect('equal', adjustable='box') #Make the scale of the plot 1:1 to preserve perpendicularity
            plt.show()

        self.thershold2_sign = self.get_side_of_threshold(self.threshold2_xy)
        self.thershold_perp_sign = self.get_side_of_runway(self.threshold_perp_xy)

    def calculate_second_threshold_point(self, slope:float, threshold1:list):
        """
        Creating a new virtual threshold at the end of the runway using the provided slope and first threshold 

        Parameters:
        slope:float 
        threshold1:list

        Returns:
        new_threshold_coordinate:tuple (new_threshold_coord_x, new_threshold_coord_y)
        """
        dx = 1000
        new_threshold_coord_x = threshold1[0] + dx
        new_threshold_coord_y = threshold1[1] + dx*slope
        new_threshold_coordinate = (new_threshold_coord_x, new_threshold_coord_y)
        return new_threshold_coordinate 

    def calculate_slopes(self, threshold1_xy: tuple, threshold2_xy: tuple):
        """
        Calculate the slope of the runway in XY space using two thresholds

        Parameters:
        threshold1_xy: tuple
        threshold2_xy: tuple

        Returns:
        slope: float (the slope of the runway in XY space)
        perp_slope: float (the negative inverse of the runway in XY space)
        """
        slope = (threshold2_xy[1] - threshold1_xy[1])/(threshold2_xy[0] - threshold1_xy[0])
        perp_slope = -1/slope
        return slope, perp_slope

    def get_side_of_line(self, coord1: tuple, coord2:tuple, coordx:tuple):
        """
        Calculate which side of the line a point is on

        Parameters:
        coord1: tuple (first point defining the line)
        coord2: tuple (second point defining the line)
        coordx: tuple (point to check)

        Returns:
        sign: int (-1 or +1 depending on the side of the line)
        """
        Ax = coord1[0]
        Ay = coord1[1]
        Bx = coord2[0]
        By = coord2[1]
        X = coordx[0]
        Y = coordx[1]
        pos = (Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax)
        sign = math.copysign(1, pos)
        return sign

    def get_side_of_threshold(self, coordx: tuple):
        """
        Check which side of the first threshold a point is in runway direction 

        Parameters:
        coordx: tuple (point to check)

        Returns:
        sign: int (-1 or +1 depending on the side of the threshold)
        """
        coord1 = self.threshold1_xy
        coord2 = self.threshold_perp_xy
        sign = self.get_side_of_line(coord1, coord2, coordx)
        return sign

    def get_side_of_runway(self, coordx: tuple):
        """
        Check which side of the runway a point is in the direction perpendicular to the runway

        Parameters:
        coordx: tuple (point to check)

        Returns:
        sign: int (-1 or +1 depending on the side of the runway)
        """
        coord1 = self.threshold1_xy
        coord2 = self.threshold2_xy
        sign = self.get_side_of_line(coord1, coord2, coordx)
        return sign

    def get_parametric_distance_from_threshold(self, x:float, y:float, plot:bool = False):
        """
        Calculate the distance of a point to the first runway threshold in parallel direction

        Parameters:
        x:float
        y:float
        plot: bool (optional: plot the relevant points for debugging purposes)

        Returns:
        distance: float
        """
        thresh_x = self.threshold1_xy[0]
        thresh_y = self.threshold1_xy[1]
        
        #Calculate the intersection of the runway line and a perpendicular line going through the given point
        intersect_x = (y-thresh_y-self.perp_slope*x+self.slope*thresh_x)/(self.slope-self.perp_slope)
        intersect_y = thresh_y + self.slope*(intersect_x - thresh_x)
        distance = distance_points([intersect_x, intersect_y], [thresh_x, thresh_y])

        #plot the relevant points for debugging purposes
        if plot:
            plt.plot([thresh_x, self.threshold2_xy[0]], [thresh_y, self.threshold2_xy[1]], label= "thresh")
            plt.scatter(self.threshold2_xy[0], self.threshold2_xy[1], label= "thresh2")
            plt.plot([thresh_x, self.threshold_perp_xy[0]], [thresh_y, self.threshold_perp_xy[1]], label= "thresh perp")
            plt.plot([thresh_x, intersect_x], [thresh_y, intersect_y], label= "line1")
            plt.plot([x, intersect_x], [y, intersect_y], label= "line2")
            plt.scatter(x, y, label= "point")
            plt.scatter(intersect_x, intersect_y, label= "intersect")
            plt.gca().set_aspect('equal', adjustable='box') #Make the scale of the plot 1:1 to preserve perpendicularity
            plt.show()

        point_sign = self.get_side_of_threshold([x, y])
        side_product = self.thershold2_sign * point_sign
        if side_product >= 0:
            distance = -distance

        return distance

    def get_parametric_distance_from_runway(self, x:float, y:float, plot:bool = False):
        """
        Calculate the distance of a point to the runway in perpendicular direction

        Parameters:
        x:float
        y:float
        plot: bool (optional: plot the relevant points for debugging purposes)

        Returns:
        distance: float
        """

        thresh_x = self.threshold1_xy[0]
        thresh_y = self.threshold1_xy[1]

        perp_x = self.threshold_perp_xy[0]
        perp_y = self.threshold_perp_xy[1]
       
        #Calculate the intersection of the line perpendicular to the runway and a line going parallel to the runway and through the tested point
        intersect_x = (-self.slope * x + y + self.perp_slope*perp_x-perp_y)/(self.perp_slope-self.slope)
        intersect_y = perp_y + self.perp_slope*(intersect_x - perp_x)
        distance = distance_points([intersect_x, intersect_y], [thresh_x, thresh_y])

        #plot the relevant points for debugging purposes
        if plot:
            plt.plot([thresh_x, self.threshold2_xy[0]], [thresh_y, self.threshold2_xy[1]], label= "thresh")
            plt.scatter(self.threshold2_xy[0], self.threshold2_xy[1], label= "thresh2")
            plt.plot([thresh_x, self.threshold_perp_xy[0]], [thresh_y, self.threshold_perp_xy[1]], label= "thresh perp")
            plt.plot([thresh_x, intersect_x], [thresh_y, intersect_y], label= "line1")
            plt.plot([x, intersect_x], [y, intersect_y], label= "line2")
            plt.scatter(x, y, label= "point")
            plt.scatter(intersect_x, intersect_y, label= "intersect")
            plt.gca().set_aspect('equal', adjustable='box') #Make the scale of the plot 1:1 to preserve perpendicularity
            plt.show()

        point_sign = self.get_side_of_runway([x, y]) 
        side_product = -self.thershold_perp_sign * point_sign
        if side_product >= 0:
            distance = -distance

        return distance

class Custom_Flight():
    def __init__(self, flight:traffic.Flight):
        """
        Creates custom flight equivalent to traffic flight with more manageable datatypes and extra features

        Parameters:
        flight:traffic.Flight

        Returns:
        /
        """
        #Read from flight
        self.tracks = list(flight.data["latitude"])
        self.longitudes = list(flight.data["longitude"])
        self.latitudes = list(flight.data["latitude"])
        self.altitudes = list(np.array(flight.data["altitude"])*0.3048) #converting from feet to meters
        self.groundspeeds = list(flight.data["groundspeed"])
        self.timestamps = list(flight.data["timestamp"])
        self.flight = flight

        #Perform own processing
        self.Xs, self.Ys = convert_to_XY(self.longitudes, self.latitudes)
        self.timestamp_delta = self.timestamps_to_delta(self.timestamps)
        self.runway_name = flight.data["runway"].iloc[0]
        self.runway = self.get_runway()
        self.parr_runway_dists, self.perp_runway_dists = self.get_relative_distances()
        self.vertical_angles, self.average_vertical_angles, self.long_term_vertical_angles = self.compute_vertical_angles()
        self.approaches = self.get_ILS_approaches()
    
    def timestamps_to_delta(self, timestamps:list):
        """
        Calculates the average time inbetween data timestamps 

        Parameters:
        timestamps:list

        Returns:
        float (average time inbetween)
        """
        times = []
        for index, timestamp in enumerate(timestamps):
            if index < len(timestamps)-2:
                tdelta = abs((timestamp - timestamps[index+1]).total_seconds())
                times.append(tdelta)
        return sum(times)/len(times)
            
    def get_runway(self):
        for runway in runways:
            if runway.name == self.runway_name:
                return runway

    def get_ILS_approaches(self):
        """
        Constructs a list of approaches on the ILS glideslope

        Parameters:
        /

        Returns:
        approaches: list<Approach()>
        """
        approaches = []
        for ILS_approach in self.flight.aligned_on_ils("ZRH"):
            start_timestamp, stop_timestamp = ILS_approach.start,  ILS_approach.stop
            start_index, stop_index = self.timestamps.index(start_timestamp), self.timestamps.index(stop_timestamp)
            
            transfer_dict = {
                "tracks": self.tracks, "longitudes": self.longitudes, "latitudes": self.latitudes, "altitudes":self.altitudes, "Xs": self.Xs, "Ys": self.Ys, "groundspeeds": self.groundspeeds, 
                "timestamps": self.timestamps, "parr_runway_dists": self.parr_runway_dists, "perp_runway_dists": self.perp_runway_dists, 
                "vertical_angles": self.vertical_angles,"average_vertical_angles": self.average_vertical_angles, "long_term_vertical_angles":  self.long_term_vertical_angles,
                }
            
            #Of all the global flight data in transfer_dict, truncate it such that only the parts relevant to the approach are transferred
            for key, value in transfer_dict.items():
                transfer_dict[key] = value[start_index: stop_index] 

            approach = Approach(transfer_dict, self.runway)
            approaches.append(approach)

        return approaches
            
    def get_relative_distances(self):
        """
        Uses runway.get_parametric_distance_from_xxx to calculate the distances from the runway in perpendicular and parallel direction

        Parameters:
        /

        Returns:
        parr_dists:list<float> (parallel distance to the runway)
        perp_dists:list<float> (perpendicular distance to the runway)
        """
        parr_dists = []
        perp_dists = []
        for (x, y) in zip(self.Xs, self.Ys):          
            parr_distance = self.runway.get_parametric_distance_from_threshold(x, y)
            perp_distance = self.runway.get_parametric_distance_from_runway(x, y)
            parr_dists.append(parr_distance)
            perp_dists.append(perp_distance)

        return parr_dists, perp_dists

    def compute_vertical_angles(self):
        """
        Computes the vertical angles, average vertical angles and long_term_vertical_angles of the flight path inbetween all data points

        Parameters:
        /

        Returns:
        vertical_angles:list<float> (raw vertical angles inbetween raw datapoints in Degrees)
        average_vertical_angles:list<float> (rolling average of raw vertical angles in Degrees)
        long_term_vertical_angles:list<float> (long term rolling average of the deviation from the 3 degree glideslope in Degrees)
        """

        #Calculate raw vertical angles
        vertical_angles = []
        for index in range(len(self.altitudes)):
            if index == 0 or index == len(self.altitudes)-1:
                vertical_angle = 0
            else:
                prev_altitude = self.altitudes[index -1]
                prev_y = self.Ys[index- 1]
                prev_x = self.Xs[index - 1]

                next_altitude = self.altitudes[index +1]
                next_y = self.Ys[index + 1]
                next_x = self.Xs[index + 1]

                distance = distance_points([prev_x, prev_y], [next_x, next_y])
                vertical_angle = math.atan2((next_altitude-prev_altitude),distance)*180/math.pi
            vertical_angles.append(vertical_angle)

        vertical_angles = vertical_angles

        #Calculate rolling average of vertical angles
        average_vertical_angles = self.get_average_angles(vertical_angles, 20)

        #Calculate the deviations from the glideslope
        deviation_angles = []
        for vertical_angle in vertical_angles:
            deviation_angle = abs(-self.runway.nom_glideslope-vertical_angle)
            deviation_angles.append(deviation_angle)

        #Calculate the longer-term rolling average of the deviation from the glideslope
        long_term_vertical_angles = self.get_average_angles(deviation_angles, 75)

        return vertical_angles, average_vertical_angles, long_term_vertical_angles

    def get_average_angles(self, angles:list, average_time:float):
        """
        Convert list to rolling averages of certain length in time. Original length of list is kept, the first x number of datapoints become zero

        Parameters:
        angles:list<float> 
        average_time:float (the time over which the roling average needs to be computed in Seconds)

        Returns:
        average_angles:list<float>
        """
        average_angles = []

        #Calculate the number of datapoints to cover average_time
        average_size = int(average_time/self.timestamp_delta)

        #Calculate the average angle over this number of following datapoints for every datapoint
        for index, angle in reversed(list(enumerate(angles))):
            if index <= average_size:
                average_angle = 0
            else:
                average_angle = sum(angles[index - average_size: index])/average_size
            average_angles.append(average_angle)  
        
        return list(reversed(average_angles))

class Approach():
    def __init__(self, transfer_dict: dict, runway: Runway):
        """
        Creates custom approach, this contains the same data values from CustomFlight() but only relevant to the given approach. 
        It further truncates the approach data using a further vertical angle criterion
        After this the standard deviations of all parameters are computed

        Parameters:
        transfer_dict: dict (
                "tracks": self.tracks, "longitudes": self.longitudes, "latitudes": self.latitudes, "altitudes":self.altitudes, 
                "Xs": self.Xs, "Ys": self.Ys, "groundspeeds": self.groundspeeds, 
                "timestamps": self.timestamps, "parr_runway_dists": self.parr_runway_dists, "perp_runway_dists": self.perp_runway_dists, 
                "vertical_angles": self.vertical_angles,"average_vertical_angles": self.average_vertical_angles, "long_term_vertical_angles":  self.long_term_vertical_angles,
                )
        runway: Runway (the runway the approach is flown on)

        Returns:
        /
        """
        self.transfer_dict = transfer_dict
        self.runway = runway

        self.narrow_ILS_approach()

        self.calculate_standard_deviations()

    def narrow_ILS_approach(self):
        """
        Further truncates the approach data using a further vertical angle criterion

        Parameters:
        /

        Returns:
        /
        """
        length = len(self.transfer_dict["average_vertical_angles"])

        #the lists are reversed to consider them starting from the runway threshold back in time
        for index, (average_vertical_angle, long_term_vertical_angles)  in enumerate(list(reversed(list(zip(self.transfer_dict["average_vertical_angles"], self.transfer_dict["long_term_vertical_angles"]))))):
            #Only start considering data more than 100 data points away from the runway TODO: Convert this into a time from runway criterion instead of number of datapoints
            if index > 100:
                #When the first vertically deviating point is found, truncate transfer dict at this point
                if (average_vertical_angle > -2 or average_vertical_angle < -4) and (long_term_vertical_angles > 1):
                    critical_index = index
                    actual_index = length - critical_index - 1

                    #truncate transfer dict at this point, the result is the transfer dict going from the truncated point towards the last datapoint
                    for key, value in self.transfer_dict.items():
                        self.transfer_dict[key] = value[actual_index :] 

                    return

    def calculate_standard_deviations(self):
        self.calculate_glideslope_standard_deviation()
        self.calculate_speed_standard_deviation()
        self.calculate_track_standard_deviation()

    def calculate_glideslope_standard_deviation(self):
        """
        Calculates the standard deviation of the vertical height using the perfect approach as a "mean" 
        (accounting for the average height of the approach, giving room for different touchdown points along the runway)

        Parameters:
        /

        Returns:
        /
        """
        dists =  self.transfer_dict["parr_runway_dists"]
        alts = self.transfer_dict["altitudes"]
        end_dist = dists[-1]

        #Calculate the average deviation from the perfect approach
        y_dists = []
        for dist, alt in zip(dists, alts): 
            distance_from_end = abs(end_dist - dist)
            y_dist = alt - math.tan(math.radians(self.runway.nom_glideslope))*distance_from_end
            y_dists.append(y_dist)
        mean_y = sum(y_dists) / len(y_dists)

        #calculate the difference between the approach and a perfect approach that fits the current approach the best using mean_y
        self.nom_heights = []
        deltas = []
        for dist, alt in zip(dists, alts): 
            distance_from_end = abs(end_dist - dist)
            #mean_y is incorporated to account for different touchdown points along the runway
            nominal_height = mean_y + math.tan(math.radians(self.runway.nom_glideslope))*distance_from_end
            self.nom_heights.append(nominal_height)
            delta = alt - nominal_height
            deltas.append(delta)

        self.glideslope_standard_deviation = calculate_standard_deviation(deltas, custom_mean = 0)

    def calculate_horizontal_standard_deviation(self):
        #TODO
        return

    def calculate_track_standard_deviation(self):
        tracks = self.transfer_dict["tracks"]
        self.track_standard_deviation = calculate_standard_deviation(tracks)

    def calculate_speed_standard_deviation(self):
        speeds = self.transfer_dict["groundspeeds"]
        self.speed_standard_deviation = calculate_standard_deviation(speeds)

def calculate_standard_deviation(dp:list, custom_mean:float = None):
        """
        Calculates the standard deviation using an optional mean

        Parameters:
        dp:list (values to calculate standard deviation of)
        custom_mean:float = None (optional: wether to use a custom mean to calculate the deviation around)

        Returns:
        var: float (standard deviation)
        """
        summation = 0
        
        if custom_mean == None:
            mean = sum(dp) / len(dp)
        else:
            mean = custom_mean

        for x in dp:
            summation += (x - mean)**2
        var = math.sqrt(summation / len(dp))
        return var

def plot_custom_flights(custom_flights: list, runway:str, plot_entire_relative:bool = False, plot_traffic_approach:bool = False):
    """
    Plots all approaches of x number of flights. Can be configured to plot whole flight or approach as given by traffic

    Parameters:
    custom_flights: list<CustomFlight()>
    runway:str (the name of the runway that is approached, "all" is a special where the approaches on multiple runway are plotted)
    plot_entire_relative:bool = False (optional: plot the entire flight relative to the runway)
    plot_traffic_approach:bool = False (optional: lot the approach both as given by the traffic library and computed customly)

    Returns:
    fig: matplotlib.pyplot.fig
    """
    flight_names = []

    fig = plt.figure()

    #https://matplotlib.org/stable/tutorials/intermediate/gridspec.html
    spec = gridspec.GridSpec(ncols=5, nrows=2, figure=fig) 
    ax0 = fig.add_subplot(spec[0,:-1])
    ax1 = fig.add_subplot(spec[1,:-1])
    ax2 = fig.add_subplot(spec[:,-1], projection=EuroPP())

    ax2.add_feature(countries())
    ax2.set_global()
    airports["ZRH"].geoencode(runways=True, labels=True)
    airports["ZRH"].plot(ax2)
    ax2.set_extent((7.8,9.2,47.89,47.05))   

    for custom_flight in custom_flights:

        #Skip flight on runway 14 that breaks the plotting code (see docs)
        if custom_flight.flight.flight_id == "EDW19M_5711":
            continue

        flight_names.append(f"{custom_flight.flight.flight_id} runway {custom_flight.runway.name}")

        #Default plotting method: plot filtered approaches only
        for approach in custom_flight.approaches:
            ax0.plot(approach.transfer_dict["parr_runway_dists"], approach.transfer_dict["altitudes"], picker=True, label=custom_flight.flight.flight_id)
            #ax0.plot(approach.transfer_dict["parr_runway_dists"], approach.nom_heights, picker=True, label=flight.flight_id, color = "black")
            ax1.plot(approach.transfer_dict["perp_runway_dists"], approach.transfer_dict["altitudes"], picker=True, label=custom_flight.flight.flight_id)

        custom_flight.flight.plot(ax2, picker=True, label=custom_flight.flight.flight_id)

        #Plot the entire flight relative to the runway
        if plot_entire_relative: 
            ax0.plot(custom_flight.parr_runway_dists, custom_flight.altitudes, picker=True, label=custom_flight.flight.flight_id)
            ax1.plot(custom_flight.perp_runway_dists, custom_flight.altitudes, picker=True, label=custom_flight.flight.flight_id)
        
        #Plot the approach both as given by the traffic library and computed customly
        if plot_traffic_approach:
            for ILS_flight in custom_flight.flight.aligned_on_ils("ZRH"):
                custom_ILS_flight = Custom_Flight(ILS_flight)
                ax0.plot(custom_ILS_flight.parr_runway_dists, custom_ILS_flight.altitudes, picker=True, label=custom_flight.flight.flight_id, color = "red")
                ILS_flight.plot(ax2, picker=True, label=custom_flight.flight.flight_id, color = "black")

    if runway == "all":
        ax0.title.set_text(f"Side view of the approach on multiple runways ")
        ax1.title.set_text(f"Straight down view of the approach on multiple runways")
    else:
        ax0.title.set_text(f"Side view of the approach on runway {runway}")
        ax1.title.set_text(f"Straight down view of the approach on runway {runway}")

    ax0.set_xlabel("Distance from runway threshold [m]") 
    ax1.set_xlabel("Perpendicular distance from runway centerline [m]") 

    ax0.set_ylabel("Altitude [m]") 
    ax1.set_ylabel("Altitude [m]") 

    ax2.title.set_text("Approach map")

    if len(flight_names) < 7:
        ax0.legend(flight_names, loc = "upper left")

    fig.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick1)

    return fig

def save_fig(fig, sub_directory, filename:str):
    directory = f"{dir_path}/{sub_directory}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig(f"{directory}/{filename}")
    
def process_standard_deviations(
    standard_deviation_collections:list,
    std:float,
    runway_name:str,
    number_of_bars:int = None,
    x_title:str = "standard deviation", y_title:str = "frequency", title:str = "frequency as a function of standard deviation", 
    save_individual_plots:bool = True, perform_individual_plots:bool = True
    ):
    """
    Plots and saves standard deviations as frequency and optionally plots and saves outliers

    Parameters:
    standard_deviation_collections: list<list<float, Custom_flight>> ([[variance, Custom_flight],[]])
    std:float (number of standard deviations that define the border between outliers and normal flights)
    runway_name:str
    number_of_bars:int = None (optional: Number of bars to plot in the frequency plot)
    x_title: string = "standard deviation"
    y_title: string = "frequency"
    title: string = "frequency as a function of standard deviation"
    save_individual_plots = False
    perform_individual_plots = False

    Returns:
    outliers: list<list<float, Custom_flight>> ([[variance, Custom_flight],[]])
    """

    fig, ax = plt.subplots()
    standard_deviations = [standard_deviation_collection[0] for standard_deviation_collection in standard_deviation_collections] #Extract actual variances

    if number_of_bars != None:
        ax.hist(standard_deviations, number_of_bars)
    else:
        ax.hist(standard_deviations)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    save_fig(fig, f"{runway_name}/Figures/{std}/{len(standard_deviations)}", f"{title}.png")

    sorted_deviation_collections = sorted(standard_deviation_collections, key=itemgetter(0))

    #Determine the border of variation
    global_standard_deviation = calculate_standard_deviation(standard_deviations)
    print(f"std: {global_standard_deviation}")
    global_mean = sum(standard_deviations)/len(standard_deviations)
    print(f"mean: {global_mean}")
    outlier_border_deviation = global_mean + global_standard_deviation*std

    critical_index = len(sorted_deviation_collections)
    for index, standard_deviation_collection in enumerate(sorted_deviation_collections):
        standard_deviation = standard_deviation_collection[0]
        if standard_deviation > outlier_border_deviation:
            critical_index = index
            break
    
    outliers = sorted_deviation_collections[critical_index:]
    print(f"{title} number of outliers: {len(outliers)}")
    
    for index, outlier in enumerate(outliers):
        if perform_individual_plots:
            flight_fig = plot_custom_flights([outlier[2]], runway_name)
        if save_individual_plots:
            save_fig(flight_fig, f'{runway_name}/Figures/{std}/{len(standard_deviations)}/{x_title}', f"outlier_{index}.png")

    flights_fig = plot_custom_flights([outlier[2] for outlier in outliers], runway_name)
    save_fig(flights_fig, f'{runway_name}/Figures/{std}/{len(standard_deviations)}/{x_title}', 'all_outliers.png')
        
    return outliers

def process_global_outliers(outliers_collection:dict, custom_flights:list, std:float, runway_name:str):
    """
    Makes a dictionary of the custom flights in the dataset and their respective outliers

    Parameters:
    outliers_collection:dict ({"name of outliers": [outliers], ...}, [outliers] = [variance, approach, custom flight])
    runway_name:str
    std:float (number of standard deviations that define the border between outliers and normal flights)
    custom_flights:list ([custom_flight, ...] list of custom flights in which outliers need to be marked)

    Returns:
    marked_flights:list (unused, contains data written to file)
    """
    marked_flights = {}
    for custom_flight in custom_flights:
        marked_flights[custom_flight.flight.flight_id] = [False for i in range(len(outliers_collection)+1)] #The +1 is to create a column for flights which are outliers in all three categories

    outlier_type = 0
    for outliers_item in outliers_collection.items():
        for outliers in outliers_item[1]:
            outlier_name = outliers[2].flight.flight_id
                
            marked_flights[outlier_name][outlier_type] = True
        outlier_type += 1

    for marked_flight in marked_flights.items():
        if all(marked_flight[1][:-1]):
            marked_flight[1][-1] = True

    L = []
    L_first = "flight_id"
    for outliers_name in outliers_collection.keys():
        L_first += f" {outliers_name}"
    L_first += " all \n" #L_first example: flight_id vertical speed track all
    L.append(L_first) 

    for marked_flight in marked_flights.items():
        marked_flight_name = marked_flight[0]
        marked_flight_data = " ".join(str(boolean) for boolean in marked_flight[1])
        L.append(f"{marked_flight_name} {marked_flight_data} \n")

    save_directory = f"{dir_path}/{runway_name}/outlier files/{std}"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    f = open(f"{save_directory}/outliers_{len(custom_flights)}.txt", "w")
    f.writelines(L)
    return marked_flights

def onpick1(event):
    """
    Matplotlib magic, I have no clue what's going on here. 
    Code inspired by: https://stackoverflow.com/a/22208861, https://stackoverflow.com/a/7909589
    """
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        print('onpick1 line:', thisline.get_label())

def get_arrival_dataset(dir:str):
    return Traffic.from_file(dir)

#This data was manually compiled using Google Maps
runways = [Runway("14", [47.477709, 8.542060], [47.461295, 8.564463], 3),
    Runway("32", [47.461295, 8.564463], [47.477709, 8.542060], 3),
    Runway("28", [47.456679, 8.569400], [47.458914, 8.537914], 3),
    Runway("10", [47.458914, 8.53791], [47.456679, 8.569400], 3),
    Runway("34", [47.445375, 8.556837], [47.475069, 8.536390], 3.3),
    Runway("16", [47.475069, 8.536390], [47.445375, 8.556837], 3)]
dir_path = os.path.dirname(os.path.realpath(__file__))