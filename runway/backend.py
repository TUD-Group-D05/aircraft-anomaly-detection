import time
from math import *
from pyproj import Proj, proj

start_time = time.time()
waypoint = start_time

def time_statement(text):
    global waypoint
    print(text, "%s seconds" % (time.time() - waypoint), "Program execution time:", "%s seconds" % (time.time() - start_time))
    waypoint = time.time()
    return

def distance_coordinates_km(first, second): #https://stackoverflow.com/questions/365826/calculate-distance-between-2-gps-coordinates
    """
    first = [x, y] (longitude,latitude)
    second = [x, y] (longitude,latitude)
    """
    R = 6371

    dLat = (second[0]-first[0])*pi/180 
    dLon = (second[1]-first[1])*pi/180

    lat1 = first[0]*pi/180
    lat2 = second[0]*pi/180

    a = sin(dLat/2) * sin(dLat/2) + sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def distance_points(first, second):
    """
    first = [lat, long] (y,x)
    second = [lat, long] (y,x)

    """
    return sqrt((second[0]-first[0])**2+(second[1]-first[1])**2)

def convert_to_XY(longitudes, latitudes):
    conversion = Proj('epsg:21781')
    Xs, Ys = conversion(longitudes, latitudes)
    return Xs, Ys