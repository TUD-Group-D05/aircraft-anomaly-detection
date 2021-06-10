from traffic.core import Traffic
from statistics import mean, stdev

arrival_dataset = Traffic.from_file("dataset/filtered/filtered_arrival_dataset.pkl")

data_struct = {}
for flight in arrival_dataset:
    runway = flight.data["runway"].iloc[0]
    count = len(flight)
    flight_id = flight.flight_id

    if str(runway) in data_struct.keys():
        data_struct[str(runway)].append([flight_id, count])
    else:
        data_struct[str(runway)] = [[flight_id, count]]

#print(data_struct)


def zvalue(x, mu, sigma):
    return (x-mu) / sigma

def outlier(lst):
    outl = []
    num_points = [i[1] for i in lst]
    mn = mean(num_points)
    ssd = stdev(num_points)
    for flt in lst:
        z_i = zvalue(flt[1], mn, ssd)
        if z_i > 2:
            outl.append(flt)
    return outl

out_rw14 = outlier(data_struct["14"])
out_rw28 = outlier(data_struct["28"])
out_rw34 = outlier(data_struct["34"])
#sort
def indextwo(elem):
    return elem[1]

def sortlist(lst):
    return lst.sort(key=indextwo, reverse=True)

sortlist(out_rw34)

f = open(r"dataset/outliers34.txt","r+")

print("flight_id    count")
for flight in out_rw34:
    print(flight[0], flight[1])
    f.write(f"{flight[0]}, {flight[1]} \n")

print(len(out_rw34))
print(len(data_struct["34"]))

f.close()

#print(sorted_rw14)
#print(out_rw28)