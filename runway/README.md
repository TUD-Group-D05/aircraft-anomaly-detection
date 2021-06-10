# Introduction 
This folder contains both the database of the results and the code that generated it. 

# Structure of the database 
The database is contained in folders "14", "16", "28", "14" and "all". These contain the results per runway where the title of the folder is the runway number. "all" contains the results that were generated when considering all runways at once.

The runways contain two folders: "figures" and "outlier files". The "figures" folder contains all relevant figures and the "outlier files" contains the .txt detailing which flights were considered outliers.

## Figures in the database 
The figures are structured like this:
"Figures"/"Standard deviation"/"Number of flights in the analysis"

Where "Standard deviation" is the standard deviation that was used to find the outliers in the subsequent folders and "Number of flights in the analysis" is the number of flights that were considered to compute the results.

### Example
	14/Figures/2/57
	The outlier figures found by analysing 57 flights with a 2 times standard deviation on runway 14.

## Outliers in the database
The outliers file follows the same name style as the figures

### Example
	14/outlier files/2/outliers_57
	The outliers found by analysing 57 flights with a 2 times standard deviation on runway 14.

# Code 
