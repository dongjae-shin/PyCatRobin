import data.extract as ex
import os
import matplotlib.pyplot as plt

# Define the home directory and path to data
home_dir = os.path.expanduser("~")
path = (home_dir +
        "/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/02 GC Experimental Data")

# Keywords to exclude
exclude_keywords = [
    "0p0005", # data with too low Rh mass, likely to be inaccurate
    "(1)",    # data mistakenly uploaded twice
    "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location" # example Excel file
]

# Create an instance of DataForGP
dataset = ex.DataForGP(path=path)

# Find and filter Excel files
dataset.find_excel_files()
dataset.filter_excel_files(
    exclude_keywords=exclude_keywords,
    verbose=True
)

# Plot the data and the corresponding slopes
for i in range(30):
    print(i, end=' ')
    ex._plot_tos_data(dataset.path_filtered[i], column='CO2 Conversion (%)', x_max_plot=20,
                      temp_threshold=3.5,
                      init_tos_buffer=0.5,
                      adjacency_slope=1.5,
                      plot_selected=True, plot_slope=True, savgol=True,
                      methods_slope=['initial slope', 'final slope',],# 'overall slope'],
                      show=False, )
    plt.show()
print()