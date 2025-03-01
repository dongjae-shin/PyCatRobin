import os

import data.extract as ex
import analysis.data_analysis as da

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
dataset.filter_excel_files(exclude_keywords=exclude_keywords, verbose=True)
# Construct the DataFrame
# dataset.construct_dataframe(extensive=True)
dataset.construct_dataframe(extensive=False)
# Convert measured values to nominal values
dataset.convert_measured_to_nominal(which_column="Rh_total_mass")
# Apply duplicate group IDs
dataset.apply_duplicate_groupid(
    exclude_columns=['filename', 'experiment_date', 'location'],
    verbose=False
)

# Calculate and add target values into the DataFrame
savgol=False

for column in [
   'CO2 Conversion (%)',
   # 'CH4 Net Production Rate (mol/molRh/s)', 'CO Net Production Rate (mol/molRh/s)',
   # 'CO Forward Production Rate (mol/molRh/s)',
   # 'Selectivity to CO (%)'
    ]:
    dataset.assign_target_values(
        methods=[
            'initial value',
            # 'final value',
            # 'initial slope',
            # 'final slope',
            # 'overall slope',
            # 'delta'
        ],
        column=column,
        temp_threshold=3.5,
        init_tos_buffer=0.5,
        adjacency_slope=1.0,
        savgol=savgol
        )

# Construct unique DataFrame using group IDs
dataset.construct_unique_dataframe(verbose=True)
# Calculate statistics DataFrame on the basis of GroupID
dataset.calculate_statistics_duplicate_group(verbose=False)

# Plot the data and the corresponding slopes
dataset.plot_tos_data(column='CO2 Conversion (%)', #'Selectivity to CO (%)',
                      x_max_plot=20, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
                      plot_selected=True, plot_slope=True,
                      methods_slope=['initial slope', 'final slope', 'overall slope'], show=True,
                      savgol=savgol,
                      gui=True)

analysis = da.DataAnalysis(dataset=dataset)
analysis.compare_targets_std_dev(target_wise=True)
# analysis._generate_histogram(column='CO2 Conversion (%)_initial value')

# Export the processed data
dataset.export_sheet(unique=True)
# dataset.export_sheet(unique=False)