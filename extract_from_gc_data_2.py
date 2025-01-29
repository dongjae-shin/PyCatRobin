import data.extract as ex
import os

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

# Construct the DataFrame
dataset.construct_dataframe(extensive=False)

# Convert measured values to nominal values
dataset.convert_measured_to_nominal(which_column="Rh_total_mass")

# Apply duplicate group IDs
dataset.apply_duplicate_groupid(verbose=False)

# Calculate and add target values into the DataFrame
for column in ['CO2 Conversion (%)', 'CH4 Net Production Rate (mol/molRh/s)', 'CO Net Production Rate (mol/molRh/s)',
               'CO Forward Production Rate (mol/molRh/s)', 'Selectivity to CO (%)']:
    dataset.assign_target_values(
        methods=['initial slope', 'final slope', 'overall slope'],
        column=column,
        temp_threshold=3.5,
        init_tos_buffer=0.5,
        adjacency_slope=0.5,
    )