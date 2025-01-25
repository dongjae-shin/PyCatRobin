import uncertainty_sampling.extract as ex
import os

# Define the home directory and path to data
home_dir = os.path.expanduser("~")
path = home_dir + "/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/02 GC Experimental Data"

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

# Calculate delta CO2 conversion
# dataset.assign_delta_co2_conv()
dataset.assign_target_values(['delta','initial slope'], column='CO2 Conversion (%)', plot_slope=True)
dataset.assign_target_values(['delta','initial value'], column='CO Forward Production Rate (mol/molRh/s)', plot_slope=True)

for i in range(len(dataset.path_filtered)):
    ex.plot_tos_data(
        dataset.path_filtered[i],
        'CO2 Conversion (%)',
        temp_threshold=2.0,
        init_tos_buffer=0.5,
        plot_selected=True, show=True
    )

# # Export the processed data
# print(
#     dataset.export_sheet(
#         unique=True,
#         # which_target='delta_CO2_conv',
#         which_target='CO2 Conversion (%)_delta',
#         mute=True
#     )
# )

print(dataset.df_us.columns[0])