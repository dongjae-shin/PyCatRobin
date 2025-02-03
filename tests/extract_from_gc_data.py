import data.extract as ex
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
dataset.construct_dataframe(extensive=False)
# Convert measured values to nominal values
dataset.convert_measured_to_nominal(which_column="Rh_total_mass")
# Apply duplicate group IDs
dataset.apply_duplicate_groupid(verbose=False)

# # Calculate and add target values into the DataFrame
# for column in ['CO2 Conversion (%)', 'CH4 Net Production Rate (mol/molRh/s)', 'CO Net Production Rate (mol/molRh/s)',
#                'CO Forward Production Rate (mol/molRh/s)', 'Selectivity to CO (%)']:
#                dataset.assign_target_values(methods=['initial value', 'final value', 'delta', 'initial slope', 'final slope', 'overall slope'],
#                                             column=column,
#                                             temp_threshold=3.5,
#                                             init_tos_buffer=0.5,
#                                             adjacency_slope=1.0,
#                                             savgol=True
#                                             )

# # Construct unique DataFrame using group IDs
# dataset.construct_unique_dataframe(verbose=True)
# # Export the processed data
# print(dataset.export_sheet(unique=True))

# Plot the data and the corresponding slopes
dataset.plot_tos_data(column='Selectivity to CO (%)', #'CO2 Conversion (%)', #'CH4 Net Production Rate (mol/molRh/s)',
                      x_max_plot=20,
                      temp_threshold=3.5,
                      init_tos_buffer=0.5,
                      adjacency_slope=1.0,
                      savgol=True,
                      plot_selected=True, plot_slope=True,
                      methods_slope=['initial slope', 'final slope', 'overall slope'], show=True, )

# dataset.calculate_statistics_duplicate_group(verbose=False)
#
# # Melt the DataFrame to long format for seaborn
# df_melted = dataset.df_stat.melt(id_vars=['GroupID'],
#                                  value_vars=[col for col in dataset.df_stat.columns if '_std' in col],
#                                  var_name='Target',
#                                  value_name='Standard Deviation')
# df_melted['Target'] = df_melted['Target'].str.rstrip('_std')
#
# # Plot the data
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df_melted, x='Target', y='Standard Deviation', hue='GroupID')
# plt.xticks(rotation=45, ha='right')
# plt.title('Standard Deviation of Target Values by GroupID')
# plt.tight_layout()
# plt.show()

print(dataset.df_us.columns[0])