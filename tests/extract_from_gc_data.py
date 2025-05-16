import os
import numpy as np

import data.extract as ex
import analysis.data_analysis as da

# Define the home directory and path to data
home_dir = os.path.expanduser("~")
path = (home_dir +
        # "/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/04 Catalysis Round Robin/01 Round Robin GC Results")
        # "/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/03 Previous Data/20250404_temporary_data_UCSB+RR")
        # "/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/02 GC Experimental Data")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/250328_RR_data")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/250430_RR_data_local_2")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/250430_RR_data_local_manual_fixed")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/250430_RR_data_local_manual_fixed_Vortex")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/250430_RR_data_local_manual_fixed_Wig-L-Bug")
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/250502_Round_Robin/250505_finalized_RR_data/alldata")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/250502_Round_Robin/250505_finalized_RR_data/alldata_temp")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/250502_Round_Robin/250505_finalized_RR_data/alldata_Rh_loading")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/250502_Round_Robin/250505_finalized_RR_data/alldata_synth")
        # "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/250502_Round_Robin/250505_finalized_RR_data/premix")

# Keywords to exclude
exclude_keywords = [
    "0p0005", # data with too low Rh mass, likely to be inaccurate
    "(1)",    # data mistakenly uploaded twice
    "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location", # example Excel file
    # "_UCSB",  # data from UCSB
    # "_Cargnello", # data from Cargnello
    # "_SLAC",   # data from SLAC
    # "PSU",
]
exclude_keywords_all = [
    "0p0005", # data with too low Rh mass, likely to be inaccurate
    "(1)",    # data mistakenly uploaded twice
    "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location", # example Excel file
]

# Create an instance of DataForGP
dataset = ex.DataForGP(path=path)
dataset.find_excel_files()
dataset.filter_excel_files(exclude_keywords=exclude_keywords, verbose=True)
dataset.construct_dataframe(extensive=False)
# dataset.convert_measured_to_nominal(which_column="Rh_total_mass")
dataset.convert_measured_to_nominal(which_column="Rh_total_mass", allowed_values=np.array([0.02])) # for Round Robin data
dataset.apply_duplicate_groupid(
    exclude_columns=['filename', 'experiment_date', 'location'],
    verbose=False
)

# # Create another DataForGP instance for the data including all the locations
# dataset_all = ex.DataForGP(path=path)
# dataset_all.find_excel_files()
# dataset_all.filter_excel_files(exclude_keywords=exclude_keywords_all, verbose=True)
# dataset_all.construct_dataframe(extensive=False)
# dataset_all.convert_measured_to_nominal(which_column="Rh_total_mass", allowed_values=np.array([0.02]))
# dataset_all.apply_duplicate_groupid(
#     exclude_columns=['filename', 'experiment_date', 'location'],
#     verbose=False
# )

# Calculate and add target values into the DataFrame
savgol=False
methods=[
            'initial value',
            'final value',
            'initial slope',
            'final slope',
            'overall slope',
            # 'delta'
        ]
for column in [
   # 'CO2 Conversion (%)',
   # 'CH4 Net Production Rate (mol/molRh/s)',
   'CO Net Production Rate (mol/molRh/s)',
   # 'CO Forward Production Rate (mol/molRh/s)',
   # 'Selectivity to CO (%)'
    ]:
    dataset.assign_target_values(
        savgol=savgol, methods=methods,
        column=column, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
        )
    # dataset_all.assign_target_values(
    #     savgol=savgol, methods=methods,
    #     column=column, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
    #     )

# Construct unique DataFrame using group IDs
dataset.construct_unique_dataframe(verbose=True)
# dataset_all.construct_unique_dataframe(verbose=True)
# Calculate statistics DataFrame on the basis of GroupID
dataset.calculate_statistics_duplicate_group(verbose=False)
# dataset_all.calculate_statistics_duplicate_group(verbose=False)

# # Plot the data and the corresponding slopes
# dataset.plot_tos_data(column='CO2 Conversion (%)', #'Selectivity to CO (%)',
#                       x_max_plot=20, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
#                       plot_selected=True, plot_slope=True,
#                       methods_slope=['initial slope', 'final slope', 'overall slope'], show=True,
#                       savgol=savgol,
#                       gui=True)

analysis = da.DataAnalysis(
    dataset=dataset,
    # dataset_all=dataset_all
)

# analysis.plot_tos_data_duplicate(column='CO Net Production Rate (mol/molRh/s)', x_max_plot=12)
# analysis.plot_tos_data_duplicate(column='CO Forward Production Rate (mol/molRh/s)')
# analysis.plot_tos_data_duplicate(column='Selectivity to CO (%)', x_max_plot=12)
# analysis.plot_tos_data_duplicate(column='CH4 Net Production Rate (mol/molRh/s)', x_max_plot=12)
# analysis.plot_tos_data_duplicate(column='CO2 Conversion (%)', x_max_plot=12)
# analysis.plot_heatmap_snr(vmax=50, use_dataset_all=True)
# analysis.plot_heatmap_snr(vmax=2.5)

# average_value = analysis.df_snr.mean().mean()
# print(f'Average value of all the values in the DataFrame: {average_value}')

analysis.compare_targets_std_dev(
    target_wise=True, plot_hist=False,
    violinplot_direction='vertical'
)

# analysis._generate_data_distribution_horizontal(
#     column='CO Net Production Rate (mol/molRh/s)_initial value',
#     plot_hist=False
# )

# Export the processed data
# dataset.export_sheet(unique=True)
# dataset.export_sheet(unique=False)