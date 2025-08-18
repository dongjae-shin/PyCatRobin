import os
import numpy as np
import data.extract as ex
import analysis.data_analysis as da

# Define the home directory and path to data
home_dir = os.path.expanduser("~")
path_temp = (home_dir +
"/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata_temp")
path_Rh_loading = (home_dir +
"/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata_Rh_loading")
path_synth = (home_dir +
"/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata_synth")
path_all = (home_dir +
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata")

exclude_keywords_overall = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location"]
exclude_keywords_ucsb = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location",
                         "_Cargnello", "_SLAC", "_PSU"]

def plot(
        exclude_keywords: list,
        prefix: str,
        average_same_location: bool,
        path=path_all, path_all=path_all,
        snr: bool = True,
        feature_impact: bool = True,
        colors: list = None
):
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
    # Create an instance of DataForGP for 'all data'
    dataset_all = ex.DataForGP(path=path_all)
    dataset_all.find_excel_files()
    dataset_all.filter_excel_files(exclude_keywords=exclude_keywords_all, verbose=True)
    dataset_all.construct_dataframe(extensive=False)
    # dataset.convert_measured_to_nominal(which_column="Rh_total_mass")
    dataset_all.convert_measured_to_nominal(which_column="Rh_total_mass", allowed_values=np.array([0.02])) # for Round Robin data
    dataset_all.apply_duplicate_groupid(
        exclude_columns=['filename', 'experiment_date', 'location'],
        verbose=False
    )

    # Calculate and add target values into the DataFrame
    savgol=False
    methods=[
                'AUC',
                'final value',
                'initial value',
                'final slope',
                'initial slope',
                'overall slope',
            ]
    for column in [
       # 'CO2 Conversion (%)',
       # 'CH4 Net Production Rate (mol/molRh/s)',
       'CO Net Production Rate (mol/molRh/s)',
       # 'Selectivity to CO (%)'
        ]:
        dataset.assign_target_values(
            savgol=savgol, methods=methods,
            column=column, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
            )
        dataset_all.assign_target_values(
            savgol=savgol, methods=methods,
            column=column, temp_threshold=3.5, init_tos_buffer=0.5, adjacency_slope=1.0,
            )

    # Construct unique DataFrame using group IDs
    dataset.construct_unique_dataframe(verbose=False)
    dataset_all.construct_unique_dataframe(verbose=False)

    # Create an instance of DataAnalysis ------
    analysis = da.DataAnalysis(dataset=dataset)
    # Calculate statistics DataFrame on the basis of GroupID; it determines which data to use as entire dataset
    analysis.calculate_statistics_duplicate_group(
        dataset_all=dataset_all,
        total='duplicate',
        verbose=False,
        average_same_location=average_same_location
    )

    # Plot heatmap of SNR values/Standard Deviation
    if snr:
        analysis.plot_heatmap(
            methods=methods, # to show the rows in a defined order
            properties=[
            # 'CH4 Net Production Rate (mol/molRh/s)',
            'CO Net Production Rate (mol/molRh/s)',
            # 'CO2 Conversion (%)',
            # 'Selectivity to CO (%)'
            ],
            which_to_plot='snr',
            snr_type='mu_sigma',
            cmap='Reds',
            vmax=5.3,# vmin=0.0,
            save_fig=True,
            prefix=prefix
            )

    if feature_impact:
        analysis.compare_targets_std_dev(
            target_wise=True,
            # snr_type='range', #'std_dev',
            snr_type='mu_sigma',
            plot_hist=True,  # False,
            save_fig=True,
            prefix=prefix,
            colors=colors
        )

# plot(exclude_keywords_overall, 'overall', average_same_location=False, path=path_all, snr=True, feature_impact=False)
# plot(exclude_keywords_ucsb, 'ucsb', average_same_location=False, path=path_all, snr=True, feature_impact=False)

# Need to internally turn off legend for violin plots
colors = ['#69BADD', '#ED712E', '#721495'] # customized colors for colorblindness
plot(exclude_keywords_overall, 'temp', average_same_location=False, path=path_temp, snr=False, feature_impact=True, colors=colors)
colors = ['#C6BB68', '#2A60DD', '#ED712E']
plot(exclude_keywords_overall, 'Rh_loading', average_same_location=False, path=path_Rh_loading, snr=False, feature_impact=True, colors=colors)
colors = ['#BBBBBB', '#2A60DD']
plot(exclude_keywords_overall, 'synth', average_same_location=False, path=path_synth, snr=False, feature_impact=True, colors=colors)


