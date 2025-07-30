import os
import numpy as np
import data.extract as ex
import analysis.data_analysis as da

# Define the home directory and path to data
home_dir = os.path.expanduser("~")
path = (home_dir +
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata")
path_all = (home_dir +
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/Accelerate/250502_Round_Robin/250505_finalized_RR_data/alldata")

exclude_keywords_overall = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location"]
exclude_keywords_interlab = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location"]
exclude_keywords_ucsb = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location",
                         "_Cargnello", "_SLAC", "_PSU"]
exclude_keywords_stanford = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location",
                         "_UCSB", "_SLAC", "_PSU"]
exclude_keywords_slac = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location",
                         "_UCSB", "_Cargnello", "_PSU"]
exclude_keywords_psu = ["0p0005", "(1)", "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location",
                        "_UCSB", "_Cargnello", "_SLAC"]

def plot_variability(
        exclude_keywords: list,
        prefix: str,
        average_same_location: bool,
        path=path, path_all=path_all
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
                # 'delta'
            ]
    for column in [
       'CO2 Conversion (%)',
       'CH4 Net Production Rate (mol/molRh/s)',
       'CO Net Production Rate (mol/molRh/s)',
       # 'CO Forward Production Rate (mol/molRh/s)',
       'Selectivity to CO (%)'
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
    return analysis.plot_heatmap(
        methods=methods, # to show the rows in a defined order
        properties=[
        'CH4 Net Production Rate (mol/molRh/s)',
        'CO Net Production Rate (mol/molRh/s)',
        'CO2 Conversion (%)',
        'Selectivity to CO (%)'
        ],
        which_to_plot='std_dev_mean_normalized',
        cmap='Blues',
        vmax=5.0,# vmin=0.0,
        save_fig=True,
        prefix=prefix
        )

plot_variability(exclude_keywords_overall, 'overall', average_same_location=False)
plot_variability(exclude_keywords_interlab, 'interlab', average_same_location=True)

df_heatmap_ucsb = plot_variability(exclude_keywords_ucsb, 'ucsb', average_same_location=False)
df_heatmap_stanford = plot_variability(exclude_keywords_stanford, 'stanford', average_same_location=False)
df_heatmap_slac = plot_variability(exclude_keywords_slac, 'slac', average_same_location=False)
df_heatmap_psu = plot_variability(exclude_keywords_psu, 'psu', average_same_location=False)

# mean of the four heatmaps
df_heatmap_avg = (df_heatmap_ucsb + df_heatmap_stanford + df_heatmap_slac + df_heatmap_psu) / 4.0

import matplotlib.pyplot as plt
import seaborn as sns
# Plot the heatmap
fig, ax = plt.subplots(figsize=(10.3, 10))
# Set the font sizes for the plot
label_size = 22; annot_size = 18; plt.rcParams.update({'font.size': label_size})
vmax = 5.0; vmin = None
vmax = df_heatmap_avg.max().max() if vmax is None else vmax
vmin = df_heatmap_avg.min().min() if vmin is None else vmin
which_to_plot = 'std_dev_mean_normalized'
cbar_label =  r'$\overline{\text{Normalized Variability}}$' #'Max{standard deviation${_i}$} / |Mean$_{entire}$|'
cmap = 'Blues'
sns.heatmap(
    df_heatmap_avg,
    annot=True, fmt='.2f',
    annot_kws={'fontsize': annot_size}, # set fontsize for the annotation
    cmap=cmap,
    cbar_kws={'label': cbar_label},
    vmax=vmax, vmin=vmin,
    ax=ax, # use the ax parameter to plot the heatmap on the provided axis
)
# change the position of the xticklabels
ax.xaxis.set_ticks_position("top")
# Rotate xtick labels
plt.xticks(ha='left', fontsize=label_size, rotation=20)
plt.yticks(ha='right', fontsize=label_size, rotation=60)
plt.tight_layout()

# reset the font size
plt.rcParams.update({'font.size': 10})
plt.savefig(f'heatmap_{which_to_plot}_intra_avg.png', dpi=300, bbox_inches='tight')