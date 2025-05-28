import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import seaborn as sns
import numpy as np
import warnings

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.pyplot import legend
from matplotlib.widgets import Button
from openpyxl.styles.builtins import styles
from statsmodels.formula.api import nominal_gee

from data.extract import DataForGP, _plot_tos_data, _extract_indices_target


class DataAnalysis:

    # Global dictionary for the location
    location_dict = {'UCSB': 0, 'Cargnello': 1, 'SLAC': 2, 'PSU': 3}

    def __init__(self, dataset: DataForGP = None, dataset_all: DataForGP = None):
        """
        Initialize the DataAnalysis class.
        Args:
            dataset: dataset to be analyzed by standard deviation, signal-to-noise ratio (SNR), etc.
            dataset_all: use this to compare the whole dataset's standard deviation with the one of each subset (w.r.t. location) in violinplot, SNR, barplot, etc.
        """
        if dataset is None:
            raise ValueError("Please provide a dataset.")

        self.dataset = dataset
        self.dataset_all = dataset_all
        self.df_stat = None
        self.unique_properties = None
        self.df_snr = None
        self.df_violinplot = None

    def calculate_statistics_duplicate_group(self, verbose: bool = False):
        """
        Calculate statistics for duplicate groups. The statistics include the mean, standard deviation of each target, and
        the total standard deviation of each target in the unique dataset. The calculated statistics are stored in the
        DataFrame `self.df_stat`. The DataFrame `self.df_stat` is constructed by integrating the columns of the first
        row in each duplicate group. The columns of the DataFrame `self.df_stat` are the same as the columns of the
        DataFrame `self.df_us` except for the target values.

        Args:
            verbose (bool): If True, print the calculated statistics.

        Returns:
            None
        """
        if 'GroupID' not in self.dataset.df_us.columns:
            raise ValueError("self.dataset.df_us does not have 'GroupID' column. Please run apply_duplicate_groupid() first.")

        if self.dataset.df_us_unique is None:
            raise ValueError("self.dataset.df_us_unique is not constructed yet. Please run construct_unique_dataframe() first.")

        if len(self.dataset.targets) == 0:
            raise ValueError("self.dataset.targets is not constructed yet. Please run assign_target_values() first.")

        # use fist row as a dummy, and remove columns of self.df_stat corresponding to target values
        self.df_stat = self.dataset.df_us.iloc[0, :] #
        self.df_stat = self.df_stat.to_frame().T
        self.df_stat = self.df_stat.drop(self.dataset.targets, axis=1)

        # add columns for statistics of target values
        for target in self.dataset.targets:
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_mean', None)
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_std', None)
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_list', None)
        # add a column for dataframe for each duplicate group
        self.df_stat.insert(len(self.df_stat.columns), 'dataframe', None)

        # ignoring warning
        with warnings.catch_warnings(action="ignore"):
            # for each duplicate group integrate columns such as filename, experiment_date, and targets
            for i, df_group in self.dataset.df_us[self.dataset.df_us["GroupID"] > 0].groupby("GroupID"):
                if verbose:
                    print(f'Group {i}: ')
                df_integrated = df_group.iloc[0, :].drop(self.dataset.targets) # use the first row in the group
                # calculate statistics of each target for each duplicate group
                for target in self.dataset.targets:
                    mean = df_group[target].mean()
                    std = df_group[target].std()
                    if verbose:
                        print(f'mean of {target}: {mean:5.2f}')
                        print(f'std. dev. of {target}: {std:5.2f}')
                    df_integrated.loc[f'{target}_mean'] = mean
                    df_integrated.loc[f'{target}_std'] = std
                    df_integrated.loc[f'{target}_list'] = df_group[target].to_list()
                df_integrated.loc['filename'] = df_group['filename'].to_list()
                df_integrated.loc['experiment_date'] = df_group['experiment_date'].dt.strftime('%Y%m%d').to_list()
                df_integrated.loc['location'] = df_group['location'].to_list()
                df_integrated.loc['dataframe'] = df_group

                # append an integrated row
                self.df_stat.loc[i-1] = df_integrated

            # add a row for total unique data
            df_integrated[:] = None # dummy
            # calculate statistics of each target for each duplicate group
            for target in self.dataset.targets:
                std = self.dataset.df_us_unique[target].std()
                if verbose:
                    print(f'total std. dev. of {target}: {std:5.2f}')
                df_integrated.loc[f'{target}_std'] = std
                df_integrated.loc['GroupID'] = 'total' # unique
                df_integrated.loc[f'{target}_list'] = self.dataset.df_us_unique[target].to_list()
            # append an integrated row
            self.df_stat.loc[self.df_stat.shape[0]] = df_integrated

    def compare_targets_std_dev(
            self, target_wise: bool = False, colormap: str = 'tab10',
            plot_hist: bool = True, violinplot_direction: str = 'vertical'
    ):
        """
        Compare the standard deviation of the target values for each column.

        Args:
            violinplot_direction (str): The direction of the violin plot. Either 'horizontal' or 'vertical'.
            target_wise (bool): If True, compare standard deviations target-wise; otherwise, compare overall.
            colormap (str): The colormap to use for the plot.
            plot_hist: If True, plot the histogram of the target values.
            violinplot_direction (str): The direction of the violin plot. Either 'horizontal' or 'vertical'.

        Returns:
            None
        """
        if target_wise:
            columns = [col for col in self.df_stat.columns if '_std' in col]

            # extract the properties from the columns
            properties = [] # e.g., 'CO2 Conversion (%)', 'Selectivity to CO (%)'
            for col in columns:
                parts = col.rsplit('_', 2)
                if len(parts) > 2:
                    properties.append(parts[0])
            self.unique_properties = list(set(properties))

            # Plot the data
            for property in self.unique_properties: # Figures over properties
                columns_property = [col for col in columns if property in col]# Figures over properties
                # Create subplots
                nrows = (len(columns_property) + 2) // 3
                ncols = min(len(columns_property), 3)
                fig, axs = plt.subplots(nrows, ncols, figsize=(13, 8))

                # set title of the fig
                fig.suptitle(f'{property}', fontsize=16)

                axs = np.ravel(axs)
                n_plot = 0
                for i, column in enumerate(columns_property): # Subplots over methods
                    # Melt the DataFrame to long format for seaborn
                    df_melted = self.df_stat.melt(id_vars=['GroupID'],
                                                          value_vars=column,
                                                          var_name='Target',
                                                          value_name='Standard Deviation')
                    # Generate colors from a predefined colormap
                    cmap = plt.cm.get_cmap(colormap, len(df_melted['GroupID'].unique()))
                    colors = [cmap(i) for i in range(len(df_melted['GroupID'].unique()))]
                    colors[-1] = 'black'
                    # Plot the data
                    sns.barplot(data=df_melted, x='GroupID', y='Standard Deviation', ax=axs[i], palette=colors,
                                hue='GroupID', legend=False)
                    # Set the title of the subplot with the signal-to-noise ratio (SNR)
                    snr = float(
                        df_melted.loc[df_melted['GroupID'] == 'total', 'Standard Deviation'] / \
                        df_melted.loc[df_melted['GroupID'] != 'total', 'Standard Deviation'].max()
                    )

                    axs[i].set_title(f'{column.split("_")[1]} (SNR={snr:.2f})', fontsize=12)

                    # Show ylabel only for the leftmost axes, and  xlabel only for the lowest axes
                    if i % ncols != 0:
                        axs[i].set_ylabel('')
                    if i < (nrows - 1) * ncols:
                        axs[i].set_xlabel('')
                    n_plot += 1

                    # Make the axes itself a button
                    def on_click(event, col=column, ax=axs[i]):
                        if event.inaxes == ax:
                            self._generate_data_distribution(column=col.rstrip("_std"), cmap=colors,
                                                             plot_hist=plot_hist,
                                                             violinplot_direction=violinplot_direction)
                    fig.canvas.mpl_connect('button_press_event', on_click)

                # Hide the blank Axes: turn off Axes.axis if Axes order > number of plotted Axes
                for i in range(len(axs)):
                    if i + 1 > n_plot:
                        axs[i].axis('off')

                # Show a set of subplots for every property
                fig.tight_layout()
                plt.show()
        else:
            # Melt the DataFrame to long format for seaborn
            df_melted = self.df_stat.melt(id_vars=['GroupID'],
                                             value_vars=[col for col in self.df_stat.columns if '_std' in col],
                                             var_name='Target',
                                             value_name='Standard Deviation')
            df_melted['Target'] = df_melted['Target'].str.rstrip('_std')

            # Plot the data
            plt.figure(figsize=(12, 6))
            sns.barplot(data=df_melted, x='Target', y='Standard Deviation', hue='GroupID', palette=colormap)
            plt.xticks(rotation=45, ha='right')
            plt.title('Standard Deviation of Target Values by GroupID')
            plt.tight_layout()
            plt.show()

    def _generate_data_distribution(
            self, column: str, cmap: str = 'tab10', plot_hist: bool = True, violinplot_direction: str = 'vertical'
    ):

        # shallow copy: connected to the original df. it's like using nickname
        df_stat = self.df_stat.copy().reset_index(drop=True)
        i=1
        # Initialize the DataFrame with the first group's data
        df = pd.DataFrame(
            {'filename'       : df_stat[df_stat['GroupID'] == i]['filename'][i-1],
             'experiment_date': df_stat[df_stat['GroupID'] == i]['experiment_date'][i-1],
             'GroupID'        : [i] * len(df_stat[df_stat['GroupID'] == i][f'{column}_list'][i-1]),
             'location'       : df_stat[df_stat['GroupID'] == i]['location'][i-1],
             f'{column}'      : df_stat[df_stat['GroupID'] == i][f'{column}_list'][i-1]}
        ) # 'GroupID' -> len(): to extract number of groups + total
        # Concatenate other groups' data to the DataFrame in axis=0
        for i, group in enumerate(df_stat['GroupID'].unique()[1:]): # slicing: to exclude the group 'total'
            df = pd.concat((
                df,
                pd.DataFrame(
                {'filename'       : df_stat[df_stat['GroupID'] == group]['filename'][i+1],
                 'experiment_date': df_stat[df_stat['GroupID'] == group]['experiment_date'][i+1],
                 'GroupID'        : [group] * len(df_stat[df_stat['GroupID'] == group][f'{column}_list'][i+1]),
                 'location'       : df_stat[df_stat['GroupID'] == group]['location'][i+1],
                 f'{column}'      : df_stat[df_stat['GroupID'] == group][f'{column}_list'][i+1]}
                )
            ), axis=0)
        df.reset_index(drop=True, inplace=True)
        # set values of column, of which GroupID is 'total', to 'all'
        df.loc[df['GroupID'] == 'total', 'location'] = 'all'
        # Save the DataFrame for later use
        self.df_violinplot = df

        # Plot a violinplot
        if violinplot_direction == 'horizontal':
            fig, axs = plt.subplots(nrows=2 if plot_hist else 1, ncols=1, sharex=True)
            if not plot_hist:
                axs = [axs]

            # hue_order = df['GroupID'].unique()

            sns.violinplot(
                df, x=column, y='GroupID',
                hue='GroupID',
                split=False, inner='stick',
                palette=cmap,
                ax=axs[0],
                legend=False,
                # hue_order=hue_order,
                zorder=0
            )

            # Scatterplot instead of stripplot was used to give different markers to different locations
            # the style argument to differentiate the locations is not supported in stripplot
            # Step 1: Map each unique GroupID to a numeric position
            groupid_labels = df['GroupID'].astype(str).unique()
            groupid_to_num = {label: i for i, label in enumerate(groupid_labels)}
            df['GroupID_num'] = df['GroupID'].astype(str).map(groupid_to_num)
            # Step 2: Add jitter
            df['GroupID_jitter'] = df['GroupID_num'] + np.random.uniform(-0.05, 0.05, size=len(df))

            sns.scatterplot(
                df, x=column, y='GroupID_jitter',
                hue='location',
                palette=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'],
                ax=axs[0], legend=True,
                style='location', edgecolor='w', s=30,
                # markers=['X', 'o', 'P', '^', '*', 'v', 'D', 'P'],
                # hue_order=hue_order,
                zorder=2
            )
            axs[0].set_yticks(list(groupid_to_num.values()))
            axs[0].set_yticklabels(list(groupid_to_num.keys()))

            if plot_hist:
                # Plot the histogram
                sns.histplot(
                    df, x=column, hue='GroupID', shrink=0.95, multiple='stack', stat='count', palette=cmap,
                    kde=False, ax=axs[1]
                )
                # Make the boundary of axs[1] the same as that of axs[0]
                axs[1].set_xlim(axs[0].get_xlim())
                # Make the y tick labels integer
                axs[1].set_yticks(np.arange(0, axs[1].get_ylim()[1], 2))
                axs[0].set_xlabel('')

            fig.suptitle(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()

        elif violinplot_direction == 'vertical':
            fig, axs = plt.subplots(nrows=1, ncols=2 if plot_hist else 1, sharey=True, figsize=(10, 6))
            if not plot_hist:
                axs = [axs]

            # hue_order = df['GroupID'].unique()

            sns.violinplot(
                df, x='GroupID', y=column,
                hue='GroupID',
                split=False, inner='stick',
                palette=cmap,
                ax=axs[0],
                legend=False,
                # hue_order=hue_order,
                zorder=0
            )

            # Scatterplot instead of stripplot was used to give different markers to different locations
            # the style argument to differentiate the locations is not supported in stripplot
            # Step 1: Map each unique GroupID to a numeric position
            groupid_labels = df['GroupID'].astype(str).unique()
            groupid_to_num = {label: i for i, label in enumerate(groupid_labels)}
            df['GroupID_num'] = df['GroupID'].astype(str).map(groupid_to_num)
            # Step 2: Add jitter
            df['GroupID_jitter'] = df['GroupID_num'] + np.random.uniform(-0.05, 0.05, size=len(df))

            sns.scatterplot(
                df, x='GroupID_jitter', y=column,
                hue='location',
                palette=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'],
                ax=axs[0], legend=True,
                style='location', edgecolor='w', s=30,
                # markers=['X', 'o', 'P', '^', '*', 'v', 'D', 'P'],
                # hue_order=hue_order,
                zorder=2
            )
            axs[0].set_xticks(list(groupid_to_num.values()))
            axs[0].set_xticklabels(list(groupid_to_num.keys()))

            if plot_hist:
                # Plot the histogram
                sns.histplot(
                    df, y=column, hue='GroupID', shrink=0.95, multiple='stack', stat='count', palette=cmap,
                    kde=False, ax=axs[1]
                )
                # Make the boundary of axs[1] the same as that of axs[0]
                axs[1].set_ylim(axs[0].get_ylim())
                # Make the x tick labels integer
                axs[1].set_xticks(np.arange(0, axs[1].get_xlim()[1], 2))
                axs[1].set_ylabel('')

            fig.suptitle(f'Distribution of {column}')
            plt.tight_layout()
            plt.show()

    def plot_heatmap_snr(
            self,
            properties: list[str] = None, methods: list[str] = None,
            vmax: float = None, vmin: float = 0, cmap: str = 'Reds',
            use_dataset_all: bool = False
    ):
        """
        Plot the heatmap of the signal-to-noise ratio (SNR) of the target values.

        Args:
            use_dataset_all: whether to use the dataset_all to calculate std. dev. as the nominator of SNR. When DataAnalysis instance does not have dataset_all, it is automatically set to False.
            properties (list(str)): The list of properties to plot.
            methods (list(str)): The list of methods to plot.
            vmax (float): The maximum value of the colorbar.
            vmin (float): The minimum value of the colorbar.
            cmap (str): The colormap to use for the heatmap.
            use_dataset_all (bool): If True, additionally use the dataset_all; otherwise, use only the dataset.

        Returns:
            None
        """
        # If properties and methods are not provided, use the unique properties
        if properties is None:
            if self.unique_properties is None:
                columns = [col for col in self.df_stat.columns if '_std' in col]
                # extract the properties from the columns
                properties = []  # e.g., 'CO2 Conversion (%)', 'Selectivity to CO (%)'
                for col in columns:
                    parts = col.rsplit('_', 2)
                    if len(parts) > 2:
                        properties.append(parts[0])
                self.unique_properties = list(set(properties))
            properties = self.unique_properties

        # If methods are not provided, use the unique methods
        if methods is None:
            methods = [col.split('_')[1] for col in self.df_stat.columns if '_std' in col]
            # make elements of methods unique
            methods = list(set(methods))

        # Make (len(properties) x (len(methods)) DataFrame consists of corresponding SNR values
        # Make the row index corresponds to methods and the column index corresponds to properties
        df_snr = pd.DataFrame(index=methods, columns=properties)

        # Sort index and column names of df_snr so the axes of the heatmap are shown in a consistent order
        df_snr.sort_index(axis=0, inplace=True)
        df_snr.sort_index(axis=1, inplace=True)

        if self.dataset_all is None:
            use_dataset_all = False
            print('DataAnalysis instance does not have dataset_all. Original definition of signal-to-noise ratio is used.')

        for prop in properties:
            for method in methods:
                column = f'{prop}_{method}_std'
                if use_dataset_all:
                    snr = self.dataset_all.df_stat[column].iloc[-1] / self.df_stat[column].iloc[:-1].max()
                else:
                    snr = self.df_stat[column].iloc[-1] / self.df_stat[column].iloc[:-1].max()
                df_snr.loc[method, prop] = snr
        self.df_snr = df_snr

        # Ensure the DataFrame is filled with float values
        df_snr = df_snr.astype(float)

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10.3, 10))
        # Set the font sizes for the plot
        label_size = 22
        annot_size = 18
        plt.rcParams.update({'font.size': label_size})
        vmax = df_snr.max().max() if vmax is None else vmax
        sns.heatmap(
            df_snr,
            annot=True, fmt='.2f',
            annot_kws={'fontsize': annot_size}, # set fontsize for the annotation
            cmap=cmap,
            cbar_kws={'label': 'Signal-to-Noise Ratio (SNR)'},
            vmax=vmax, vmin=vmin,
            ax=ax, # use the ax parameter to plot the heatmap on the provided axis
        )
        # Rotate xtick labels
        plt.xticks(ha='left', fontsize=label_size, rotation=-30)
        plt.yticks(ha='right', fontsize=label_size, rotation=30)
        plt.tight_layout()
        plt.show()

        # reset the font size
        plt.rcParams.update({'font.size': 10})

    def plot_tos_data_duplicate(self,
                                column: str = 'CO Forward Production Rate (mol/molRh/s)',
                                cmap_location: str = 'Set1',
                                location_dict_auto: bool = False,
                                x_max_plot: float = None,
                                y_max_plot: float = None,
                                ):
        group_ids = list(set(self.dataset.df_us['GroupID'][self.dataset.df_us['GroupID'] > 0]))
        # If True, automatically make Dictionary that connects the location to unique integer values
        # If False, use the predefined location_dict
        if location_dict_auto:
            location_dict = {loc: i for i, loc in enumerate(self.dataset.df_us['location'].unique())}
        else:
            location_dict = self.location_dict
        # Extract discrete colors corresponding to the number of locations from the colormap, cmap_location
        cmap = plt.cm.get_cmap(cmap_location, len(location_dict))

        for group_id in group_ids:
            file_names = self.df_stat[self.df_stat['GroupID'] == group_id]['filename'].tolist()[0]
            locations = self.df_stat[self.df_stat['GroupID'] == group_id]['location'].tolist()[0]
            reaction_temp = self.df_stat[self.df_stat['GroupID'] == group_id]['reaction_temp'].values[0]
            w_Rh = self.df_stat[self.df_stat['GroupID'] == group_id]['Rh_weight_loading'].values[0]
            m_Rh = self.df_stat[self.df_stat['GroupID'] == group_id]['Rh_total_mass'].values[0]
            synth_method = self.df_stat[self.df_stat['GroupID'] == group_id]['synth_method'].values[0]

            # Group data by location
            location_data = {}
            for i, file_name in enumerate(file_names):
                path = [s for s in self.dataset.path_filtered if file_name in s][0]
                temp_threshold = 3.5
                init_tos_buffer = 0.5
                duration = 10
                tos, temp, col_val, initial_index, final_index, selected_index = \
                    _extract_indices_target(path, column, duration, temp_threshold, init_tos_buffer)
                if locations[i] not in location_data:
                    location_data[locations[i]] = {'tos': [], 'col_val': [], 'count': 0}
                location_data[locations[i]]['tos'].extend(tos)
                location_data[locations[i]]['col_val'].extend(col_val)
                location_data[locations[i]]['count'] += 1

            # Plot data for each location
            for location, data in location_data.items():
                plt.scatter(
                    data['tos'], data['col_val'],
                    s=10,
                    edgecolors='gray',
                    linewidths=0.2,
                    color=cmap(location_dict[location]),
                    label=f'{location} ({data["count"]})'
                )
            if x_max_plot:
                plt.xlim(0, x_max_plot)
            if y_max_plot:
                plt.ylim(0, y_max_plot)
            plt.xlabel('Time on stream (hrs)')
            plt.ylabel(column)
            plt.title(f'{reaction_temp} C, {w_Rh} wt%, {m_Rh} mg, "{synth_method}" (GroupID={group_id})')
            plt.legend()
            plt.show()


    def pearson_correlation_target(self):
        """
        Calculate the Pearson correlation coefficient between the target values.

        Returns:
            None
        """

        pass
