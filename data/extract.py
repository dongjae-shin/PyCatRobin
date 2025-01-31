import os
from typing import List, Tuple
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import warnings
from datetime import timedelta
from IPython.display import display

class DataForGP:
    """
    A class to handle and process experimental data for Gaussian Process (GP) modeling.

    Attributes:
        path (str): Path to the directory containing the data files.
        path_found (list): List of paths to the found Excel files.
        path_filtered (list): List of paths to the filtered Excel files.
        path_removed (list): List of paths to the removed Excel files.
        df_us (pd.DataFrame): DataFrame containing the processed data.
        df_us_unique (pd.DataFrame): DataFrame containing the unique processed data.
        df_stat (pd.DataFrame): DataFrame containing statistical data of duplicate groups.
        targets (list): List to store target values.

    Methods:
        find_excel_files():
            Finds all Excel files in the specified directory.

        filter_excel_files(exclude_keywords, verbose=False):
            Filters out Excel files based on the provided keywords.

        construct_dataframe(extensive=False):
            Constructs a DataFrame from the filtered Excel files.

        convert_measured_to_nominal(allowed_values=None, which_column='Rh_total_mass'):
            Converts measured values in the specified column to the closest nominal values.

        check_most_recent(buffer_recent=1, column='CO2 Conversion (%)', method='delta', ...):
            Checks and plots the most recent data based on the specified column and method.

        apply_duplicate_groupid(verbose=False):
            Applies group IDs to duplicate entries in the DataFrame.

        assign_target_values(methods, column, verbose=False, ...):
            Assigns target values to the DataFrame based on the specified methods.

        construct_unique_dataframe(verbose=False):
            Constructs a unique DataFrame by averaging targets and integrating duplicate groups.

        calculate_statistics_duplicate_group():
            Calculates statistics for duplicate groups.

        export_sheet(unique=True):
            Exports the processed data to an Excel sheet.
    """
    def __init__(self, path):
        self.path = path
        self.path_found = None
        self.path_filtered = None
        self.path_removed = None
        self.df_us = None
        self.df_us_unique = None
        self.df_stat = None
        self.targets = []

    def find_excel_files(self):
        extension = '*.xlsx'
        self.path_found = glob(pathname=os.path.join(self.path, extension))
        print(f'{len(self.path_found)} excel files were found:')

    def filter_excel_files(self, exclude_keywords, verbose=False):
                    """
                    Filters out excel files that satisfy any of `exclude_keywords`.

                    Args:
                        exclude_keywords (list): List of keywords to exclude files.
                        verbose (bool): If True, print the number and names of filtered files.

                    Returns:
                        None
                    """
                    self.path_filtered = []
                    self.path_removed = []
                    for path in self.path_found:
                        if any(keyword in path for keyword in exclude_keywords):
                            self.path_removed.append(path)  # store removed elements
                        else:
                            self.path_filtered.append(path)  # store filtered elements

                    if verbose:
                        print(f'{len(self.path_removed)} files were filtered out:')
                        for path in self.path_removed:
                            print(path)

    def construct_dataframe(self, extensive:bool=False):
        if extensive:
            self.df_us = pd.DataFrame(
                {'reaction_temp': [],
                 'Rh_weight_loading': [],
                 'Rh_total_mass': [],
                 'synth_method': [],
                 'filename': [],
                 'experiment_date': [],
                 'diluent_mass': [],
                 'bed_length': [],
                 'inner_diameter': [],
                 'pretreat_ramp_rate': [],
                 'pretreat_gas_composition': [],
                 'diluent': [], }
            )
        else:
            self.df_us = pd.DataFrame(
                {'reaction_temp': [],
                 'Rh_weight_loading': [],
                 'Rh_total_mass': [],
                 'synth_method': [],
                 'filename': [],
                 'experiment_date': []}
            )
        for i in range(len(self.path_filtered)):
            self.df_us.loc[i] = get_input_vector(
                self.path_filtered[i],
                extensive=extensive
            )  # adding a row
            # self.df_us.index = self.df_us.index + 1  # shifting index
            # self.df_us = self.df_us.sort_index()  # sorting by index

        # convert 'experiment_date' column into datetime format
        self.df_us['experiment_date'] = pd.to_datetime(self.df_us['experiment_date'], format='%Y%m%d')

    def convert_measured_to_nominal(self,
                                    allowed_values: np.array = None,
                                    which_column: str = 'Rh_total_mass',
                                    ):
        """
        convert measured values in the column that are not allowed by `allowed_values` into closest values
        to the allowed ones.
        Args:
            allowed_values: numpy array of allowed values specified by `which_column`
            which_column: column of which values are converted into the closest allowed values

        Returns:

        """
        if not allowed_values:
            # allowed grid values for m_rh
            m_rh_max = 0.02
            m_rh_min = 0.005
            m_rh_step = 0.0025
            n_m_rh_grid = int((m_rh_max - m_rh_min) / m_rh_step + 1)
            m_rh = np.linspace(m_rh_min, m_rh_max, n_m_rh_grid)

            # allowed grid values + out-of-range nominal value in training data (0.04)
            allowed_values = np.concatenate((m_rh, [0.04]))
        for i in np.argwhere(_is_not_nominal_value_vectorized(self.df_us[which_column], allowed_values)).reshape(-1):
            i_min = np.argmin(np.abs(allowed_values - self.df_us.loc[i, which_column]))
            print('data indexed', i, 'is not nominal: ', self.df_us.loc[i, which_column], '->', allowed_values[i_min])
            self.df_us.loc[i, which_column] = allowed_values[i_min]

    def check_most_recent(
            self, buffer_recent: int = 1, column: str = 'CO2 Conversion (%)', method: str = 'delta',
            x_max_plot: float | None = None,
            y_max_plot: float | None = None,
            temp_threshold: float = 3.5,
            init_tos_buffer: float = 1.0,
            plot_selected: bool = False,
            plot_slope: bool = False,
            savgol: bool = True,
            temp_max: float = False,
            adjacency_slope: float = 0.1,
            duration: float = 10,
            verbose: bool = False
    ):
        """
        Check and plot the most recent data based on the specified column and method.

        Args:
            buffer_recent (int): Buffer in days to consider for recent data.
            column (str): Column to check and plot.
            method (str): Method to calculate the target value.
            x_max_plot (float | None): Maximum value for the x-axis.
            y_max_plot (float | None): Maximum value for the y-axis.
            temp_threshold (float): Temperature threshold for initial index calculation.
            init_tos_buffer (float): Initial time-on-stream buffer for index calculation.
            plot_selected (bool): Whether to plot the selected region.
            plot_slope (bool): Whether to plot the slope of the target values.
            savgol (bool): Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.
            temp_max (float): Maximum value for the temperature axis.
            adjacency_slope (float): Slope threshold for initial and final slope calculations.
            duration (float): Duration to calculate the final index.
            verbose (bool): If True, print additional information.

        Returns:
            None
        """
        # get the most recent date in the column
        most_recent_date = self.df_us['experiment_date'].max()

        # find the row with the most recent date with buffer 
        most_recent_rows = self.df_us[(most_recent_date - self.df_us['experiment_date']) <= timedelta(buffer_recent)]

        # obtain the row numbers
        row_numbers = most_recent_rows.index.to_list()

        print("Most Recent Date:", most_recent_date)
        print("Most Recent Row Numbers:", row_numbers)
        print("Most Recent Row Data:")
        display(self.df_us.iloc[row_numbers, :])

        # plot recent data
        for i, row_number in enumerate(row_numbers):
            print(f"{i + 1}: Recent Row Plot of CO2 conversion:")
            _plot_tos_data(
                self.path_filtered[row_number], column, x_max_plot, y_max_plot, temp_threshold,
                init_tos_buffer, plot_selected, plot_slope, temp_max=temp_max, savgol=savgol
            )
            print(
                f"{column}_{method}: {_calculate_target(
                    self.path_filtered[row_number], column, method, verbose, adjacency_slope, duration, temp_threshold, 
                    init_tos_buffer, savgol
                ) :5.3f}"
            )

    def apply_duplicate_groupid(self, verbose=False):
        """
        Apply group IDs to duplicate entries in the DataFrame.

        Args:
            verbose (bool): If True, print the DataFrame with Group IDs and duplicate groups.

        Returns:
            None
        """
        # Create a new column 'GroupID' to identify duplicate rows
        # Group duplicate rows and assign a unique group number to each group
        subset_columns = [col for col in self.df_us.columns if col not in ['filename', 'experiment_date']]
        self.df_us["GroupID"] = (self.df_us.loc[self.df_us.duplicated(subset=subset_columns, keep=False)]
                                 .groupby(subset_columns)
                                 .ngroup() + 1)

        # set non-duplicate rows as GroupID=0
        self.df_us["GroupID"] = self.df_us["GroupID"].fillna(0).astype(int)

        if verbose:
            print("\nDataFrame with Group IDs:")
            print(self.df_us)
            # show by group
            groups = [group for _, group in self.df_us.groupby("GroupID") if _]
            print("\nDuplicate Groups:")
            for i, group in enumerate(groups, start=1):
                print(f"Group {i}:")
                print(group)

    def assign_target_values(
            self, methods: List[str], column: str, verbose: bool = False,
            adjacency_slope: float = 0.1,
            duration: float = 10,
            temp_threshold: float = 3.5,
            init_tos_buffer: float = 1.0,
            savgol: bool = True
    ):
        """
        Assign target values to the DataFrame based on the specified methods.

        Args:
            methods (List[str]): List of target methods to calculate and include in the DataFrame.
            column (str): The column to calculate the target values for.
            verbose (bool): If True, print the calculated target values.
            adjacency_slope (float): Slope threshold for initial and final slope calculations.
            duration (float): Duration to calculate the final index.
            temp_threshold (float): Temperature threshold for initial index calculation.
            init_tos_buffer (float): Initial time-on-stream buffer for index calculation.
            savgol: Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.

        Returns:
            None
        """

        for method in methods:
            target_values = []
            for path in self.path_filtered:
                target_value = _calculate_target(path, column, method, verbose, adjacency_slope, duration,
                                                 temp_threshold, init_tos_buffer, savgol)
                target_values.append(target_value)
            self.df_us[f'{column}_{method}'] = target_values
            self.targets.append(f'{column}_{method}')
            
    def construct_unique_dataframe(self, verbose: bool = False):
        if 'GroupID' not in self.df_us.columns:
            raise ValueError("self.df_us does not have 'GroupID' column. Please run apply_duplicate_groupid() first.")

        if len(self.targets) == 0:
            raise ValueError("self.targets is not constructed yet. Please run assign_target_values() first.")

        # calculate each group's average target value
        self.df_us_unique = self.df_us[self.df_us["GroupID"] == 0]
        # ignoring warning
        with warnings.catch_warnings(action="ignore"):
            # convert datetime column, self.df_us_unique['experiment_date'], to string
            self.df_us_unique['experiment_date'] = self.df_us_unique['experiment_date'].dt.strftime('%Y%m%d')
            # for each duplicate group integrate columns such as filename, experiment_date, and targets
            for i, df_group in self.df_us[self.df_us["GroupID"] > 0].groupby("GroupID"):
                if verbose:
                    print(f'Group {i}: ')
                df_integrated = df_group.iloc[0, :]  # use the first row in the group
                # calculate mean and std. dev. of each target for each duplicate group
                for target in self.targets:
                    mean = df_group[target].mean()
                    if verbose:
                        print(f'mean of {target}: {mean:5.2f}')
                    df_integrated.loc[target] = mean
                df_integrated.loc['filename'] = ', '.join(df_group['filename'].to_list())
                df_integrated.loc['experiment_date'] = ', '.join(df_group['experiment_date']
                                                                 .dt.strftime('%Y%m%d').to_list())

                # append an integrated row
                self.df_us_unique.loc[-1] = df_integrated
                self.df_us_unique.index = self.df_us_unique.index + 1
                self.df_us_unique = self.df_us_unique.sort_index()
        self.df_us_unique = self.df_us_unique.sort_index(ascending=False)
        self.df_us_unique = self.df_us_unique.reset_index(drop=True)

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
        if 'GroupID' not in self.df_us.columns:
            raise ValueError("self.df_us does not have 'GroupID' column. Please run apply_duplicate_groupid() first.")

        if self.df_us_unique is None:
            raise ValueError("self.df_us_unique is not constructed yet. Please run construct_unique_dataframe() first.")

        if len(self.targets) == 0:
            raise ValueError("self.targets is not constructed yet. Please run assign_target_values() first.")

        # use fist row as a dummy, and remove columns of self.df_stat corresponding to target values
        self.df_stat = self.df_us.iloc[0, :] #
        self.df_stat = self.df_stat.to_frame().T
        self.df_stat = self.df_stat.drop(self.targets, axis=1)

        # add columns for statistics of target values
        for target in self.targets:
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_mean', None)
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_std', None)
            self.df_stat.insert(len(self.df_stat.columns), f'{target}_std_total', None)

        # ignoring warning
        with warnings.catch_warnings(action="ignore"):
            # for each duplicate group integrate columns such as filename, experiment_date, and targets
            for i, df_group in self.df_us[self.df_us["GroupID"] > 0].groupby("GroupID"):
                if verbose:
                    print(f'Group {i}: ')
                df_integrated = df_group.iloc[0, :].drop(self.targets) # use the first row in the group
                # calculate statistics of each target for each duplicate group
                for target in self.targets:
                    mean = df_group[target].mean()
                    std = df_group[target].std()
                    std_total = self.df_us_unique[target].std()
                    if verbose:
                        print(f'mean of {target}: {mean:5.2f}')
                        print(f'std. dev. of {target}: {std:5.2f}')
                    df_integrated.loc[f'{target}_mean'] = mean
                    df_integrated.loc[f'{target}_std'] = std
                    df_integrated.loc[f'{target}_std_total'] = std_total
                df_integrated.loc['filename'] = ', '.join(df_group['filename'].to_list())
                df_integrated.loc['experiment_date'] = ', '.join(df_group['experiment_date']
                                                                 .dt.strftime('%Y%m%d').to_list())
                # append an integrated row
                self.df_stat.loc[i-1] = df_integrated

    def export_sheet(self, unique: bool = True):
        """
        Export the processed data to an Excel sheet.

        Args:
            unique (bool): If True, export a unique dataset by averaging targets and integrating.

        Returns:
            pd.DataFrame: The exported DataFrame.
        """

        if self.df_us_unique is None:
            raise ValueError("self.df_us_unique is not constructed yet. Please run construct_unique_dataframe() first.")

        dates = str(datetime.date.today()).rsplit('-')

        # export unique data set made by averaging targets and integrating
        if unique:
            self.df_us_unique.to_excel(
                f'./{dates[0] + dates[1] + dates[2]}_sheet_for_ML_unique.xlsx',
                index=False
            )
            return self.df_us_unique
        # export data set with duplicate data
        else:
            self.df_us.to_excel(
                f'./{dates[0] + dates[1] + dates[2]}_sheet_for_ML_duplicate.xlsx',
                index=False
            )
            return self.df_us

    def plot_tos_data(
        self,
        column: str = 'CO2 Conversion (%)',
        x_max_plot: float = None, y_max_plot: float = None,
        temp_threshold: float = 3.5, init_tos_buffer: float = 1.0,
        plot_selected: bool = False, plot_slope: bool = False, methods_slope=None,
        temp_max: float = False, show: bool = True, savefig: str = None,
        duration: float = 10.0, adjacency_slope: float = 0.1, savgol: bool = True
        ):
        """
        Plot Time-on-Stream (TOS) data with specific target and temperature.

        Args:
            column (str): Column name to plot. Defaults to 'CO2 Conversion (%)'.
            x_max_plot (float, optional): Maximum value for the x-axis. Defaults to None.
            y_max_plot (float, optional): Maximum value for the y-axis. Defaults to None.
            temp_threshold (float, optional): Temperature threshold for initial index calculation. Defaults to 3.5.
            init_tos_buffer (float, optional): Initial time-on-stream buffer for index calculation. Defaults to 1.0.
            plot_selected (bool, optional): Whether to plot the selected region. Defaults to False.
            plot_slope (bool, optional): Whether to plot the slope of the target values. Defaults to False.
            methods_slope (list, optional): List of methods to calculate the slope. Defaults to None.
            temp_max (float, optional): Maximum value for the temperature axis. Defaults to False.
            show (bool, optional): Whether to display the plot. Defaults to True.
            savefig (str, optional): Path to save the figure. Defaults to None.
            duration (float, optional): Duration to calculate the final index. Defaults to 10.0.
            adjacency_slope (float, optional): Slope threshold for initial and final slope calculations. Defaults to 0.1.
            savgol(bool, optional): Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.

        Returns:
            None
        """
        # Iterate over each filtered path and plot the TOS data
        for i, path in enumerate(self.path_filtered):
            _plot_tos_data(path, column, x_max_plot, y_max_plot, temp_threshold, init_tos_buffer, plot_selected,
                           plot_slope, methods_slope, temp_max, show, savefig, duration, adjacency_slope, savgol,
                           filename=f'({i + 1}/{len(self.path_filtered)}) ' + path.rsplit('/')[-1])

def _is_not_nominal_value_vectorized(to_be_tested: pd.Series,
                                    allowed_values: np.array) -> pd.Series:
    """
    True if the element of a given DataFrame column is not allowed/
    In other words, True if the elements are not close ot any of allowed values
    """
    return to_be_tested.apply(lambda x: not np.any(np.abs(allowed_values - x) == 0))

def get_input_vector(excel_path: str = None,
                     extensive: bool = False,
                     mute: bool = True) -> List:
    """
    Extract input vector for each excel file (241115)
    Args:
        excel_path: placeholder
        extensive: placeholder
        mute: placeholder

    Returns:

    """
    if excel_path == None:
        print('excel_path should be give.')
        return

    df = pd.read_excel(
        excel_path,
        sheet_name='Constants'
    )

    # including only four experimental variables & meta information (for v1 model)
    temp = float(df[df['Variable'] == 'Reaction Temperature']['Value'].values[0])
    w_rh = float(df[df['Variable'] == 'Weight Loading']['Value'].values[0])
    m_catal = float(df[df['Variable'] == 'Catalyst Mass']['Value'].values[0])
    m_rh = m_catal * w_rh / 100.
    synth_method = df[df['Variable'] == 'Synthesis Method']['Value'].values[0]
    expt_date = df[df['Variable'] == 'Experiment Date']['Value'].values[0]
    filename = excel_path.rsplit('/')[-1]

    if extensive:
        # including other experimental variables (for v2 model)
        diluent_mass = float(df[df['Variable'] == 'Diluent Mass']['Value'].values[0])
        bed_length = float(df[df['Variable'].str.lower() == 'Bed Length'.lower()]['Value'].values[
                               0])  # lower(): ignore case for SLAC output
        inner_diameter = float(df[df['Variable'] == 'Reactor Tube ID']['Value'].values[0])
        pretreat_ramp_rate = float(df[df['Variable'] == 'Pretreatment Ramp Rate']['Value'].values[0])
        pretreat_gas_comp = float(df[df['Variable'] == 'Pretreatment Gas Concentration']['Value'].values[0])
        diluent = df[df['Variable'] == 'Diluent']['Value'].values[0]

    if not mute:
        print('filename :', filename)
        print('temp: ', temp)
        print('w_rh: ', w_rh)
        print('m_rh: ', m_rh)
        print('synth_method: ', synth_method)
        print('expt_date: ', expt_date)

        if extensive:
            print('diluent_mass: ', diluent_mass)
            print('bed_length: ', bed_length)
            print('inner_diameter: ', inner_diameter)
            print('pretreat_ramp_rate: ', pretreat_ramp_rate)
            print('pretreat_gas_comp: ', pretreat_gas_comp)
            print('diluent: ', diluent)
    if extensive:
        return [temp, w_rh, m_rh, synth_method, filename, expt_date,
                diluent_mass, bed_length, inner_diameter, pretreat_ramp_rate, pretreat_gas_comp, diluent]
    else:
        return [temp, w_rh, m_rh, synth_method, filename, expt_date]

def _plot_tos_data(
        path: str, column: str = None,
        x_max_plot: float = None, y_max_plot: float = None,
        temp_threshold: float = 3.5,
        init_tos_buffer: float = 1.0,
        plot_selected: bool = False,
        plot_slope: bool = False,
        methods_slope=None,
        temp_max: float = False,
        show: bool = True,
        savefig: str = None,
        duration: float = 10.0,
        adjacency_slope: float = 0.1,
        savgol: bool = True,
        filename: str = None
):
    """
    Plot Time-on-Stream (TOS) data with specific target and temperature.

    Args:
        path (str): Path to the Excel file containing the data.
        column (str, optional): Column name to plot. Defaults to None.
        x_max_plot (float, optional): Maximum value for the x-axis. Defaults to None.
        y_max_plot (float, optional): Maximum value for the y-axis. Defaults to None.
        temp_threshold (float, optional): Temperature threshold for initial index calculation. Defaults to 3.5.
        init_tos_buffer (float, optional): Initial time-on-stream buffer for index calculation. Defaults to 1.0.
        plot_selected (bool, optional): Whether to plot the selected region. Defaults to False.
        plot_slope (bool, optional): Whether to plot the slope of the target values. Defaults to False.
        methods_slope (list, optional): List of methods to calculate the slope. Defaults to ['delta'].
        temp_max (float, optional): Maximum value for the temperature axis. Defaults to False.
        show (bool, optional): Whether to display the plot. Defaults to True.
        savefig (str, optional): Path to save the figure. Defaults to None.
        duration (float, optional): Duration to calculate the final index. Defaults to 10.0.
        adjacency_slope (float, optional): Slope threshold for initial and final slope calculations. Defaults to 0.1.
        savgol: Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.
        filename (str, optional): Filename to display on the plot. Defaults to None.

    Returns:
        None
    """
    if methods_slope is None:
        methods_slope = ['delta']

    tos, temp, col_val, initial_index, final_index, selected_index = \
        _extract_indices_target(path, column, duration, temp_threshold, init_tos_buffer)

    # Plot
    l1 = plt.scatter(tos, col_val,
                     color=[0.5, 1.0, 0.5, 1.0], s=5, label=column)  # whole profile
    if plot_selected:
        l2 = plt.scatter(tos[selected_index], col_val[selected_index],
                         color='g', s=5, label='selected')  # selected region
        plt.scatter(
            [tos[initial_index], tos[final_index]],
            [col_val[initial_index], col_val[final_index]],
            edgecolors='g', color='y', s=18
        )
        plt.axvline(x=tos[initial_index], color='gray', linestyle='--')

    plt.xlabel('Time on stream (hrs)')
    plt.ylabel(column, c='g')
    if y_max_plot:
        plt.ylim(0, y_max_plot)
    if x_max_plot:
        plt.xlim(0, x_max_plot)

    if filename:
        plt.title(
            f'duration: {tos[final_index] - tos[initial_index]:.2f}\n{filename}')
    else:
        plt.title(
            f'duration: {tos[final_index] - tos[initial_index]:.2f}')

    # plot linear line for slopes
    if plot_slope:
        for method in methods_slope:
            # Get proper indices and values according to 'method'
            _, _, _, initial_index, final_index, _ = \
                _extract_indices_target(path, column, duration, temp_threshold, init_tos_buffer, method,
                                        adjacency_slope)
            # Plot using the extracted indices
            if method in ['overall slope']:
                _plot_linear_line_two_points(tos[initial_index], tos[final_index], col_val[initial_index],
                                             col_val[final_index], show=False)
            if method in ['initial slope', 'final slope']:
                _plot_linear_line_fitting(tos[initial_index], tos[final_index], col_val[initial_index],
                                          col_val[final_index], tos, col_val, savgol, show=False)

    # Plot temperature: secondary axis for temperature
    plt.subplot(111)
    axs_2nd = plt.twinx()

    l3 = axs_2nd.scatter(tos, temp, s=5, color='r', alpha=0.5, zorder=0)
    axs_2nd.set_ylabel('Temperature (C)', color='r')
    if temp_max:
        axs_2nd.set_ylim(0, temp_max)

    # legends & title
    if plot_selected:
        ls = [l1, l2, l3]
    else:
        ls = [l1, l3]
    labs = [l.get_label() for l in ls]
    # plt.legend(ls, labs)#, loc=(0.55,0.73))
    plt.legend(ls, labs, loc='lower left', bbox_to_anchor=(0.58, 0))

    if savefig:
        # plt.tight_layout()
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()

# maybe the implementation for each `method` should be different depending on `column` ...
def _calculate_target(
        path: str, column: str, method:str, verbose: bool = False,
        adjacency_slope: float = 0.1,
        duration: float = 10,
        temp_threshold: float = 3.5,
        init_tos_buffer: float = 1.0,
        savgol: bool = True
)->float:
    """
    General version of target value calculator.

    Args:
        path (str): Path to individual GC Excel file.
        column (str): Column name to calculate the target values for.
        method (str): Method to calculate the target value. Options are 'delta', 'initial value', 'final value', 'initial slope', 'final slope', 'overall slope'.
        verbose (bool): If True, print the calculated target values.
        adjacency_slope (float): Slope threshold for initial and final slope calculations.
        duration (float): Duration to calculate the final index.
        temp_threshold (float): Temperature threshold for initial index calculation.
        init_tos_buffer (float): Initial time-on-stream buffer for index calculation.
        savgol: Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.

    Returns:
        float: Calculated target value.
    """
    df = pd.read_excel(path, sheet_name='Data')
    df = df.fillna(value=0)  # some data set includes nan at the end of a column.
    
    if column not in df.columns:
        raise ValueError(f"Keyword '{column}' is not included in {df.columns.tolist()}")

    # Get proper indices and values according to 'method'
    tos, temp, col_val, initial_index, final_index, selected_index = \
        _extract_indices_target(path, column, duration, temp_threshold, init_tos_buffer, method, adjacency_slope)

    if   method == 'delta':
        target = col_val[final_index] - col_val[initial_index]
    elif method == 'initial value':
        target = col_val[initial_index]
    elif method == 'final value':
        target = col_val[final_index]
    elif method == 'initial slope':
        # calculate slope using linear fitting
        target = _plot_linear_line_fitting(tos[initial_index], tos[final_index], col_val[initial_index],
                                           col_val[final_index], tos, col_val, savgol, plot=False, show=False)
    elif method == 'final slope':
        # calculate slope using linear fitting
        target = _plot_linear_line_fitting(tos[initial_index], tos[final_index], col_val[initial_index],
                                           col_val[final_index], tos, col_val, savgol, plot=False, show=False)
    elif method == 'overall slope':
        target = (col_val[final_index] - col_val[initial_index]) / (tos[final_index] - tos[initial_index])
    # elif method == 'decaying rate':
    #     print('not implemented yet')
    #     return

    if verbose:
        print(f"{column}->{method}: {target:.2f}")

    else:
        return target


def _extract_indices_target(
        path: str, column: str, duration: float = 10.0, temp_threshold: float = 3.5, init_tos_buffer: float = 1.0,
        method: str = 'delta', adjacency_slope: float = 0.1
) -> Tuple[pd.Series, pd.Series, pd.Series, int, int, np.ndarray]:
    """
    Processes a DataFrame to extract specific columns and calculate indices.

    Args:
        path (str): The path to the Excel file.
        column (str): The column name to be processed.
        duration (float): The duration to calculate the final index.
        temp_threshold (float): Temperature threshold for initial index calculation.
        init_tos_buffer (float): Initial time-on-stream buffer for index calculation.
        method (str): Method to calculate the target value.
        adjacency_slope (float): Slope threshold for initial and final slope calculations.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, int, int, np.ndarray]:
            - Time on stream (tos) series.
            - Temperature series.
            - Column values series.
            - Initial index.
            - Final index.
            - Selected index array.
    """
    # Read the Excel file and extract the reference temperature
    df = pd.read_excel(
        path,
        sheet_name='Constants'
    )
    temp_ref = float(df[df['Variable'] == 'Reaction Temperature']['Value'].values[0])

    # Read the Excel file and fill NaN values with 0
    df = pd.read_excel(path, sheet_name='Data').fillna(0)

    # Check if the specified column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Keyword '{column}' is not included in {df.columns.tolist()}")

    # Extract the 'Time', 'Temperature', and specified column values, and sort by 'Time'
    df_sorted = df.sort_values(by=df.filter(like='Time').columns[0])
    tos = df_sorted.filter(like='Time').iloc[:, 0]
    temp = df_sorted.filter(like='Temperature').iloc[:, 0]
    col_val = df_sorted.filter(like=column).iloc[:, 0]

    # Detect and remove duplicates based on 'tos' and 'temp'; sometimes identical pair of data points are included.
    df_unique = pd.DataFrame({
            'Time': tos,
            'Temperature': temp,
            column: col_val
        }).drop_duplicates(subset=['Time', 'Temperature']).reset_index(drop=True)
    tos = df_unique['Time']
    temp = df_unique['Temperature']
    col_val = df_unique[column]

    # Find the initial index where the temperature is close to the reference temperature and time is non-negative
    assert (temp_threshold > 0), "temp_threshold should be positive."
    condition1 = np.abs(temp - temp_ref) <= temp_threshold
    test_val = np.abs(temp - temp_ref)
    condition2 = tos >= 0
    initial_index = np.argwhere(condition1 & condition2).reshape(-1)[0]

    # Adjust the initial index to account for the initial TOS buffer
    initial_index = np.argwhere(tos >= tos[initial_index] + init_tos_buffer).reshape(-1)[0]

    # Find the final index based on the duration from the initial index
    final_index = np.argwhere(tos >= tos[initial_index] + duration).reshape(-1)[0]

    # Find the selected indices within the initial and final index range
    selected_index = np.arange(initial_index, final_index + 1)

    # Modify indices according to the given `method` argument for 'plot_slope'
    if method == 'initial slope':
        # use the same initial_index
        try:
            # choosing final index which is close, in tos, to initial index
            final_index = np.argwhere(tos >= tos[initial_index] + adjacency_slope).reshape(-1)[0]
        except Exception as e:
            print(e, f'has occurred while calculating `final_index` for {method}.')
    elif method == 'final slope':
        # use the same final_index
        try:
            # choosing initial index which is close, in tos, to final index
            initial_index = np.argwhere(tos <= tos[final_index] - adjacency_slope).reshape(-1)[-1]
        except Exception as e:
            print(e, f'has occurred while calculating `initial_index` for {method}.')

    return tos, temp, col_val, initial_index, final_index, selected_index

def _plot_linear_line_two_points(
        t_init: float, t_final: float, y_init: float, y_final: float, plot: bool = True, show: bool = False
) -> float:
    """ plot linear line connecting two points

    Args:
        t_init (float): Initial time.
        t_final (float): Final time.
        y_init (float): Initial value.
        y_final (float): Final value.
        plot (bool): Whether to plot the fitted line.
        show (bool): Whether to display the plot.

    Returns:
        float: Slope of the fitted line
    """
    def linear_func(x, x1, x2, y1, y2):
        a = (y2 - y1) / (x2 - x1)
        b = y2 - a * x2
        return a * x + b, a

    x_plot = np.linspace(t_init-1, t_final+1, 100) # plot buffer of 1
    y_plot, slope = linear_func(x_plot,
                                t_init,
                                t_final,
                                y_init,
                                y_final)

    if plot:
        plt.plot(x_plot, y_plot, c='k', alpha=0.5, label='two-point linear')
        plt.scatter([t_init, t_final],
                    [y_init, y_final],
                    color='orange', edgecolors='gray'
                    )
        plt.text(
            (t_init + t_final) / 2,
            (y_init + y_final) / 2,
            f'slope={slope:.2f}'
        )
    if show:
        plt.show()
    return slope

def _plot_linear_line_fitting(
        t_init, t_final, y_init, y_final, tos, col_val, savgol: bool = True, plot: bool=True, show: bool=False
) -> float:
    """
    Plot linear line fitting data points around t_init.

    Args:
        t_init (float): Initial time.
        t_final (float): Final time.
        y_init (float): Initial value.
        y_final (float): Final value.
        tos (pd.Series): Time on stream series.
        col_val (pd.Series): Column values series.
        savgol: Whether to apply Savitzky-Golay filter to the data when calculating initial and final slopes. Defaults to True.
        plot (bool): Whether to plot the fitted line.
        show (bool): Whether to display the plot.

    Returns:
        float: Slope of the fitted line
    """
    # Find the index of t_init
    t_init_index = np.argmin(np.abs(tos - t_init))
    t_final_index = np.argmin(np.abs(tos - t_final))

    # Select data points around t_init for fitting
    start_index = max(0, t_init_index)
    end_index = min(len(tos), t_final_index + 1)
    t_fit = tos[start_index:end_index]
    y_fit = col_val[start_index:end_index]

    if savgol:
        # Apply Savitzky-Golay filter to smooth the data
        from scipy.signal import savgol_filter
        y_fit = savgol_filter(y_fit, window_length=min(len(y_fit),10), polyorder=1)
    if plot and savgol:
        # Plot Savitzky-Golay-filtered data points used for fitting
        plt.scatter(t_fit, y_fit, c='blue', s=5, label='savgol')

    # Perform linear fitting
    coeffs = np.polyfit(t_fit, y_fit, 1)
    linear_func = np.poly1d(coeffs)

    if plot:
        # Generate points for plotting the fitted line
        x_plot = np.linspace(t_init - 1, t_final + 1, 100)
        y_plot = linear_func(x_plot)

        # Plot the fitted line
        plt.plot(x_plot, y_plot, c='b', alpha=0.5, label='linear fit')
        # Plot the initial and final points that define data range for fitting
        plt.scatter([t_init, t_final], [y_init, y_final], color='orange', edgecolors='gray')
        # Annotate the slope value
        plt.text(t_final, y_final, f'slope={coeffs[0]:.2f}')

    if show:
        plt.show()
    return coeffs[0]