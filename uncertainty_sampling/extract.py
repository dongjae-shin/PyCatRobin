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
    """"""
    def __init__(self, path):
        self.path = path
        self.path_found = None
        self.path_filtered = None
        self.path_removed = None
        self.df_us = None
        self.co2_convs = []

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

    def check_most_recent(self, buffer_recent: int = 1):
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
            plot_tos_data(
                self.path_filtered[row_number], column='CO2 Conversion', x_max_plot=None, y_max_plot=60
            )
            print(
                f"delta_CO2_conversion (%): {calculate_delta_co2_conv(self.path_filtered[row_number], percent=False, verbose=False) :5.3f}")

    def apply_duplicate_groupid(self, verbose=False):
        """
        Apply group IDs to duplicate entries in the DataFrame.

        Args:
            verbose (bool): If True, print the DataFrame with Group IDs and duplicate groups.

        Returns:
            None
        """
        # Drop the 'filename' and 'experiment_date' columns from the DataFrame
        self.df_us = self.df_us.drop(labels=['filename', 'experiment_date'], axis=1)

        # Create a new column 'GroupID' to identify duplicate rows
        # Group duplicate rows and assign a unique group number to each group
        self.df_us["GroupID"] = (self.df_us.loc[self.df_us.duplicated(keep=False)]
                                    .groupby(self.df_us.columns.tolist())
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

    def assign_delta_co2_conv(self, percent=False, verbose=False):
        for i in range(len(self.path_filtered)):
            self.co2_convs.append(
                calculate_delta_co2_conv(self.path_filtered[i], percent=percent, verbose=verbose)
            )
        self.df_us = self.df_us.assign(delta_CO2_conv=self.co2_convs)

    def assign_target_values(self, methods: List[str], column: str, verbose: bool = False):
        """
        Assign target values to the DataFrame based on the specified methods.

        Args:
            methods (List[str]): List of target methods to calculate and include in the DataFrame.
            column (str): The column to calculate the target values for.
            verbose (bool): If True, print the calculated target values.

        Returns:
            None
        """

        for target in methods:
            target_values = []
            for path in self.path_filtered:
                target_value = calculate_target(path, column, target, verbose=verbose)
                target_values.append(target_value)
            self.df_us[f'{column}_{target}'] = target_values

    def calculate_init_slope(self):
        pass

    def export_sheet(self,
                     unique:bool=True,
                     which_target:str='delta_CO2_conv',
                     mute:bool=True):
        # export unique data set made by averaging targets and integrating
        if unique:
            # calculate each group's average target value
            df_unique = self.df_us[self.df_us["GroupID"] == 0]
            # ignore warning: A value is trying to be set on a copy of a slice from a DataFrame
            with warnings.catch_warnings(action="ignore"):
                for i, df_group in self.df_us[self.df_us["GroupID"] > 0].groupby("GroupID"):
                    # display(df_group)
                    mean = df_group[which_target].mean()
                    if not mute:
                        print(f'Group {i}: ')
                        print(f'mean: {mean:5.2f}')
                    df_integrated = df_group.iloc[0, :] # use the first row in the group
                    df_integrated.loc[which_target] = mean
                    df_integrated.loc['filename'] = f'group {i} integrated'
                    df_unique.loc[-1] = df_integrated # append an integrated row
                    df_unique.index = df_unique.index + 1
                    df_unique = df_unique.sort_index()
            df_unique = df_unique.sort_index(ascending=False)
            df_unique = df_unique.reset_index(drop=True)

            # export excel file
            dates = str(datetime.date.today()).rsplit('-')
            df_unique.to_excel(
                f'./{dates[0] + dates[1] + dates[2]}_sheet_for_ML_unique.xlsx',
                index=False
            )
            return df_unique
        # export data set with duplicate data
        else:
            dates = str(datetime.date.today()).rsplit('-')
            self.df_us.to_excel(
                f'./{dates[0] + dates[1] + dates[2]}_sheet_for_ML_duplicate.xlsx',
                index=False
            )
            return self.df_us

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

def plot_tos_data(
        path: str,
        column: str = None, x_max_plot: float = None, y_max_plot: float = None,
        temp_threshold: float = 3.5,
        plot_selected: bool = False,
        temp_max: float = False,
        show: bool = True,
        savefig: str = None,
        duration: float = 10.0,
        plt = plt
):
    """
        Plot Time-on-Stream (TOS) data with specific target and temperature.

        Args:
            path (str): Path to the Excel file containing the data.
            column (str, optional): Column name to plot. Defaults to None.
            x_max_plot (float, optional): Maximum value for the x-axis. Defaults to None.
            y_max_plot (float, optional): Maximum value for the y-axis. Defaults to None.
            plot_selected (bool, optional): Whether to plot the selected region. Defaults to False.
            show (bool, optional): Whether to display the plot. Defaults to True.
            savefig (str, optional): Path to save the figure. Defaults to None.
            duration (float, optional): Duration to calculate the final index. Defaults to 10.0.
            plt (module, optional): Matplotlib pyplot module. Defaults to plt.

        Returns:
            None
        """
    tos, temp, col_val, vertex_index, initial_index, final_index, selected_index = \
        extract_indices_target(path, column, duration, temp_threshold=temp_threshold)

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
        plt.axvline(x=tos[initial_index], color='g', linestyle='--')

    plt.xlabel('Time on stream (hrs)')
    plt.ylabel(column, c='g')
    if y_max_plot:
        plt.ylim(0, y_max_plot)
    if x_max_plot:
        plt.xlim(0, x_max_plot)

    # secondary axis for temperature
    axs_2nd = plt.twinx()
    l3 = axs_2nd.scatter(tos, temp, s=5, color='r', alpha=0.5)
    axs_2nd.set_ylabel('Temperature (C)', color='r')
    if temp_max:
        axs_2nd.set_ylim(0, temp_max)

    # legends & title
    if plot_selected:
        ls = [l1, l2, l3]
    else:
        ls = [l1, l3]
    labs = [l.get_label() for l in ls]
    plt.legend(ls, labs)#, loc=(0.55,0.73))

    if savefig:
        # plt.tight_layout()
        plt.savefig(savefig, bbox_inches='tight')
    if show:
        plt.show()

# deprecated
def calculate_delta_co2_conv(path: str, percent: bool = True, verbose: bool = False)->float:
    df = pd.read_excel(path, sheet_name='Data')
    df = df.fillna(value=0)  # some data set includes nan at the end of a column.

    # to deal with heterogeneous column names between expt groups
    ind_match = np.argwhere(np.char.find(list(df.columns), 'Time') + 1)[0][0]
    tos = df[df.columns[ind_match]]
    ind_match = np.argwhere(np.char.find(list(df.columns), 'Temperature') + 1)[0][0]
    temp = df[df.columns[ind_match]]
    ind_match = np.argwhere(np.char.find(list(df.columns), 'Conversion') + 1)[0][0]
    conv = df[df.columns[ind_match]]

    # defining const-temp region: delta temp
    temp_plus = np.roll(temp, -1)
    temp_plus[-1] = 0
    delta_temp = temp_plus - temp  # np.diff(temp, axis=0, prepend=0)
    delta_temp = delta_temp.abs()

    # vertex_index: `maximizer` + `slicing` (:100): to exclude spurious outlier when applying argmin
    vertex_index = np.argmax(conv[:100])

    # initial_index: `constant temp.` and `t>=0` and `t>=t_vertex`
    initial_index = \
        np.argwhere((delta_temp < 3.5) &  # 1: constant-temperature
                    (0 <= tos) &  # 2: 0<=t
                    (np.argwhere(conv > -999).reshape(-1) >= vertex_index)
                    # 3:index >= vertex index, among all indices
                    ).reshape(-1)[0]

    # final_index: initial_index + ~10 hrs
    final_index = np.argwhere(tos >= tos[initial_index] + 10).reshape(-1)[0]

    # selected_index: to be used for highlighting range in plot
    selected_index = np.argwhere(
        (tos >= tos[initial_index]) &
        (tos <= tos[final_index])
    ).reshape(-1)

    if percent:
        d_co2_conv = (conv[final_index] - conv[initial_index]) / conv[initial_index] * 100
        if verbose:
            print("percent delta conv (%): {:.2f}".format(d_co2_conv))
    else:
        d_co2_conv = conv[final_index] - conv[initial_index]
        if verbose:
            print("delta conv (%): {:.2f}".format(d_co2_conv))

    return d_co2_conv

# maybe the implementation for each `method` should be different depending on `column` ...
def calculate_target(
        path: str, column: str, method:str, verbose: bool = False,
        adjacency: float = 0.1,
        duration: float = 10,
        plot_slope: bool = False,
        plt = plt
)->float:
    """
    General version of target value calculator
    Args:
        path: path to individual GC excel file
        column:
        method: 'delta', 'initial value', 'final value', 'initial slope', 'final slope', 'overall slope', #'decaying rate'
        verbose:
        adjacency: used to treat fluctuation of measured y values. It indicates how close two points will be for initial and final slopes, not used for other methods.
        plot_slope: works with plot_tos_data()

    Returns:

    """
    df = pd.read_excel(path, sheet_name='Data')
    df = df.fillna(value=0)  # some data set includes nan at the end of a column.
    
    if column not in df.columns:
        raise ValueError(f"Keyword '{column}' is not included in {df.columns.tolist()}")

    tos, temp, col_val, vertex_index, initial_index, final_index, selected_index =\
        extract_indices_target(path, column, duration)

    # print(final_index)
    # print(selected_index)

    # calculate target value according to `method` argument
    methods = ['delta', 'initial value', 'final value', 'initial slope', 'final slope',
               'overall slope',] # 'decaying rate']
    if method not in methods:
        raise ValueError(f"Keyword '{method}' is not included in {methods}")

    if   method == 'delta':
        target = col_val[final_index] - col_val[initial_index]
    elif method == 'initial value':
        target = col_val[initial_index]
    elif method == 'final value':
        target = col_val[final_index]
    elif method == 'initial slope':
        # use the same initial_index
        try:
            # choosing final index which is close, in tos, to initial index
            final_index = np.argwhere(tos >= tos[initial_index] + adjacency).reshape(-1)[0]
        except Exception as e:
            print(e, f'has occurred while calculating `tos_ind_final` for {method}.')
        target = (col_val[final_index] - col_val[initial_index]) / (tos[final_index] - tos[initial_index])
    elif method == 'final slope':
        # use the same final_index
        try:
            # choosing initial index which is close, in tos, to final index
            initial_index = np.argwhere(tos <= tos[final_index] - adjacency).reshape(-1)[-1]
        except Exception as e:
            print(e, f'has occurred while calculating `tos_ind_final` for {method}.')
        target = (col_val[final_index] - col_val[initial_index]) / (tos[final_index] - tos[initial_index])
    elif method == 'overall slope':
        target = (col_val[final_index] - col_val[initial_index]) / (tos[final_index] - tos[initial_index])
    # elif method == 'decaying rate':
    #     print('not implemented yet')
    #     return

    # plot linear line for slope
    if method in ['initial slope', 'final slope', 'overall slope'] and plot_slope:
        _plot_linear_line(
            tos[initial_index], tos[final_index], col_val[initial_index], col_val[final_index],
            show=False, plt=plt
        )
        # plt.title(f'duration: {tos[final_index] - tos[initial_index]:.2f}')

    if verbose:
        print(f"{column}->{method}: {target:.2f}")
    return target


def extract_indices_target(path: str, column: str, duration: float = 10.0, temp_threshold: float = 3.5) -> Tuple[
    pd.Series, pd.Series, pd.Series, int, int, int, np.ndarray]:
    """
    Processes a DataFrame to extract specific columns and calculate indices.

    Args:
        path (str): The path to the Excel file.
        column (str): The column name to be processed.
        duration (float): The duration to calculate the final index.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, int, int, int, np.ndarray]:
            - Time on stream (tos) series.
            - Temperature series.
            - Column values series.
            - Vertex index.
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

    # Extract the 'Time', 'Temperature', and specified column values
    tos = df.filter(like='Time').iloc[:, 0]
    temp = df.filter(like='Temperature').iloc[:, 0]
    col_val = df.filter(like=column).iloc[:, 0]

    # Detect and remove duplicates based on 'tos' and 'temp'; sometimes identical pair of data points are included.
    df_unique = pd.DataFrame({'Time': tos, 'Temperature': temp, column: col_val}).drop_duplicates(subset=['Time', 'Temperature'])
    tos = df_unique['Time']
    temp = df_unique['Temperature']
    col_val = df_unique[column]

    # Calculate the absolute difference in temperature
    delta_temp = np.abs(np.roll(temp, -1) - temp)
    delta_temp[-1] = 0

    # Find the index of the maximum value in the specified column (first 100 rows)
    vertex_index = np.argmax(col_val[:100])

    # Find the initial index where the temperature is close to the reference temperature and time is non-negative
    assert (temp_threshold > 0), "temp_threshold should be positive."
    initial_index = np.argwhere((np.abs(temp-temp_ref) <= temp_threshold) & (tos >= 0)).reshape(-1)[0]
    test_val = np.abs(temp-temp_ref)
    test_val2 = (np.abs(temp-temp_ref) <= temp_threshold)
    # Find the final index based on the duration from the initial index
    final_index = np.argwhere(tos >= tos[initial_index] + duration).reshape(-1)[0]

    # Find the selected indices within the initial and final index range
    selected_index = np.arange(initial_index, final_index + 1)

    return tos, temp, col_val, vertex_index, initial_index, final_index, selected_index

def _plot_linear_line(t_init: float, t_final: float, y_init: float, y_final: float, show: bool = False, plt=plt):
    """ plot linear line connecting two points"""
    def linear_func(x, x1, x2, y1, y2):
        a = (y2 - y1) / (x2 - x1)
        b = y2 - a * x2
        return a * x + b, a

    x_plot = np.linspace(t_init-1, t_final+1, 100) # plot buffer of 1
    y_plot, slope = linear_func(x_plot, t_init,
                                t_final,
                                y_init,
                                y_final)

    plt.plot(x_plot, y_plot, c='k', alpha=0.5, label='two-point linear')
    plt.scatter([t_init, t_final],
                [y_init, y_final],
                color='red', edgecolors='gray'
                )
    plt.text(
        (t_init + t_final) / 2,
        (y_init + y_final) / 2,
        f'slope={slope:.2f}'
    )
    if show:
        plt.show()