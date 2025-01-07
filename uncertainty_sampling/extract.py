import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple
from glob import glob


# Extract input vector for each excel file (241115)
def get_input_vector(excel_path: str = None,
                     extensive: bool = False,
                     mute: bool = True) -> List:
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


def plot_tos_data(path, keyword_to_plot=None, x_max_plot=None, y_max_plot=100):
    df = pd.read_excel(path, sheet_name='Data')

    # getting columns dealing with heterogeneous column names between expt groups
    ind_match = np.argwhere(np.char.find(list(df.columns), 'Time') + 1)[0][0]
    tos = df[df.columns[ind_match]]
    ind_match = np.argwhere(np.char.find(list(df.columns), 'Temperature') + 1)[0][0]
    temp = df[df.columns[ind_match]]
    ind_match_prop = np.argwhere(np.char.find(list(df.columns), keyword_to_plot) + 1)[0][0]
    prop = df[df.columns[ind_match_prop]]

    # defining const-temp region: delta temp
    temp_plus = np.roll(temp, -1)
    temp_plus[-1] = 0
    delta_temp = temp_plus - temp  # np.diff(temp, axis=0, prepend=0)
    delta_temp = delta_temp.abs()

    # defining t_final = t_vertex + 10 hrs
    tos_ind_vertex = np.argmax(prop[:100])
    try:
        tos_ind_10hrs = np.argwhere(tos >= tos[tos_ind_vertex] + 10).reshape(-1)[0]
    except Exception as e:
        print(e, f'has occurred while calculating `tos_ind_10hrs`.')
        # continue

    # Plot
    l1 = plt.scatter(tos, prop,
                     color=[0.5, 1.0, 0.5, 1.0], s=5, label=df.columns[ind_match_prop])  # whole profile
    # l2 = plt.scatter(tos[tos_ind_selected], prop[tos_ind_selected],
    #                  color='g', s=5, label='selected')
    plt.xlabel('Time on stream (hrs)')
    plt.ylabel(df.columns[ind_match_prop], c='g')
    plt.ylim(0, y_max_plot)
    if x_max_plot:
        plt.xlim(0, x_max_plot)

    # secondary axis
    axs_2nd = plt.twinx()
    l3 = axs_2nd.scatter(tos, temp, s=5, color='r')
    axs_2nd.set_ylabel('Temperature (C)', color='r')
    axs_2nd.set_ylim(0, 520)

    # # legends & title
    # ls = [l1, l3] #l2
    # labs = [l.get_label() for l in ls]
    # plt.legend(ls, labs)#, loc=(0.55,0.73))

    # plt.savefig(path_imgs+"/{}_fig_{}.png".format(prefix, num))
    plt.show()


def calculate_delta_co2_conv(path, percent=True, mute=False):
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

    # defining t_final = t_vertex + 10 hrs
    tos_ind_vertex = np.argmax(conv[:100])
    try:
        tos_ind_10hrs = np.argwhere(tos >= tos[tos_ind_vertex] + 10).reshape(-1)[0]
    except Exception as e:
        print(e, f'has occurred while calculating `tos_ind_10hrs`.')
        # continue

    # select tos range:
    tos_ind_selected = \
        np.argwhere((delta_temp < 3.5) &  # 1: constant-temperature
                    (tos <= tos[tos_ind_10hrs]) & (0 <= tos) &  # 2: 0<=t<=t_vert+10 hrs
                    (np.argwhere(conv > -999).reshape(-1) \
                     >= np.argmax(conv[:100]))
                    # 3: after point with strongest negative change of slope
                    # slicing (:100) when argmin to exclude spurious outlier
                    ).reshape(-1)

    # locating the max & min points
    initial_index = tos_ind_selected[0]
    final_index = tos_ind_selected[-1]

    if percent:
        d_co2_conv = (conv[final_index] - conv[initial_index]) / conv[initial_index] * 100
        if not mute:
            print("percent delta conv (%): {:.2f}".format(d_co2_conv))
    else:
        d_co2_conv = conv[final_index] - conv[initial_index]
        if not mute:
            print("delta conv (%): {:.2f}".format(d_co2_conv))

    return d_co2_conv