import os

import active_learning.acquisition as aq
import active_learning.gaussian_process as gpc
from matplotlib.pyplot import tick_params

# Define the home directory and path to data
home_dir = os.path.expanduser("~")

# initial CO2 conversion
path = (home_dir +
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx")
# new target value
# path = (home_dir +
#         "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250319_sheet_for_ML_unique.xlsx")



gp1 = gpc.GaussianProcess()
gp1.preprocess_data_at_once(
    path=path,
    x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1]
    )
gp1.train_gp()


discrete_grid = aq.DiscreteGrid(
    x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1],
    x_step=[50, 0.1, 0.0025, 1],
    gp=gp1
)

discrete_grid.construct_grid()

print(discrete_grid.list_grids)

# discrete_grid.uncertainty_sampling_discrete(gp1.gp, gp1.transformer_X, synth_method='WI', n_candidates=5) # need to fix this (See Bookmark notes)

# function that calculates uncertainty & mean in original scale
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

@np.vectorize
def wrapper_std_colloidal(wrh, mrh, temperature=500):
    # dataframing for preprocessor
    X = pd.DataFrame(
        np.array([[temperature, wrh, mrh, 1]])
    )

    # adding column information for preprocessor
    X.columns = gp1.columns.drop(labels=gp1.columns[-1])

    # scaling and tensorizing
    X_tensor = torch.tensor(
        gp1.transformer_X.transform(X)
    )
    US = aq.PosteriorStandardDeviation(gp1.gp)

    return US.forward(
        X_tensor.reshape(len(X_tensor), 1, 4)
    ).detach().numpy()

def plot_grid(x_points, y_points):
    X, Y = np.meshgrid(x_points, y_points)
    plt.scatter(
        X, Y,
        s=0.1, c='k', label='grid'
    )

def plot_train_data(Xtrain: pd.DataFrame):
    plt.scatter(
        Xtrain['Rh_weight_loading'], Xtrain['Rh_total_mass'],
        s=17.0, c='r', marker='D', label='train data'
    )

# print(f'std: {wrapper_std_colloidal(0.1, 0.005, 500)}')

# # grid for larger view
# w_rh_axis = np.linspace(0, 6, 50)
# m_rh_axis = np.linspace(0, 0.05, 50)
# W_rh, M_rh = np.meshgrid(w_rh_axis, m_rh_axis)

# grid for smaller view
w_rh_axis = np.linspace(discrete_grid.x_range_min[1], discrete_grid.x_range_max[1], 50)
m_rh_axis = np.linspace(discrete_grid.x_range_min[2], discrete_grid.x_range_max[2], 50)
W_rh, M_rh = np.meshgrid(w_rh_axis, m_rh_axis)

# grid for plotting grid points
w_rh_axis_grid = discrete_grid.list_grids[1]
m_rh_axis_grid = discrete_grid.list_grids[2]
W_rh_grid, M_rh_grid = np.meshgrid(w_rh_axis_grid, m_rh_axis_grid)

# visualize the uncertainty distribution on the grid
fig = plt.figure(figsize=(6, 5.8))
# set the font sizes for the plot
plt.rcParams.update({'font.size': 15})

sigma_max = 1.1
levels = np.linspace(0, sigma_max, 32)  # for std
# levels = np.linspace(-2,2,11) # mean

# for temperature in [300, 350, 400, 450, 500, 550]:
for temperature in [500]:
    cmap = plt.contourf(
        W_rh,
        M_rh,
        wrapper_std_colloidal(W_rh, M_rh, temperature=temperature),
        # wrapper_mean_colloidal(W_rh, M_rh, temperature=temperature),
        extend='max',
        levels=levels
    )

    cbar = plt.colorbar(
        cmap,
        label='posterior std. dev.',
        ticks=np.linspace(0, sigma_max, int(sigma_max / 0.1 + 1))
    )
    cbar.ax.tick_params(labelsize=12)

    plot_grid(W_rh_grid, M_rh_grid)
    plot_train_data(gp1.df_Xtrain[(gp1.df_Xtrain['synth_method'] == 1) & (gp1.df_Xtrain['reaction_temp'] == temperature)])

    plt.xlim(w_rh_axis.min(), w_rh_axis.max())
    plt.ylim(m_rh_axis.min(), m_rh_axis.max())
    # plt.xlim(0.1, 6.0)
    # plt.ylim(0.000, 0.05)
    plt.xlabel('Rh weight loading (wt%)')
    plt.ylabel('Rh mass in reactor (mg)')
    # set ticklabel font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'temperature: {temperature} ℃')
    plt.legend(bbox_to_anchor=(1.00, -0.18))
    plt.tight_layout()
    plt.show()
