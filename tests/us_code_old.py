import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading excel file
import os
from glob import glob

# path = "/Users/djayshin/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx"
path = "/Users/dongjae/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx"

df = pd.read_excel(path, header=0)
df = df.drop(labels=['filename','experiment_date', 'location', 'GroupID'], axis=1)
# df = df.drop(labels=['source','name'], axis=1)

df.replace(
    {
        'WI': 0,
        'NP': 1
    },
    inplace=True
)

print(df.dtypes)

df_Xtrain = df.drop(labels=['CO2 Conversion (%)_initial value'], axis=1)
df_ytrain = df[['CO2 Conversion (%)_initial value']]
# print('df_Xtrain.dtypes', df_Xtrain.dtypes)

columns = df.columns

# Converting dataframe into a numpy array
data = df.to_numpy().reshape(len(df),-1)
Xtrain = data[:,:-1]
ytrain = data[:,-1:]

print(f'Xtrain {Xtrain.shape}: \n{Xtrain}')
print(f'ytrain {ytrain.shape}: \n{ytrain}')

# @title Scaling & descaling functions and preprocessor
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

def scale(data, max, min):
    # Original(physical) space => Scaled space [0, 1]
    data_scaled = (data - min) / (max - min)
    return data_scaled

def descale(data, max, min):
    # Scaled space [0, 1] => Original(physical) space
    data_descaled = data * (max - min) + min
    return data_descaled

def scaler_X(X, x_range_max, x_range_min):
    scaled = X.copy()
    for i in range(X.shape[1]):
      scaled[:,i] = scale(scaled[:,i], x_range_max[i], x_range_min[i])
      # Temperature, Pressure, fCO2, wRh, mcatal
    return scaled

def descaler_X(X, x_range_max, x_range_min):
    descaled = X.copy()
    for i in range(X.shape[1]):
      descaled[:,i] = descale(descaled[:,i], x_range_max[i], x_range_min[i])
      # Temperature, Pressure, fCO2, wRh, mcatal
    return descaled



# select numerical feature columns & define numerical transformer
numerical_columns_selector = selector(dtype_exclude=object)
numerical_features = numerical_columns_selector(df_Xtrain)
print('numerical_features (selected): ', numerical_features)

# Construct custom scaler that scales both X and y
x_range_min = [300, 0.1, 0.005, 0]
x_range_max = [550, 1.0, 0.02, 1]
# x_range_min = [300, 0.1, 0.0005, 0]
# x_range_max = [550, 1.0, 0.0205, 1]

numerical_transformer=FunctionTransformer(
    func        =scaler_X,
    kw_args     ={'x_range_max': x_range_max,
                  'x_range_min': x_range_min},
    inverse_func=descaler_X,
    inv_kw_args ={'x_range_max': x_range_max,
                  'x_range_min': x_range_min},
    validate=True,
    check_inverse=True
)

# select categorical feature columns & define categorical transformer
categorical_columns_selector = selector(dtype_include=object)
categorical_features = categorical_columns_selector(df_Xtrain)
print('categorical_features (selected): ', categorical_features)

# categorical_transformer = Pipeline(
#     steps=[("encoder", OneHotEncoder(handle_unknown="ignore")),]
# )

# Combining together
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_transformer, numerical_features),
        # ("categorical", categorical_transformer, categorical_features),
    ],
    # remainder="passthrough",
)

preprocessor_y = StandardScaler()

preprocessor.fit(df_Xtrain)
preprocessor_y.fit(df_ytrain)
Xtrain_trans = preprocessor.transform(df_Xtrain)
ytrain_trans = preprocessor_y.transform(df_ytrain)

import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import LogNormalPrior

# making tensor for input data
Xtrain_tensor = torch.tensor(Xtrain_trans)
# making tensor for output data
ytrain_tensor = torch.tensor(ytrain_trans)

from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.optim import optimize_acqf, optimize_acqf_mixed

# Instantiate a GPR
# gp = MixedSingleTaskGP(
#     Xtrain_tensor,
#     ytrain_tensor,
#     cat_dims = [3, 4]
# )

# covar_module = ScaleKernel(RBFKernel(ard_num_dims=4))
# covar_module = ScaleKernel(MaternKernel(
#     nu=2.5,
#     ard_num_dims=4,
#     lengthscale_constraint=Interval(1e-04, 5.0)
# ))
#Note: it seems the constraints in gpytorch are not compatible with botorch's fit_gpytorch_mll

# covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))

# default covar_module is MaternKernel
gp = SingleTaskGP(
    Xtrain_tensor,
    ytrain_tensor,
    # covar_module=covar_module
)

# Optimize kernel parameter & noise
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Instantiate a acquisition function
US = PosteriorStandardDeviation(gp)
# cf. UCB = UpperConfidenceBound(gp, beta=0.1)

bounds = torch.stack([
    torch.tensor([0., 0., 0., 0.]),
    torch.tensor([1., 1., 1., 1.])
    ])

candidate, acq_value = optimize_acqf(
    US, bounds=bounds, q=1, num_restarts=10, raw_samples=20,
)

print(f'candidate: {candidate}\nacq_value: {acq_value}\nUS.forward(candidate): {US.forward(candidate)}')


# -------
print(gp.covar_module)

for param_name, param in gp.named_parameters():
    if 'lengthscale' in param_name:
        print(f'Parameter name: {param_name:42} \nvalue = {param}')

#-------
# unlabeled data set for all the possible combinations of input features
temp_max = 550
temp_min = 300
temp_step = 50
n_temp_grid = int((temp_max - temp_min) / temp_step + 1)

m_rh_max = 0.02
m_rh_min = 0.005
m_rh_step = 0.0025
n_m_rh_grid = int((m_rh_max - m_rh_min) / m_rh_step + 1)

w_rh_max = 1.0
w_rh_min = 0.1
w_rh_step = 0.1
n_w_rh_grid = int((w_rh_max - w_rh_min) / w_rh_step + 1)

temp = np.linspace(temp_min, temp_max, n_temp_grid)
m_rh = np.linspace(m_rh_min, m_rh_max, n_m_rh_grid)
w_rh = np.linspace(w_rh_min, w_rh_max, n_w_rh_grid)

# print(f'temp:\n{temp}\nm_rh:\n{m_rh}\nw_rh:\n{w_rh}\n')
n_total = len(temp)*len(m_rh)*len(w_rh)
print(f'{n_total} combinations are possible.')

# all the combinations of T, w_Rh, and m_Rh
X_discrete = np.array(
    [[i, j, k] for i in temp for j in w_rh for k in m_rh]
)
X_discrete[:,0] = X_discrete[:,0].round(decimals=0)
X_discrete[:,1] = X_discrete[:,1].round(decimals=1)
X_discrete[:,2] = X_discrete[:,2].round(decimals=4)

# unlabeled data set for 'colloidal'
X_discrete_colloidal = pd.DataFrame(
    np.concatenate(
        (X_discrete, np.full((n_total,1), fill_value=1)),
        axis=1
        )
)

# unlabeled data set for 'impregnation'
X_discrete_impregnation = pd.DataFrame(
    np.concatenate(
        (X_discrete, np.full((n_total,1), fill_value=0)),
        axis=1
        )
)

print(X_discrete_colloidal.shape, X_discrete_impregnation.shape)

# for WI synthesis

# pd.set_option('display.max_rows', 500)
pd.reset_option('display.max_rows')

# number of conditions to be suggested
n_candidates = 5

# adding column information for preprocessor
X_discrete_impregnation.columns=columns.drop(labels='CO2 Conversion (%)_initial value')

# scaling and making tensor
X_discrete_impregnation_trans = torch.tensor(
    preprocessor.transform(X_discrete_impregnation)
)


# calculate posterior standard deviation for all the possible feature vectors
std = US.forward(
    X_discrete_impregnation_trans.reshape(
        len(X_discrete_impregnation_trans),1,4)
).detach().numpy()

top_ids = np.argsort(-std)[:n_candidates] # negativity: sort in reverse order

# show top 'n_candidates' uncertain conditions
print(
    X_discrete_impregnation.join(
        pd.DataFrame(std, columns=['std. dev.']) # append uncertainty info.
    ).iloc[top_ids,:]
)