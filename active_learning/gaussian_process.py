import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer #,MinMaxScaler
from sklearn.compose import make_column_selector as selector

import torch
from botorch.models import SingleTaskGP #, MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
# from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
# from gpytorch.constraints import GreaterThan, Interval

class GaussianProcess:
    def __init__(self):
        self.df = None
        self.columns = None
        self.df_Xtrain = None
        self.df_ytrain = None
        self.x_range_min = None
        self.x_range_max = None
        self.transformer_X = None
        self.transformer_y = None
        self.df_Xtrain_trans = None
        self.df_ytrain_trans = None
        self.tensor_Xtrain = None
        self.tensor_ytrain = None
        self.gp = None

    def preprocess_data_at_once(self,
                                path: str,
                                x_range_min: list = [300, 0.1, 0.005, 0],
                                x_range_max: list = [550, 1.0, 0.02, 1]):
        """
        Preprocess data by reading, constructing transformers, and transforming data.

        Args:
            path (str): Path to the Excel file.
            x_range_min (list): Minimum values for X scaling.
            x_range_max (list): Maximum values for X scaling.
        """
        self.read_data(path)
        self.construct_transformer(x_range_min=x_range_min, x_range_max=x_range_max)
        self.transform_data()
        self.convert_to_tensor()

    def read_data(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read the Excel file and return Xtrain and ytrain DataFrames.

        Args:
            path: path to the Excel file made by data.extract.DataForGP
        """
        self.df = pd.read_excel(path, header=0)
        self.df = self.df.drop(labels=['filename','experiment_date', 'location', 'GroupID'], axis=1)
        self.df.replace(
            {
                'WI': 0,
                'NP': 1
            },
            inplace=True
        )
        print(f'self.df.dtypes: {self.df.dtypes}')

        self.columns = self.df.columns

        self.df_Xtrain = self.df.drop(labels=self.columns[-1], axis=1)
        self.df_ytrain = self.df[[self.columns[-1]]]
        return self.df_Xtrain, self.df_ytrain

    def construct_transformer(self,
                              x_range_min: list = [300, 0.1, 0.005, 0],
                              x_range_max: list = [550, 1.0, 0.02, 1]):

        # Save x_range_min and x_range_max as attributes
        self.x_range_min = x_range_min
        self.x_range_max = x_range_max

        # Select numerical feature columns & define numerical transformer
        numerical_columns_selector = selector(dtype_exclude=object)
        numerical_features = numerical_columns_selector(self.df_Xtrain)
        print('numerical_features (selected): ', numerical_features)
        # Define numerical transformer
        numerical_transformer = FunctionTransformer(
            func=scaler_X,
            kw_args={'x_range_max': x_range_max,
                     'x_range_min': x_range_min},
            inverse_func=descaler_X,
            inv_kw_args={'x_range_max': x_range_max,
                         'x_range_min': x_range_min},
            validate=True,
            check_inverse=True
        )

        # Select categorical feature columns & define categorical transformer
        categorical_columns_selector = selector(dtype_include=object)
        categorical_features = categorical_columns_selector(self.df_Xtrain)
        print('categorical_features (selected): ', categorical_features)
        # Define categorical transformer
        categorical_transformer = Pipeline(
            steps=[("encoder", OneHotEncoder(handle_unknown="ignore")),]
        )

        # Combining numerical and categorical transformers
        transformer_X = ColumnTransformer(
            transformers=[
                ("numerical", numerical_transformer, numerical_features),
                ("categorical", categorical_transformer, categorical_features),
            ],
            # remainder="passthrough",
        )
        # Define y transformer
        transformer_y = StandardScaler()

        self.transformer_X = transformer_X
        self.transformer_y = transformer_y

    def transform_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.transformer_X.fit(self.df_Xtrain)
        self.transformer_y.fit(self.df_ytrain)
        self.df_Xtrain_trans = self.transformer_X.transform(self.df_Xtrain)
        self.df_ytrain_trans = self.transformer_y.transform(self.df_ytrain)
        return self.df_Xtrain_trans, self.df_ytrain_trans

    def convert_to_tensor(self):
        self.tensor_Xtrain = torch.tensor(self.df_Xtrain_trans)
        self.tensor_ytrain = torch.tensor(self.df_ytrain_trans)


    def train_gp(self, X: torch.tensor = None, y: torch.tensor = None, covar_module=None):
        # If X and y are not provided, use the transformed data within the class
        if X is None and y is None:
            X = self.tensor_Xtrain
            y = self.tensor_ytrain

        if covar_module is None:
            # Define the model (default kernel: MaternKernel)
            model = SingleTaskGP(
                train_X=X,
                train_Y=y,
                outcome_transform=Standardize(m=1)
            )
        else:
            model = SingleTaskGP(
                train_X=X,
                train_Y=y,
                outcome_transform=Standardize(m=1),
                covar_module=covar_module # ScaleKernel(RBFKernel(ard_num_dims=4)) / ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))
            )

        # For the case of mixed input features
        # gp = MixedSingleTaskGP(
        #     Xtrain_tensor,
        #     ytrain_tensor,
        #     cat_dims = [3, 4]
        # )

        # Define the marginal log likelihood (MLL)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # Fit the model
        fit_gpytorch_mll(mll)

        self.gp = model

        return self.gp

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
    for i in range(X.shape[1]): # loop over columns
      scaled[:,i] = scale(scaled[:,i], x_range_max[i], x_range_min[i])
    return scaled

def descaler_X(X, x_range_max, x_range_min):
    descaled = X.copy()
    for i in range(X.shape[1]): # loop over columns
      descaled[:,i] = descale(descaled[:,i], x_range_max[i], x_range_min[i])
    return descaled