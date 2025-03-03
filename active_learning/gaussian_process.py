import pandas as pd
import numpy as np

import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer

class GaussianProcess:
    def __init__(self):
        self.df = None
        self.df_Xtrain = None
        self.df_ytrain = None
        self.transformer_X = None
        self.transformer_y = None
        self.df_Xtrain_trans = None
        self.df_ytrain_trans = None

    def read_data(self, path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            path: path to the Excel file made by data.extract.DataForGP

        Returns:

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

        self.df_Xtrain = self.df.drop(labels=['CO2 Conversion (%)_initial value'], axis=1)
        self.df_ytrain = self.df[['CO2 Conversion (%)_initial value']]
        return self.df_Xtrain, self.df_ytrain

    def construct_transformer(self,
                              x_range_min: list = [300, 0.1, 0.005, 0],
                              x_range_max: list = [550, 1.0, 0.02, 1]):

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