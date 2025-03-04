import pandas as pd
import numpy as np
from botorch.acquisition import PosteriorStandardDeviation, UpperConfidenceBound
# from botorch.optim import optimize_acqf, optimize_acqf_mixed

class DiscreteGrid:
    def __init__(self, x_range_min: list, x_range_max: list, x_step: list):
        self.x_range_min = x_range_min
        self.x_range_max = x_range_max
        self.x_step = x_step
        self.list_grids = None
        self.n_grid_total = None
        self.X_discrete_wi = None
        self.X_discrete_np = None

    def construct_grid(self):
        # unlabeled data set for all the possible combinations of input features
        list_grids = []
        # Create a grid for each input feature
        for i in range(len(self.x_step)):
            n_grid = int((self.x_range_max[i] - self.x_range_min[i]) / self.x_step[i] + 1)
            grid = np.linspace(self.x_range_min[i], self.x_range_max[i], n_grid)
            list_grids.append(grid)
        self.list_grids = list_grids

        # Calculate the total number of combinations
        self.n_grid_total = np.prod([len(grid) for grid in self.list_grids])
        print(f'{self.n_grid_total} combinations are possible in the constructed grid.')

        # Create a meshgrid of all combinations
        grid_combinations = np.meshgrid(*self.list_grids, indexing='ij')
        # Reshape and stack to get all combinations as rows
        X_discrete = np.column_stack([g.flatten() for g in grid_combinations])
        X_discrete = pd.DataFrame(X_discrete)

        # Unlabeled data set for WI and NP
        self.X_discrete_wi = X_discrete[X_discrete.iloc[:, -1] == 0].reset_index(drop=True)
        self.X_discrete_np = X_discrete[X_discrete.iloc[:, -1] == 1].reset_index(drop=True)

    def uncertainty_sampling_discrete(self, gp, synth_method: str, n_candidates: int = 5):
        # suggest n_candidates samples with the highest uncertainty on discrete grid

        # Instantiate a acquisition function
        US = PosteriorStandardDeviation(gp)

    def upper_confidence_bound_discrete(self, gp, n_candidates):
        # suggest n_candidates samples with the highest upper confidence bound on discrete grid

        UCB = UpperConfidenceBound(gp, beta=0.1)

    def expected_improvement_discrete(self, gp, n_candidates):
        # suggest n_candidates samples with the highest expected improvement on discrete grid
        pass