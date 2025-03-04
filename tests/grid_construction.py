import active_learning.acquisition as aq
import active_learning.gaussian_process as gpc

# path = "/Users/djayshin/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx"
path = "/Users/dongjae/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx"

gp1 = gpc.GaussianProcess()
gp1.preprocess_data_at_once(
    path=path,
    x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1]
    )
gp1.train_gp()


discrete_grid = aq.DiscreteGrid(x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1], x_step=[50, 0.1, 0.0025, 1])
discrete_grid.construct_grid()

# print(discrete_grid.list_grids)

discrete_grid.uncertainty_sampling_discrete(gp1.gp, gp1.transformer_X, synth_method='WI', n_candidates=5)