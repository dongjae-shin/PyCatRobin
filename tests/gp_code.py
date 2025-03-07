import os
import active_learning.gaussian_process as gpc

home_dir = os.path.expanduser("~")
path = (home_dir +
        "/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx")
# path = "/Users/dongjae/Dropbox/0.Dongjae/04.SUNCAT@SLAC,Standford(2402~)/231128_research/240602_ML_codes/CatNaviGATE/tests/20250228_sheet_for_ML_unique.xlsx"

gp1 = gpc.GaussianProcess()
gp1.preprocess_data_at_once(
    path=path,
    x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1]
    )
gp1.train_gp()

# print the attributes
print('gp1.tensor_Xtrain: ', gp1.tensor_Xtrain)
print('gp1.tensor_ytrain: ', gp1.tensor_ytrain)
print('gp1.gp: ', gp1.gp)
print('gp1.gp.kernel: ', gp1.gp.covar_module)