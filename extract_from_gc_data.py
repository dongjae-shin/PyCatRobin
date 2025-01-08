import uncertainty_sampling.extract as ex
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np

from glob import glob
import os

path = "/Users/djayshin/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/02 GC Experimental Data"
# path = "/Users/dongjae/Google Drive/Shared drives/Accelerating Innovations Team Drive/2. Research/8. Data/02 GC Experimental Data"

# keywords to exclude
exclude_keywords = [
    "0p0005", # data with too low Rh mass, likely to be inaccurate
    "(1)",    # data mistakenly uploaded twice
    "PercentLoading_Synthesis_MassLoading_Temperature_Date_Location" # example excel file
]

dataset = ex.DataForGP(path=path)
dataset.find_excel_files()
dataset.filter_excel_files(exclude_keywords=exclude_keywords)
dataset.construct_dataframe(extensive=False)
dataset.convert_measured_to_nominal(which_column="Rh_total_mass")
dataset.check_most_recent(buffer_recent=1)

display(dataset.df_us)
