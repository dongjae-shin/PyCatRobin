import active_learning.acquisition as aq

discrete_grid = aq.DiscreteGrid(x_range_min=[300, 0.1, 0.005, 0], x_range_max=[550, 1.0, 0.02, 1], x_step=[50, 0.1, 0.0025, 1])
discrete_grid.construct_grid()

# print(discrete_grid.list_grids)

