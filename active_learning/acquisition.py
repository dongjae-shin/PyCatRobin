from botorch.acquisition import PosteriorStandardDeviation, UpperConfidenceBound


def uncertainty_sampling_discrete(gp, n_candidates):
    # suggest n_candidates samples with the highest uncertainty on discrete grid

    # Instantiate a acquisition function
    US = PosteriorStandardDeviation(gp)

def upper_confidence_bound_discrete(n_candidates):
    # suggest n_candidates samples with the highest upper confidence bound on discrete grid

    UCB = UpperConfidenceBound(gp, beta=0.1)

def expected_improvement_discrete(n_candidates):
    # suggest n_candidates samples with the highest expected improvement on discrete grid
    pass