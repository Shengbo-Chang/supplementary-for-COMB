from .base import Meta_OM, likelihood_none_zero
# import , likelihood_none_zero
import numpy as np


class OM(Meta_OM):
    
    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history

        previous_bid = bids_till_now[-2, :]
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_bid
        SIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        # SIM_single_step[np.where((-0.1 <= SIM_single_step) & (SIM_single_step <= 0))] = 0
        SIM_single_step[np.where(SIM_single_step <= 0)] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(SIM_single_step * SIM_single_step) / (2 * sigma * sigma))
        return likelihood_single

