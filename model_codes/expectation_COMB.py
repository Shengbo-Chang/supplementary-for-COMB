from .base import Meta_OM, likelihood_none_zero
import numpy as np


class OM(Meta_OM):
    
    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        sigma = self.SIGMA
        bids_till_now = self.onehot_bids_history
        previous_expected_bid = bids_till_now[:-1, :].mean(axis=0)
        newest_bid = bids_till_now[-1, :]

        diff_bid = newest_bid - previous_expected_bid
        CIM_single_step = (ufuns * diff_bid).sum(axis = 1)

        CIM_single_step[np.where(CIM_single_step <= 0)] = 0
        likelihood_single = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(CIM_single_step ** 2) / (2 * sigma * sigma))
        return likelihood_single
