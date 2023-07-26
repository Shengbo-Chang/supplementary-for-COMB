from .base import Meta_OM, likelihood_none_zero
import numpy as np


class OM(Meta_OM):
    compact_version = 'Moving'
    
    @likelihood_none_zero
    def _likelihood_func(self, ufuns):
        
        sigma = self.SIGMA
        # print(partial_ufuns)
        num_ufuns = ufuns.shape[0]
        bids_till_now = self.onehot_bids_history #one-hot coded
        num_bids = bids_till_now.shape[0]
        bids_till_now_expand = bids_till_now.repeat(num_ufuns, axis = 0).reshape([num_bids, num_ufuns, -1])
        bids_utilities = (bids_till_now_expand * ufuns).sum(axis = -1) #axis-0: bids; axis-1: utility of different hyppothesis
        ufuns_mean = bids_utilities.mean(axis = 0)
        ufuns_diff = bids_utilities - ufuns_mean.reshape([1, -1])
        time_mean = self.time_sequence.mean()
        time_diff = (self.time_sequence - time_mean).reshape([-1, 1])

        BD = np.square(ufuns_diff).sum(axis = 0)
        
        B1 = (ufuns_diff * time_diff).sum(axis = 0) / BD

        B1[np.where(BD == 0)] = 0

        B0 = ufuns_mean - B1 * time_mean
        B0 = B0 - 0.95
        
        B1[np.where(B1 <= 0)] = 0
        B0[np.where(B0 >= 0)] = 0

        likelihood = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(B1**2 + B0**2) / (2 * sigma * sigma))
        return likelihood