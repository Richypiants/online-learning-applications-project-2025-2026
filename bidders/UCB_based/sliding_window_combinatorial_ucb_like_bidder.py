import numpy as np

from bidders.UCB_based.combinatorial_ucb_like_bidder import CombinatorialUCBLikeBidder

class SlidingWindowCombinatorialUCBLikeBidder(CombinatorialUCBLikeBidder):
    '''Combinatorial-UCB with a sliding window over past observations, for slightly non-stationary environments.'''

    def __init__(self, B, T, valuations, environment, window_size=None):
        super().__init__(B, T, valuations, environment)

        # TODO: check the default window size choice: tau = O(sqrt(T log T)), since it should be divided by the # changes or by T if unknown
        self.window_size = int(window_size if window_size is not None else min(T, np.ceil(2 * np.sqrt(np.log(T)))))      # theoretical choice: tau = O(sqrt(T log T))
        # sliding-window caches: one entry per round, per-campaign utility/cost vectors with NaN on unplayed arms (like the SW-UCB cache)
        self.f_cache = np.full((self.window_size, self.N_CAMPAIGNS, self.K), np.nan)
        self.c_cache = np.full((self.window_size, self.N_CAMPAIGNS, self.K), np.nan)

    def learn(self, f_t, c_t, m_t=None):
        super().learn(f_t, c_t)

        f_vec = np.full((self.N_CAMPAIGNS, self.K), np.nan)
        f_vec[np.arange(self.N_CAMPAIGNS), self.a_t] = f_t
        c_vec = np.full((self.N_CAMPAIGNS, self.K), np.nan)
        c_vec[np.arange(self.N_CAMPAIGNS), self.a_t] = c_t

        # remove oldest observation and add the new one
        self.f_cache = np.delete(self.f_cache, 0, axis=0)
        self.c_cache = np.delete(self.c_cache, 0, axis=0)
        self.f_cache = np.vstack((self.f_cache, f_vec[None, :, :]))
        self.c_cache = np.vstack((self.c_cache, c_vec[None, :, :]))

    def _confidence_bounds(self):
        # aggregated quantities restricted to the last window_size rounds
        N_pulls_last_W = self.window_size - np.isnan(self.f_cache).sum(axis=0)          # shape (N_CAMPAIGNS, K)
        avg_f = np.nanmean(self.f_cache, axis=0)
        avg_c = np.nanmean(self.c_cache, axis=0)

        mask = N_pulls_last_W != 0
        radius = np.full_like(N_pulls_last_W, self.range, dtype=float)      # TODO: reflect changes in base_ucb, or find a way to make them the same
        radius[mask] *= np.sqrt(2 * np.log(min(self.t, self.window_size)) / N_pulls_last_W[mask])
        f_ucbs = np.clip(np.where(mask, avg_f, 0) + radius, None, 1)
        c_lcbs = np.clip(np.where(mask, avg_c, 0) - radius, 0, None)
        return f_ucbs, c_lcbs

    # TODO: I am afraid that the NaNs might break the LP? Maybe when no samples put the initialization value from base UCB instead?
    # In standard UCB with argmax and not the probs distribution from LP, NaN was fine because it was selected from the argmax (I guess)...
    # Can we do something similar for the LP too instead, so as to force exploration?
