import numpy as np
import scipy

# In an online learning setting, you don't know the actual distribution $\mathcal{D}$ of the highest competing bid. An UCB-like method needs to estimate
#  $\bar{f}(b)$ and $\bar{f}(b)$ in the optimization program, before solving it for $\gamma$. Using optimism means that, instead of using the Monte Carlo 
# average of the samples, we replace them with $f_{UCB}(b)$ and $c_{LCB}(b)$, estimatred through bandit feedback. (REWRITE BETTER, also they should have
# bars too in theory, no?)

class UCBLikeBidder():
    def __init__(self, B, T, valuation, feasible_bids, range=1.0):
        self.B = B  # budget
        self.T = T  # number of rounds (= number of auctions = number of users)
        self.RHO = B/T  # budget per round

        self.valuation = valuation  # true value of its own ad (assumed to be known)
        self.bids = feasible_bids[feasible_bids <= valuation]  # possible bids = action space of size K
        self.K = len(self.bids)  # cardinality of the action space (possible bids)

        self.t = 0  # current round
        self.a_t = None  # current action (as the index of the arm played, not the bid itself)
        self.range = range  # range for the confidence bounds (should be 1.0, since the rewards and costs are in [0,1]), but is it more like an std since it does not clip?

        self.avg_f = np.zeros(self.K)
        self.avg_c = np.zeros(self.K)
        self.N_pulls = np.zeros(self.K)

    def bid(self):
        if self.B < 1:  # if the budget is exhausted:
            return 0  # bid 0th bid = 0.0
        if self.t < self.K:     # in the first K rounds:
            self.a_t = self.t   # play each arm once (exploration phase)
        else:
            f_ucbs = self.avg_f + self.range*np.sqrt(2*np.log(self.t)/self.N_pulls)     # shouldn't this be clipped instead of being multiplied by range?
            c_lcbs = self.avg_c - self.range*np.sqrt(2*np.log(self.t)/self.N_pulls)     ## see below; also, used current round t instead of total T
            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            self.a_t = np.random.choice(self.K, p=gamma_t)
        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs):
        if np.sum(c_lcbs <= np.zeros(len(c_lcbs))):     # if any of the lower confidence bounds are negative, then the linear program is infeasible, so we just pick the arm with the highest upper confidence bound
            gamma = np.zeros(len(f_ucbs))               ## check if it is truly infeasible, or if it is just convenient since a negative cost means the arm is "free" to play
            gamma[np.argmax(f_ucbs)] = 1                ## also, it should never happen that the lower confidence bound is negative, since the cost is always positive, so maybe we should just clip it to 0 instead 
            return gamma
        c = -f_ucbs
        a = c_lcbs
        b = np.ones(self.K)
        res = scipy.optimize.linprog(c, A_ub=[a], b_ub=[self.RHO], A_eq=[b], b_eq=[1], bounds=(0,1))
        gamma = res.x
        return gamma

    def learn(self, f_t, c_t):
        self.N_pulls[self.a_t] += 1
        self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t])/self.N_pulls[self.a_t]
        self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t])/self.N_pulls[self.a_t]
        self.B -= c_t
        self.t += 1