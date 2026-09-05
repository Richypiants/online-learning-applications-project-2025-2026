import numpy as np

from campaigns.auction import Auction

class FirstPriceAuction(Auction):
    def __init__(self, qs):
        self.qs = qs

    def get_winners(self, bids):
        public_values = self.qs * bids
        public_ranking = np.argsort(public_values)
        winner = public_ranking[-1]
        return winner

    def get_payments_per_click(self, winners, bids):
        payment = bids[winners]
        return payment
