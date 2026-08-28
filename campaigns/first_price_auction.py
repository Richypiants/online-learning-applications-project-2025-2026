import numpy as np

from campaigns.auction import Auction

class FirstPriceAuction(Auction):
    def __init__(self, qs):
        self.qs = qs

    def get_winners(self, bids):
        # In a first-price auction, the winner is the highest bidder
        public_values = self.qs * bids
        public_ranking = np.argsort(public_values)
        winner = public_ranking[-1]
        return winner

    def get_payments_per_click(self, winners, bids):
        # In a first-price auction, the winner pays their own bid
        payment = bids[winners]
        return payment

    

# class FirstPriceAuction(Auction):
#     def get_winner(self, bids):
#         # The winner is the bidder with the highest bid
#         return max(bids, key=bids.get)

#     def get_payment(self, winner, bids):
#         # In a first-price auction, the payment is equal to the winning bid
#         return bids[winner]