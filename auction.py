from abc import ABC, abstractmethod

class Auction(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def get_winners(self, bids):
        pass

    @abstractmethod
    def get_payments_per_click(self, winners, bids):
        pass

    def round(self, bids):
        winners = self.get_winners(bids)
        payments = self.get_payments_per_click(winners, bids)
        return winners, payments



# class Auction(Environment):
#     def get_winner(self):
#         raise NotImplementedError

#     def get_payment(self):
#         raise NotImplementedError

#     def round(self, bids):
#         winner = self.get_winner()
#         payment = self.get_payment()
#         return winner, payment