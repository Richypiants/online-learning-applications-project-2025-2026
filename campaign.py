from first_price_auction import FirstPriceAuction

class Campaign(FirstPriceAuction):
    def __init__(self, ad_qualities):
        super().__init__(ad_qualities)      # consider removing ad_qualities from the constructor of FirstPriceAuction and just passing it to the methods that need it, since the auction itself isn't really defined by the qualities
        self.N_ADVERTISERS = len(ad_qualities)
        self.N_COMPETITORS = self.N_ADVERTISERS - 1

    def __str__(self):
        return (
            "-- CAMPAIGN: --\n"
            f"Ad qualities: {self.qs}\n"
            f"Number of advertisers: {self.N_ADVERTISERS}\n"
            f"Number of competitors: {self.N_COMPETITORS}"
        )

    __repr__ = __str__